import ast
import base64
import json
import logging
import logging.handlers
import os
import sys
import time
from typing import List

import openai
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index import (GPTListIndex, LangchainEmbedding, PromptHelper,
                         QueryBundle, ServiceContext, SimpleDirectoryReader,
                         StorageContext, VectorStoreIndex,
                         load_index_from_storage, set_global_service_context)
from llama_index.indices.base import BaseIndex
from llama_index.indices.composability import ComposableGraph
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.llm_predictor import LLMPredictor
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.response_synthesizers import get_response_synthesizer
from llama_index.response_synthesizers.type import ResponseMode
from llama_index.retrievers import VectorIndexRetriever
from llama_index.storage.storage_context import DEFAULT_PERSIST_DIR
from llama_index.vector_stores import SimpleVectorStore
from llama_index.vector_stores.types import VectorStore
from llama_index.callbacks import CallbackManager, LlamaDebugHandler, CBEventType

from app.prompts.qna import (get_chat_prompt_template, get_prompt_template,
                             get_refined_prompt)

load_dotenv()

app = Flask(__name__)
CORS(app)

#storage_account_name = os.environ["STORAGE_ACCNT_NAME"]
key_vault_name          = os.environ["KEY_VAULT_NAME"]
openai_endpoint_name    = os.environ["OPENAI_ENDPOINT_NAME"]
#deployment_names         = [name.strip() for name in str.split(os.environ["OPENAI_DEPLOYMENT_NAME"], ",")]

models = {
    "gpt-4": {"name": "gpt-4", "context_window": 8192, "index": {} },
    "gpt-35-turbo-16k": {"name": "gpt-35-turbo-16k", "context_window": 16384, "index": {} }
}

#openai_api_version      = "2023-03-15-preview" # this may change in the future
openai_api_version      = "2023-07-01-preview"

kv_uri              = f"https://{key_vault_name}.vault.azure.net"
azure_openai_uri    = f"https://{openai_endpoint_name}.openai.azure.com"

credential  = DefaultAzureCredential()
client      = SecretClient(vault_url=kv_uri, credential=credential)

openai.api_type    = os.environ["OPENAI_API_TYPE"]    = 'azure'
openai.api_base    = os.environ["OPENAI_API_BASE"]    = azure_openai_uri
openai.api_version = os.environ["OPENAI_API_VERSION"] = openai_api_version
azure_openai_key   = client.get_secret(os.getenv("OPENAI_KEY_NAME", "AzureOpenAIKey")).value
if azure_openai_key is not None:
    openai.api_key = os.environ["OPENAI_API_KEY"] = azure_openai_key

'''
Keeping the chat history for the context (what is sent to ChatGPT 3.5) to 5, seems big enough. 

And we will keep the embedding chat history to a minimum (and we also restrict it to outputs only ..)
'''
_max_history = 5
_max_embeddings_history = 1

_default_index_name = os.getenv("INDEX_NAME", "2023-07-19")


'''
Bootstrap function to pre-load the vector index(ices)
'''
def _bootstrapIndex():
    for model in models.values():
        start = time.time()
        print("Loading up the index: {}... (with model: {})".format(_default_index_name, model))
        set_global_service_context(_get_service_context(model["name"], model["context_window"]))
        # indices are pre-loaded with the same model they will be queried with
        model["index"] = _get_index(index_name=_default_index_name)
        end = time.time()
        print("Took {} seconds to load.".format(end-start))
    set_global_service_context(None) # want to enforce people to set it properly

@app.route("/health", methods=["GET"])
def health(): 
    return jsonify({"msg":"Healthy"})

@app.route("/query", methods=["POST"])
def query():

    query = ""
    k = 2 # default
    temperature = 0.7 # default
    body = request.get_json(force=True)
    lang = "en"
    pretty = False # wether or not to pretty print medatada, used for the MS Teams chatbot ..
    history = {'inputs': [], 'outputs': []}
    response_mode = ResponseMode.TREE_SUMMARIZE
    model_data = {}
    num_output = 800

    if "query" not in body:
        return jsonify({"error":"Request body must contain a query."}), 400
    else:
        query = body["query"]

    if "index" in body:
        index_name = body["index"]
           
    if "temp" in body:
        temperature = float(body["temp"])

    if "k" in body:
        k = int(body["k"])

    if "num_output" in body:
        num_output = int(body["num_output"])

    if "pretty" in body:
        pretty = bool(body["pretty"])

    if "lang" in body:
        if str(body["lang"]) == "fr":
            lang = "fr"
    if "response_mode" in body:
        match body["response_mode"]: # else default to TREE_SUMARIZE
            case ResponseMode.REFINE.value:
                response_mode = ResponseMode.REFINE
            case ResponseMode.COMPACT.value:
                response_mode = ResponseMode.COMPACT
            case ResponseMode.SIMPLE_SUMMARIZE.value:
                response_mode = ResponseMode.SIMPLE_SUMMARIZE
            case ResponseMode.ACCUMULATE.value:
                response_mode = ResponseMode.ACCUMULATE
            case ResponseMode.COMPACT_ACCUMULATE.value:
                response_mode = ResponseMode.COMPACT_ACCUMULATE
        print("using {} response mode".format(response_mode))
    
    if "chat_history" in body:
        # try and conver the chat history into a map
        try:
            encoded = body["chat_history"]
            decoded = base64.b64decode(encoded)
            history = json.loads(decoded)
            # validate we have an input and output key
            if 'inputs' not in history or 'outputs' not in history:
                raise Exception("Unable to properly parse chat_history")
        except:
            print("Unable to convert chat_history to a proper map that contains input and outputs..")
            history = {'inputs': [], 'outputs': []}

    if "model" in body:
        model = body["model"]
        print("Loading up data for model: {}".format(model))
        try: 
            model_data = models[model]
        except KeyError as ex:
            print("unable to load model, will use default", ex)
            return jsonify({"error": str(ex)}), 500
        
    start = time.time()
    service_context = _get_service_context(model_data["name"], model_data["context_window"], temperature, num_output)
    #set_global_service_context(service_context) # remove?
    end = time.time()
    print("Service context set (took {} seconds). Using model: {}".format(end-start, model_data["name"]))
    
    query_bundle = query

    #prompt building, and query bundle
    if _has_history(history):
        print("using chat history for prompt and query")
        text_qa_template=get_chat_prompt_template(lang, _history_as_str(history))
        query_bundle = QueryBundle(
            query,
            custom_embedding_strs=_get_history_embeddings(history)
        )
    else:
        text_qa_template=get_prompt_template(lang)

    retriever = VectorIndexRetriever(index=model_data["index"], similarity_top_k=k) # type: ignore

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(response_mode=response_mode, service_context=service_context)

    # assemble query engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        service_context=service_context,
        text_qa_template=text_qa_template,
        refine_template=get_refined_prompt(lang),
        verbose=True
    )

    response = query_engine.query(query_bundle)
    metadata = _response_metadata(response, pretty)

    history = _update_chat_history(history, query, str(response), lang)
    history_bytes = json.dumps(history).encode('utf-8')
    history_enc = base64.b64encode(history_bytes)

    r = {
            'query': query,
            'answer': str(response),
            'metadata': metadata,
            'chat_history': str(history_enc, 'utf-8')
        }

    #for data in r.keys():
    #    print(f"{data}: {r[data]}")

    return jsonify(r)

"""

Loads a Vector Index from the local filesystem

"""    
def _get_index(index_name: str, storage_location: str = DEFAULT_PERSIST_DIR):
    storage_context = StorageContext.from_defaults(persist_dir=os.path.join(storage_location, index_name))
    # index needs to be recreated its missing metadata for this code to work properly.
    #vector_store = SimpleVectorStore.from_persist_dir(persist_dir=os.path.join(storage_location, index_name))
    #return VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return load_index_from_storage(storage_context)

def _get_service_context(model: str, context_window: int, temperature: float = 0.7, num_output: int = 800) -> "ServiceContext":
    chunk_overlap_ratio = 0.1 # overlap for each token fragment

    # using same dep as model name because of an older bug in langchains lib (now fixed I believe)
    llm = _get_llm(model, temperature)

    llm_predictor = _get_llm_predictor(llm)

    prompt_helper = PromptHelper(context_window=context_window, num_output=num_output, chunk_overlap_ratio=chunk_overlap_ratio,)

    # limit is chunk size 1 atm
    embedding_llm = LangchainEmbedding(
        OpenAIEmbeddings(
            model="text-embedding-ada-002", 
            deployment="text-embedding-ada-002", 
            openai_api_key=openai.api_key,
            openai_api_base=openai.api_base,
            openai_api_type=openai.api_type,
            openai_api_version=openai.api_version),
            embed_batch_size=1)
    
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])

    return ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper, embed_model=embedding_llm, callback_manager=callback_manager)

def _get_llm(model: str, temperature: float = 0.7):
    return AzureChatOpenAI(model=model, 
                           deployment_name=model,
                           temperature=temperature,)

def _get_llm_predictor(llm) -> LLMPredictor:
    return LLMPredictor(llm=llm,)

def _response_metadata(response: RESPONSE_TYPE, pretty: bool):
    metadata = {}
    if not pretty:
        scores = {}
        for node in response.source_nodes:
            print("docid " + str(node.node.ref_doc_id) + " and here is the current node score: " + str(node.score))
            if node.node.ref_doc_id not in metadata:
                scores[node.node.ref_doc_id] = [node.score]
                metadata[node.node.ref_doc_id] = node.node.extra_info
                metadata[node.node.ref_doc_id]["text"] = [node.text]
            else:
                scores[node.node.ref_doc_id].append(node.score)
                metadata[node.node.ref_doc_id]["text"].append(node.text)

        [v.update({"node_scores": scores[k]}) for k, v in metadata.items()]
    else:
        simple = []
        for node in response.source_nodes:
            if node.node.ref_doc_id not in metadata:
                metadata[node.node.ref_doc_id] = None
                info = node.node.extra_info
                print("this is the info" + str(node))
                simple.append(info['url'] if info['url'] else f"{info['filename']} ({info['source']})")
        metadata = simple

    return metadata

def _update_chat_history(history: dict, query: str, response: str, lang: str) -> dict:
    ai_prefix = "System: "
    human_prefix = "User: "
    if lang == "fr":
        ai_prefix = "Système: "
        human_prefix = "Utilisateur: "

    while len(history['inputs']) >= _max_history:
        print("history is too big, truncating ..")
        history['inputs'].pop(0)
        history['outputs'].pop(0)
         
    history['inputs'].append(human_prefix + query)
    history['outputs'].append(ai_prefix + response)

    return history

def _history_as_str(history: dict) -> str: 
    inputs = history['inputs']
    outputs = history['outputs']

    history_str = ""

    for index, _ in enumerate(inputs):
        history_str += inputs[index] + "\n"
        history_str += outputs[index] + "\n"

    return history_str

'''
This one just returns a list of the outputs for the embeddings search. 
Since the question asked by the user might refer to previous anwser
'''
def _get_history_embeddings(history: dict) -> List[str]: 
    inputs = history['inputs']
    outputs = history['outputs']
    size = len(inputs) - 1
    if not inputs:
        return [""]  
    
    history_list = []
    # get last n items of that list
    for i in range(_max_embeddings_history):
        history_list.append(inputs[size-i] + "\n" + outputs[size-i] + "\n")
        #history_list.append(outputs[size-i])
        print(f"history added is  {history_list}")
    
    return history_list

"""
Check if the history map has 0 outputs and outputs
"""
def _has_history(history: dict) -> bool:
    if len(history['inputs']) > 0 and len(history['outputs']) > 0:
        return True
    return False

# bootstrap
_bootstrapIndex()