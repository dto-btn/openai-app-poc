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

from app.prompts.qna import (get_chat_prompt_template, get_prompt_template,
                             get_refined_prompt)

load_dotenv()

app = Flask(__name__)
CORS(app)

#storage_account_name = os.environ["STORAGE_ACCNT_NAME"]
key_vault_name          = os.environ["KEY_VAULT_NAME"]
openai_endpoint_name    = os.environ["OPENAI_ENDPOINT_NAME"]
deployment_name         = os.environ["OPENAI_DEPLOYMENT_NAME"]
_basepath               = "./container/"
openai_api_version      = "2023-03-15-preview" # this may change in the future

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

_default_index_name = "2023-07-19"

@app.route("/health", methods=["GET"])
def health(): 
    return jsonify({"msg":"Healthy"})

@app.route("/query", methods=["POST"])
def query():

    query = ""
    k = 2 # default
    temperature = 0.7 # default
    body = request.get_json(force=True)
    debug = False
    lang = "en"
    index_name = _default_index_name
    pretty = False # wether or not to pretty print medatada, used for the MS Teams chatbot ..
    history = {'inputs': [], 'outputs': []}
    history_embeddings = False

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

    if "debug" in body:
        debug = bool(body["debug"])

    if "pretty" in body:
        pretty = bool(body["pretty"])

    if "lang" in body:
        if str(body["lang"]) == "fr":
            lang = "fr"
    
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

    if "history_embeddings" in body:
        history_embeddings = bool(body["history_embeddings"])

    service_context = _get_service_context(temperature)
    set_global_service_context(service_context)
    
    index = _get_index(index_name=index_name)
    query_bundle = query

    #prompt building, and query bundle
    if _has_history(history):
        text_qa_template=get_chat_prompt_template(lang, _history_as_str(history))
        if history_embeddings:
            query_bundle = QueryBundle(
                query,
                custom_embedding_strs=_get_history_embeddings(history)
            )
    else:
        text_qa_template=get_prompt_template(lang)

    retriever = VectorIndexRetriever(index=index, similarity_top_k=k) # type: ignore

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.REFINE)

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

    for data in r.keys():
        print(f"{data}: {r[data]}")

    if debug:
        print(service_context.llama_logger.get_logs())
        r['logs'] = service_context.llama_logger.get_logs()

    return jsonify(r)

"""
 Builds Vector index based on root folder(s) contained in a blob storage container (container_name)
""" 
@app.route("/build", methods=["POST"])
def build_index():
    '''name of container from where we will download the files, else ignore'''
    container_name = None
    body = request.get_json(force=True)
    if "name" in body:
        container_name = body['name']

    '''aka save-as'''
    storage = DEFAULT_PERSIST_DIR
    if "storage" in body:
        storage = os.path.join(storage, body["storage"])

    '''folders to ignore when building the index'''
    ignore = []
    if "ignore" in body:
        ignore = body["ignore"]

    if container_name:
        storage_connection = client.get_secret("openai-storage-connection").value
        if storage_connection is not None:
            blob_service_client = BlobServiceClient.from_connection_string(storage_connection)
            container_client = blob_service_client.get_container_client(container=container_name)
            for blob in container_client.list_blobs():
                _download_blob_to_file(blob_service_client, container_name=container_name, blob_name=blob.name)
    '''loop over base container folder root documents to create an index for each'''
    documents = []
    for dir in os.listdir(_basepath):
        # list dirs that you want to skip index creation for (big ones that take 10-15 minutes ..)
        if dir not in ignore:
            #documents = SimpleDirectoryReader(input_dir=os.path.join(_basepath,container_name), recursive=True, file_metadata=_filename_fn).load_data()
            documents = documents + SimpleDirectoryReader(input_dir=os.path.join(_basepath, dir), recursive=True, file_metadata=_filename_fn).load_data()
    
    service_context = _get_service_context()
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    print(f"Creating index: {storage}")
    index.storage_context.persist(persist_dir=storage)

    return jsonify({'msg': "index loaded successfully"})

"""

Download files from an Azure Blob storage to the local FS

""" 
def _download_blob_to_file(blob_service_client: BlobServiceClient, container_name, blob_name):

    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    isExist = os.path.exists(_basepath + os.path.dirname(blob_name))
    if not isExist:
        os.makedirs(_basepath + os.path.dirname(blob_name))

    with open(file=_basepath + blob_name, mode="wb") as sample_blob:
        download_stream = blob_client.download_blob()
        sample_blob.write(download_stream.readall())

"""

Loads a Vector Index from the local filesystem

"""    
def _get_index(index_name: str, storage_location: str = DEFAULT_PERSIST_DIR):
    storage_context = StorageContext.from_defaults(persist_dir=os.path.join(storage_location, index_name))
    # index needs to be recreated its missing metadata for this code to work properly.
    #vector_store = SimpleVectorStore.from_persist_dir(persist_dir=os.path.join(storage_location, index_name))
    #return VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return load_index_from_storage(storage_context)

def _get_service_context(temperature: float = 0.7) -> "ServiceContext":
    # Define prompt helper
    context_window = 4096
    num_output = 256 #hard limit
    chunk_size_limit = 1000 # token window size per document
    chunk_overlap_ratio = 0.1 # overlap for each token fragment

    # using same dep as model name because of an older bug in langchains lib (now fixed I believe)
    llm = _get_llm(temperature)

    logging.info(llm)
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

    return ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper, embed_model=embedding_llm)

def _get_llm(temperature: float = 0.7):
    return AzureChatOpenAI(deployment_name=deployment_name, temperature=temperature)

def _get_llm_predictor(llm) -> LLMPredictor:
    return LLMPredictor(llm=llm,)


"""
Metadata building for the index nodes, stored in extra_info at response time.

filenames will look like this comming in : container/sscplus/sites/default/files/2022-12/cgci-web.docx

"""
def _filename_fn(filename: str) -> dict:

    print(f"Processing current filename: {filename}")
    # gather metadata about the file or url ... and add it as a dict
    lastmod = time.ctime(os.path.getmtime(filename))
    fn = os.path.basename(filename)
    url = ""

    """ 
    parse html file for the meta tag `canonical` and grab url. else leave it blank..

    Examples: 

        <meta name="url" content="https://www.tbs-sct.canada.ca/agreements-conventions/view-visualiser.aspx?id=1" />
        or
        <link rel="canonical" href="https://plus.ssc-spc.gc.ca/en/active-alerts" />
        or 
        <meta name="savepage-url" content="https://163gc.sharepoint.com/sites/VF-LFDF/SitePages/How-To-Guide.aspx">
        <meta name="savepage-url" content="https://163gc.sharepoint.com/sites/VF-LFDF">
    """
    if filename.endswith(".html"):
        with open(filename, "r") as fp:
            soup = BeautifulSoup(fp, "html.parser", from_encoding="UTF-8")
  
        if soup.find('link', {'rel': 'canonical'}):
            url = soup.find('link', {'rel': 'canonical'})["href"] # type: ignore
        elif soup.find('meta', {'name': 'url'}):
            url = soup.find('meta', {'name': 'url'})["content"] # type: ignore
        elif soup.find('meta', {'name': 'savepage-url'}):
            url = soup.find('meta', {'name': 'savepage-url'})["content"] # type: ignore

        if url != "":
            print(f"Found URL in html file: {url}")

    # first level will be _basepath, we ignore it and we take the first level folder which is the "source"
    source = filename.split(os.path.sep)[1]

    '''
    WARN: Temporary "trick" to process sscplus .txt files as they were html 
            since we do not have the proper html content as of now ..
    '''
    if filename.endswith(".txt") and source == "sscplus":
        f = ''.join(filename.split("container/sscplus/", 1)).replace(".txt", "")
        url = f"https://plus.ssc-spc.gc.ca/{f}"
        fn = f

    return {"filename": fn, "lastmodified": lastmod, "url": url, "source": source}

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
    ai_prefix = "AI: "
    human_prefix = "Human: "
    if lang == "fr":
        ai_prefix = "IA: "
        human_prefix = "Humain: "

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
Since the question asked by the user might refer to previous "context"
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
        print(f"range is {i}")
        history_list.append(inputs[size-i] + "\n" + outputs[size-i] + "\n")
        print(f"history added is  {history_list}")
    
    return history_list

"""
Check if the history map has 0 outputs and outputs
"""
def _has_history(history: dict) -> bool:
    if len(history['inputs']) > 0 and len(history['outputs']) > 0:
        return True
    return False