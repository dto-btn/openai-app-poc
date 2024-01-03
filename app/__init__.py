import base64
import json  # bourne
import logging
import logging.handlers
import os
import sys
import time
from typing import Dict, List, Optional, Union

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from llama_index import (GPTListIndex, PromptHelper,
                         QueryBundle, ServiceContext, StorageContext,
                         load_index_from_storage, set_global_service_context)
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.llm_predictor import LLMPredictor
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.response_synthesizers import get_response_synthesizer
from llama_index.response_synthesizers.type import ResponseMode
from llama_index.retrievers import VectorIndexRetriever
from llama_index.storage.storage_context import DEFAULT_PERSIST_DIR
from llama_index.prompts.default_prompt_selectors import (
    DEFAULT_REFINE_PROMPT_SEL,
    DEFAULT_TEXT_QA_PROMPT_SEL,
    DEFAULT_TREE_SUMMARIZE_PROMPT_SEL,
)

from app.prompts.qna import (get_chat_prompt_template, get_prompt_template,
                             get_refined_prompt)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

load_dotenv()

app = Flask(__name__)
CORS(app)

key_vault_name          = os.environ["KEY_VAULT_NAME"]
openai_endpoint_name    = os.environ["OPENAI_ENDPOINT_NAME"]

models = {
    "gpt-4": {"name": "gpt-4", "deployment": os.getenv("GPT4_DEPLOYMENT", "gpt-4-1106"), "context_window": 8192, "index": {} },
}

kv_uri              = f"https://{key_vault_name}.vault.azure.net"
azure_openai_uri    = f"https://{openai_endpoint_name}.openai.azure.com"

credential  = DefaultAzureCredential()
client      = SecretClient(vault_url=kv_uri, credential=credential)
api_key     = client.get_secret(os.getenv("OPENAI_KEY_NAME", "AzureOpenAIKey")).value
api_version = "2023-07-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=azure_openai_uri,
    api_key=api_key
)


'''
Keeping the chat history for the context (what is sent to ChatGPT 3.5) to 5, seems big enough. 

And we will keep the embedding chat history to a minimum (and we also restrict it to outputs only ..)
'''
_max_history = 5
_max_embeddings_history = 1

_default_index_name = os.getenv("INDEX_NAME", "2023-10-18")

'''
Bootstrap function to pre-load the vector index(ices)
'''
def _bootstrapIndex():
    for model in models.values():
        start = time.time()
        print("Loading up the index: {}... (with model: {})".format(_default_index_name, model))
        set_global_service_context(_get_service_context(model["name"], model["deployment"] , model["context_window"]))
        # indices are pre-loaded with the same model they will be queried with
        model["index"] = _get_index(index_name=_default_index_name)
        end = time.time()
        print("Took {} seconds to load.".format(end-start))
    set_global_service_context(None) # want to enforce people to set it properly

@app.route("/health", methods=["GET"])
def health(): 
    return jsonify({"msg":"Healthy"})

@app.route("/chat", methods=["POST"])
def chat():
    """
    Send a question to GPT directly along with your context prompt
    """
    data = request.get_json()

    query = data.get('query')
    prompt = data.get('prompt')
    temp = data.get('temp', 0.7)
    tokens = data.get('token', 800)
    history = data.get('history', [])
    past_msg_incl = data.get('past_msg_incl', 10)

    r = _generate_response(query, prompt=prompt, temp=temp, tokens=tokens, history=history, past_msg_incl=past_msg_incl)

    return jsonify(r)

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

    service_context = _get_service_context(model=model_data["name"], deployment=model_data["deployment"], context_window=model_data["context_window"], temperature=temperature, num_output=num_output)
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
    response_synthesizer = get_response_synthesizer(response_mode=response_mode, 
                                                    service_context=service_context,
                                                    #TODO: fix the prompt templates localisation
                                                    # https://docs.llamaindex.ai/en/stable/examples/prompts/advanced_prompts.html
                                                    #text_qa_template=text_qa_template,
                                                    #summary_template=DEFAULT_TREE_SUMMARIZE_PROMPT_SEL,
                                                    #simple_template=text_qa_template,
                                                    #refine_template=text_qa_template,
                                                    )

    # assemble query engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        service_context=service_context,
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

    logging.info(f"[QUERY]: {r['query']} [ANSWER]:{r['answer']}")
    return jsonify(r)

def _get_index(index_name: str, storage_location: str = DEFAULT_PERSIST_DIR):
    """
    Loads a Vector Index from the local filesystem
    """
    storage_context = StorageContext.from_defaults(persist_dir=os.path.join(storage_location, index_name))
    # index needs to be recreated its missing metadata for this code to work properly.
    #vector_store = SimpleVectorStore.from_persist_dir(persist_dir=os.path.join(storage_location, index_name))
    #return VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return load_index_from_storage(storage_context)

def _get_service_context(model: str, deployment: str, context_window: int, num_output: int = 800, temperature: float = 0.7,) -> "ServiceContext":
    # using same dep as model name because of an older bug in langchains lib (now fixed I believe)
    llm = _get_llm(model, deployment, temperature)

    llm_predictor = _get_llm_predictor(llm)

    chunk_overlap_ratio = 0.1 # overlap for each token fragment
    prompt_helper = PromptHelper(context_window=context_window, num_output=num_output, chunk_overlap_ratio=chunk_overlap_ratio,)

    # limit is chunk size 1 atm
    embedding_llm = AzureOpenAIEmbeddings(
            model="text-embedding-ada-002", api_key=api_key, azure_endpoint=azure_openai_uri)

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])

    return ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embedding_llm, callback_manager=callback_manager, prompt_helper=prompt_helper)

def _get_llm(model: str, deployment: str, temperature: float = 0.7):
    return AzureChatOpenAI(model=model, azure_deployment=deployment,
                           temperature=temperature,api_key=api_key, api_version=api_version, azure_endpoint=azure_openai_uri)

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


def _get_history_embeddings(history: dict) -> List[str]:
    """
    This one just returns a list of the outputs for the embeddings search. 
    Since the question asked by the user might refer to previous anwser
    """
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

def _has_history(history: dict) -> bool:
    """
    Check if the history map has 0 outputs and outputs
    """
    if len(history['inputs']) > 0 and len(history['outputs']) > 0:
        return True
    return False

def _generate_response(query: str, prompt: str, temp: float, tokens: int, history: List[Dict[str, str]], past_msg_incl: int) -> Union[Dict, tuple]:

    '''minimal validation'''
    if not isinstance(history, list):
        return jsonify({"error":"history must be a list of dictionaries"}), 400
    if not query:
        return jsonify({"error":"Request body must contain a query"}), 400
    if not prompt and not history:
        return jsonify({"error":"Request body must contain at least a prompt or an history"}), 400
    if past_msg_incl > 20:
        past_msg_incl = 20

    # if we have an history (and no prompt) ... use it, else reset history so to speak to force new prompt
    # kiss principle here and the goal is to mimick what they do on the Azure OpenAI playground right now for simplicy
    if history and not prompt:
        history.append({"role":"user", "content":query})
        # truncate history if needed (and keep the first item in the list since it contains the prompt set)
        if len(history) > past_msg_incl:
            history = [history[0]] + (history[-(past_msg_incl-1):] if past_msg_incl > 1 else []) #else if 1 we end up with -0 wich is interpreted as 0: (whole list)
    else:
       logging.info(f"PROMPT: {prompt}")
       history = [{"role":"system","content":prompt}, {"role":"user","content":query}]

    response = client.chat.completions.create(
        model=os.getenv("GPT4_DEPLOYMENT", "gpt-4-1106"),
        messages = history, # type: ignore
        temperature=temp,
        max_tokens=tokens,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)

    content = response.choices[0].message.content
    role = response.choices[0].message.role
    logging.info(f"[QUERY] {query} [ANSWER] {content} (role: {role})")

    r = {'message': [], 'created': '', 'history': [], 'id': '', 'model': '', 'object': '', 'usage': {'completion_tokens': int, 'prompt_tokens': int, 'total_tokens': int}}

    r['id'] = response.id
    r['message'] = {"role": role,"content": content}
    history.append(r['message'])
    r['history'] = history
    r['created'] = response.created
    if response.usage is not None:
        r['usage'] = {'completion_tokens': response.usage.completion_tokens, 'prompt_tokens': response.usage.prompt_tokens, 'total_tokens': response.usage.total_tokens}
    return r

# bootstrap
_bootstrapIndex()