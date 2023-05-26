import logging
import os
import sys
import time
from typing import List
import ast

import openai
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

from llama_index import (GPTListIndex, GPTVectorStoreIndex, LangchainEmbedding,
                         PromptHelper,
                         ResponseSynthesizer, ServiceContext, StorageContext,
                         download_loader, load_index_from_storage, QueryBundle)

from llama_index.indices.composability import ComposableGraph
from llama_index.indices.postprocessor import SimilarityPostprocessor

from llama_index.llm_predictor import LLMPredictor
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.storage.storage_context import DEFAULT_PERSIST_DIR

from .prompts import (get_chat_prompt_template, get_refined_prompt) 

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

load_dotenv()

app = Flask(__name__)

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

blob_service_client = BlobServiceClient.from_connection_string(client.get_secret("openai-storage-connection").value)

openai.api_type    = os.environ["OPENAI_API_TYPE"]    = 'azure'
openai.api_base    = os.environ["OPENAI_API_BASE"]    = azure_openai_uri
openai.api_key     = os.environ["OPENAI_API_KEY"]     = client.get_secret("AzureOpenAIKey").value
openai.api_version = os.environ["OPENAI_API_VERSION"] = openai_api_version

_default_index_name = "all"

@app.route("/health", methods=["GET"])
def health(): 
    return jsonify({"msg":"Healthy"})

@app.route("/query", methods=["POST"])
def query():

    query = ""
    k = 2 # default
    temperature = 0.7 # default
    body = request.json
    debug = False
    lang = "en"
    index_name = _default_index_name
    pretty = False # wether or not to pretty print medatada, used for the MS Teams chatbot ..
    history = {'inputs': [], 'outputs': []}

    if "query" not in body:
        return jsonify({"error":"Request body must contain a query."}), 400
    else:
        query = request.json["query"]

    if "index" in body:
        index_name = request.json["index"]
           
    if "temp" in body:
        temperature = float(request.json["temp"])

    if "k" in body:
        k = int(request.json["k"])

    if "debug" in body:
        debug = bool(request.json["debug"])

    if "pretty" in body:
        pretty = bool(request.json["pretty"])

    if "lang" in body:
        if str(request.json["lang"]) == "fr":
            lang = "fr"
    
    if "chat_history" in body:
        # try and conver the chat history into a map
        try:
            history = ast.literal.eval(request.json["chat_history"])
            # validate we have an input and output key
            if 'inputs' not in history or 'outputs' not in history:
                raise Exception("Unable to properly parse chat_history")
        except:
            logging.error("Unable to convert chat_history to a proper map that contains input and outputs..")
            history = {'inputs': [], 'outputs': []}

    service_context = _get_service_context(temperature)
    
    index = _get_index(service_context=service_context, storage_name=index_name)
    
    retriever = index.as_retriever(
        retriever_mode="default", 
        similarity_top_k=k
    )

    # configure response synthesizer
    response_synthesizer = ResponseSynthesizer.from_args(
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.7)
        ],
    )

    # assemble query engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        service_context=service_context,
        text_qa_template=get_chat_prompt_template(lang, _history_as_str(history)),
        refine_template=get_refined_prompt(lang),
        response_mode="refine",
        verbose=True
    )

    #build QueryBundle
    query_bundle = QueryBundle(
        query,
        custom_embedding_strs=_history_as_list(history)
    )
    response = query_engine.query(query_bundle)
    metadata = _response_metadata(response, pretty)
    history = _update_chat_history(history, query, str(response), lang)

    r = {
            'query': query,
            'answer': str(response),
            'metadata': metadata,
            'chat_history': history
        }

    if debug:
        r['logs'] = service_context.llama_logger.get_logs()

    return jsonify(r)
    
@app.route("/build", methods=["POST"])
def build_index():
    if "name" not in request.json:
        return jsonify({"error":"Request body must contain a name for the index to create"}), 400
    
    download = True
    if "download" in request.json:
        download = bool(request.json["download"])
    
    container_name = request.json['name']
    storage = DEFAULT_PERSIST_DIR

    if "storage" in request.json:
        storage = request.json["storage"]

    if download:
        container_client = blob_service_client.get_container_client(container=container_name)
        for blob in container_client.list_blobs():
            _download_blob_to_file(blob_service_client, container_name=container_name, blob_name=blob.name)

    #filename_fn = lambda filename: {'filename': filename}
   
    SimpleDirectoryReader  = download_loader("SimpleDirectoryReader")
    documents = SimpleDirectoryReader(input_dir=_basepath, recursive=True, file_metadata=_filename_fn).load_data()

    service_context = _get_service_context()
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    logging.info(f"Creating index: {container_name}")
    index.storage_context.persist(persist_dir=os.path.join(storage,container_name))

    #return GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    return jsonify({'msg': "index loaded successfully"})

@app.route("/buildgraph", methods=["POST"])
def build_graph():
    if "indices" not in request.json:
        return jsonify({"error":"Request body must contain name(s) of indicies to create the graph from"}), 400
    if "graph_name" not in request.json:
        return jsonify({"error":"Request body must contain name of the graph to create"}), 400
    
    indices = request.json['indices']
    graph_name = request.json['graph_name']

    service_context = _get_service_context()
    storage_context = StorageContext.from_defaults()

    # define a list index over the vector indices
    # allows us to synthesize information across each index
    index_set = {}
    for name in indices:
        index_set[name] = _get_index(service_context=service_context, storage_name=name)

    index_summaries = []
    for name in indices:
        if name in _index_summaries:
            summary = ":".join([t for t in _index_summaries[name]['en']]) + "\n" + ":".join([t for t in _index_summaries[name]['fr']])
            print(summary)
            index_summaries.append(summary)
        else:
            index_summaries.append(f"Shared Services Canada (SSC) information Vector index that contains information about :{name}")
    
    graph = ComposableGraph.from_indices(
        GPTListIndex,
        [index_set[name] for name in indices], 
        index_summaries=index_summaries,
        service_context=service_context,
        storage_context=storage_context
    )
    root_id = graph.root_id

    storage_context.persist(persist_dir=os.path.join(DEFAULT_PERSIST_DIR, graph_name))

    return jsonify({'msg': f"graph created successfully with root_id: {root_id}"})

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
def _get_index(service_context: ServiceContext, storage_name: str, storage_location: str = DEFAULT_PERSIST_DIR) -> "GPTVectorStoreIndex":
    return load_index_from_storage(service_context=service_context, 
                                   storage_context=StorageContext.from_defaults(persist_dir=os.path.join(storage_location, storage_name)))

def _get_service_context(temperature: float = 0.7, history: ConversationBufferMemory = None) -> "ServiceContext":
    # Define prompt helper
    max_input_size = 4096
    num_output = 256 #hard limit
    chunk_size_limit = 1000 # token window size per document
    max_chunk_overlap = 20 # overlap for each token fragment

    # using same dep as model name because of an older bug in langchains lib (now fixed I believe)
    llm = _get_llm(temperature)

    logging.info(llm)
    llm_predictor = _get_llm_predictor(llm)

    prompt_helper = PromptHelper(max_input_size=max_input_size, num_output=num_output, max_chunk_overlap=max_chunk_overlap, chunk_size_limit=chunk_size_limit,)

    # limit is chunk size 1 atm
    embedding_llm = LangchainEmbedding(
        OpenAIEmbeddings(
            model="text-embedding-ada-002",
            ), 
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

    # check if it's an html file, if so treat it as a url instead.
    if filename.endswith(".html"):
        # parse html file for the meta tag `canonical` and grab url. else leave it blank..
        with open(filename, "r") as fp:
            soup = BeautifulSoup(fp, "html.parser", from_encoding="UTF-8")
            url = soup.find('link', {'rel': 'canonical'})['href']
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
            else:
                scores[node.node.ref_doc_id].append(node.score)

        [v.update({"node_scores": scores[k]}) for k, v in metadata.items()]
    else:
        simple = []
        for node in response.source_nodes:
            if node.node.ref_doc_id not in metadata:
                metadata[node.node.ref_doc_id] = None
                info = node.node.extra_info
                simple.append(info['url'] if info['url'] != "" else f"{info['filename']} ({info['source']})")
        metadata = simple

    return metadata

def _update_chat_history(history: dict, query: str, response: str, lang: str) -> str:
    ai_prefix = "AI: "
    human_prefix = "Human: "
    if lang == "fr":
        ai_prefix = "IA: "
        human_prefix = "Humain: "
    
    history['inputs'].append(human_prefix + query)
    history['outputs'].append(ai_prefix + response)

def _history_as_str(history: dict) -> str: 
    inputs = history['inputs']
    outputs = history['outputs']

    history_str = ""

    for index, _ in enumerate(inputs):
        history_str += inputs[index] + "\n"
        history_str += outputs[index] + "\n"

    return history_str

def _history_as_list(history: dict) -> List[str]:    
    inputs = history['inputs']
    outputs = history['outputs']

    history_list = []

    for index, _ in enumerate(inputs):
        print([index] + "\n" + outputs[index] + "\n")
        history_list.append([index] + "\n" + outputs[index] + "\n")

    if not history_list:
        return None
    return history_list