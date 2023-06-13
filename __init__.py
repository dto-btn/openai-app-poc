import logging
import os
import sys
import time
from typing import List
import ast
import base64
import json

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

from .prompts import (get_chat_prompt_template, get_refined_prompt, get_prompt_template) 

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

openai.api_type    = os.environ["OPENAI_API_TYPE"]    = 'azure'
openai.api_base    = os.environ["OPENAI_API_BASE"]    = azure_openai_uri
openai.api_key     = os.environ["OPENAI_API_KEY"]     = client.get_secret("AzureOpenAIKey").value
openai.api_version = os.environ["OPENAI_API_VERSION"] = openai_api_version

'''
Keeping the chat history for the context (what is sent to ChatGPT 3.5) to 5, seems big enough. 

And we will keep the embedding chat history to a minimum (and we also restrict it to outputs only ..)
'''
_max_history = 5
_max_embeddings_history = 2

_default_index_name = "one"

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
            encoded = request.json["chat_history"]
            decoded = base64.b64decode(encoded)
            history = json.loads(decoded)
            # validate we have an input and output key
            if 'inputs' not in history or 'outputs' not in history:
                raise Exception("Unable to properly parse chat_history")
        except:
            logging.error("Unable to convert chat_history to a proper map that contains input and outputs..")
            history = {'inputs': [], 'outputs': []}

    service_context = _get_service_context(temperature)
    
    index = _get_index(service_context=service_context, storage_name=index_name)
    
    retriever = index.as_retriever(
        retriever_mode="embedding", 
        similarity_top_k=k
    )

    # configure response synthesizer
    response_synthesizer = ResponseSynthesizer.from_args(
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.7)
        ],
    )

    #prompt building, and query bundle
    if _has_history(history):
        text_qa_template=get_chat_prompt_template(lang, _history_as_str(history))
        query_bundle = QueryBundle(
            query,
            custom_embedding_strs=_get_history_embeddings(history)
        )
    else:
        text_qa_template=get_prompt_template(lang)
        query_bundle = query


    # assemble query engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        service_context=service_context,
        text_qa_template=text_qa_template,
        refine_template=get_refined_prompt(lang),
        response_mode="tree_summarize",
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

    if debug:
        r['logs'] = service_context.llama_logger.get_logs()

    return jsonify(r)

"""
 Builds Vector index based on root folder(s) contained in a blob storage container (container_name)
""" 
@app.route("/build", methods=["POST"])
def build_index():
    if "name" not in request.json:
        container_name = None
    else:
        container_name = request.json['name']

    storage = DEFAULT_PERSIST_DIR
    if "storage" in request.json:
        storage = request.json["storage"]

    if container_name:
        blob_service_client = BlobServiceClient.from_connection_string(client.get_secret("openai-storage-connection").value)
        container_client = blob_service_client.get_container_client(container=container_name)
        for blob in container_client.list_blobs():
            _download_blob_to_file(blob_service_client, container_name=container_name, blob_name=blob.name)

    '''loop over base container folder root documents to create an index for each'''
    for dir in os.listdir(_basepath):
        # list dirs that you want to skip index creation for (big ones that take 10-15 minutes ..)
        if dir not in []:
            SimpleDirectoryReader  = download_loader("SimpleDirectoryReader")
            #documents = SimpleDirectoryReader(input_dir=os.path.join(_basepath,container_name), recursive=True, file_metadata=_filename_fn).load_data()
            documents = SimpleDirectoryReader(input_dir=os.path.join(_basepath, dir), recursive=True, file_metadata=_filename_fn).load_data()

            service_context = _get_service_context()
            index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
            logging.info(f"Creating index: {dir}")
            index.storage_context.persist(persist_dir=os.path.join(storage, dir))

    return jsonify({'msg': "index loaded successfully"})

@app.route("/buildgraph", methods=["POST"])
def build_graph():
    index_summaries = {
        _default_index_name: {
            'en': ["Shared Services Canada (SSC) information about the department", "Contains various information about the intranet website SSCPlus (MySSC) and the EVEC and ITSM group"],
            'fr': ["Informations à propos du département des Services partagés Canada (SPC)", "Contient des information variées à propos du site intranet MonSpC ainsi que des groups EVEC et ITSM"]}
    }
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
        if name in index_summaries:
            summary = ":".join([t for t in index_summaries[name]['en']]) + "\n" + ":".join([t for t in index_summaries[name]['fr']])
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
            url = soup.find('link', {'rel': 'canonical'})["href"]
        elif soup.find('meta', {'name': 'url'}):
            url = soup.find('meta', {'name': 'url'})["content"]
        elif soup.find('meta', {'name': 'savepage-url'}):
            url = soup.find('meta', {'name': 'savepage-url'})["content"]

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

def _update_chat_history(history: dict, query: str, response: str, lang: str) -> dict:
    ai_prefix = "AI: "
    human_prefix = "Human: "
    if lang == "fr":
        ai_prefix = "IA: "
        human_prefix = "Humain: "

    while len(history['inputs']) >= _max_history:
        logging.debug("history is too big, truncating ..")
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
    if not inputs:
        return None  
    
    history_list = []
    # get last n items of that list
    for i in range(_max_embeddings_history):
        history_list.append(inputs[-i] + "\n")
    
    return history_list

"""
Check if the history map has 0 outputs and outputs
"""
def _has_history(history: dict) -> bool:
    if len(history['inputs']) > 0 and len(history['outputs']) > 0:
        return True
    return False