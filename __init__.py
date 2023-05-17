import logging
import os
import sys
import time
from typing import List

import openai
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import (AIMessagePromptTemplate,
                                    ChatPromptTemplate,
                                    HumanMessagePromptTemplate)
from llama_index import (GPTListIndex, GPTVectorStoreIndex, LangchainEmbedding,
                         PromptHelper, QuestionAnswerPrompt, ServiceContext,
                         StorageContext, download_loader,
                         load_index_from_storage)
from llama_index.indices.composability import ComposableGraph
from llama_index.llm_predictor import LLMPredictor
from llama_index.logger import LlamaLogger
from llama_index.prompts.prompts import RefinePrompt
from llama_index.storage.storage_context import DEFAULT_PERSIST_DIR

from llama_index.response.schema import (
    RESPONSE_TYPE,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

load_dotenv()

app = Flask(__name__)

#storage_account_name = os.environ["STORAGE_ACCNT_NAME"]
key_vault_name          = os.environ["KEY_VAULT_NAME"]
openai_endpoint_name    = os.environ["OPENAI_ENDPOINT_NAME"]
deployment_name         = os.environ["OPENAI_DEPLOYMENT_NAME"]
_basepath              = "./container/"
openai_api_version = "2023-03-15-preview" # this may change in the future

kv_uri              = f"https://{key_vault_name}.vault.azure.net"
azure_openai_uri    = f"https://{openai_endpoint_name}.openai.azure.com"

credential  = DefaultAzureCredential()
client      = SecretClient(vault_url=kv_uri, credential=credential)

blob_service_client = BlobServiceClient.from_connection_string(client.get_secret("openai-storage-connection").value)

openai.api_type    = os.environ["OPENAI_API_TYPE"]    = 'azure'
openai.api_base    = os.environ["OPENAI_API_BASE"]    = azure_openai_uri
openai.api_key     = os.environ["OPENAI_API_KEY"]     = client.get_secret("AzureOpenAIKey").value
openai.api_version = os.environ["OPENAI_API_VERSION"] = openai_api_version

@app.route("/health", methods=["GET"])
def health(): 
    return jsonify({"msg":"Healthy"})

@app.route("/query", methods=["POST"])
def query():

    query = ""
    k = 3 # default
    #temperature = 0.7 # default
    body = request.json
    debug = False
    lang = "en"
    index = []

    if "query" not in body or "index" not in body:
        return jsonify({"error":"Request body must contain a query and index"}), 400
    else:
        index = request.json["index"]
        query = request.json["query"]
           
    if "temp" in body:
        temperature = float(request.json["temp"])

    if "k" in body:
        k = int(request.json["k"])

    if "debug" in body:
        debug = bool(request.json["debug"])

    if "lang" in body:
        if str(request.json["lang"]) == "fr":
            lang = "fr"

    service_context = _get_service_context(temperature)

    indices = {}
    for i in index:
        si = _get_index(storage_name=i)
        if not isinstance(si, GPTVectorStoreIndex): # error loading the index ...
            return jsonify({'error':f'unable to load index: {i}'}), 500
        if i == "itsm":
            indices[i] = (si, "Contains various information about ITSM, EVEC and Onyx systems.")
        elif i == "sscplus":
            indices[i] = (si, "Contains generic information about processes, branches and offices inside Shared Services Canada (SSC).")
        else:
            indices[i] = (si, i)

    graph = ComposableGraph.from_indices(GPTListIndex, [v[0] for v in indices.values()] , index_summaries=[v[1] for v in indices.values()])
    
    custom_query_engines = {
        i[0].index_id: i[0].as_query_engine(
            mode="embedding", 
            text_qa_template=_get_prompt_template(lang), 
            similarity_top_k=k, 
            # https://github.com/jerryjliu/llama_index/blob/main/docs/guides/primer/usage_pattern.md#configuring-response-synthesis
            response_mode="tree_summarize", # other modes are default and compact 
            refine_template=_get_refined_prompt(lang), 
            service_context=service_context
        )
        for i in indices.values()
    }

    if(len(indices) == 1):
        query_engine = list(custom_query_engines.values())[0]
    else:
        custom_query_engines[graph.root_id] = graph.root_index.as_query_engine(
            response_mode="tree_summarize",
            service_context=service_context,
        )

        query_engine = graph.as_query_engine(custom_query_engines=custom_query_engines)

    response = query_engine.query(query)
    #return StreamingResponse(index.query(query, streaming=True).response_gen)
    #print(response.get_formatted_sources())
    #print(service_context.llama_logger.get_logs())

    metadata = _response_metadata(response)

    r = {
            'query':query,
            'answer':str(response),
            'metadata': metadata
        }

    if debug:
        r['logs'] = service_context.llama_logger.get_logs()

    return jsonify(r)
"""
@app.route("/build", methods=["POST"])
def build_index():
    if "name" not in request.json:
        return jsonify({"error":"Request body must contain a name for the index to create"}), 400
    
    container_name = request.json['name']
    storage = DEFAULT_PERSIST_DIR

    if "storage" in request.json:
        storage = request.json["storage"]

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
"""
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
def _get_index(storage_name: str, storage_location: str = DEFAULT_PERSIST_DIR) -> "GPTVectorStoreIndex":
    # check if index file is present on fs ortherwise build it ...
    loc = os.path.join(storage_location, storage_name)
    if os.path.exists(loc):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=loc)
            return load_index_from_storage(storage_context)
        except:
            return None
    else:
        return None

def _get_service_context(temperature: float = 0.7) -> "ServiceContext":
    # Define prompt helper
    max_input_size = 4096
    num_output = 256 #hard limit
    chunk_size_limit = 1000 # token window size per document
    max_chunk_overlap = 20 # overlap for each token fragment

    # using same dep as model name because of an older bug in langchains lib (now fixed I believe)
    llm = AzureChatOpenAI(deployment_name=deployment_name, 
                            temperature=temperature,)
    print(llm)
    # https://gist.github.com/csiebler/32f371470c4e717db84a61874e951fa4
    llm_predictor = LLMPredictor(llm=llm,)

    prompt_helper = PromptHelper(max_input_size=max_input_size, num_output=num_output, max_chunk_overlap=max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # limit is chunk size 1 atm
    embedding_llm = LangchainEmbedding(
        OpenAIEmbeddings(
            deployment="text-embedding-ada-002", 
            chunk_size=1, 
            openai_api_key= openai.api_key,
            openai_api_base=openai.api_base,
            openai_api_type=openai.api_type,
            openai_api_version=openai.api_version,
            ), 
        embed_batch_size=1)

    llama_logger = LlamaLogger()

    return ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper, embed_model=embedding_llm, llama_logger=llama_logger)

"""

NOTE: for refined prompt templates for bilingual 
we also have to modify the refined prompt templates, 
which generally will change the original french answer to an english one

SEE: 
    * https://github.com/jerryjliu/llama_index/blob/main/llama_index/prompts/chat_prompts.py and 
    * https://github.com/jerryjliu/llama_index/issues/1335

"""
def _get_prompt_template(lang: str):

    if lang == "fr":
        QA_PROMPT_TMPL = (
            "Vous êtes un assistant de Services partagés Canada (SPC). Nous avons fourni des informations contextuelles ci-dessous. \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "Compte tenu de ces informations, veuillez répondre à la question suivante dans la langue française: {query_str}\n"
        )
    else:
        QA_PROMPT_TMPL = (
            "You are a Shared Services Canada (SSC) assistant. We have provided context information below. \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "Given this information, please answer the question: {query_str}\n"
    )

    return QuestionAnswerPrompt(QA_PROMPT_TMPL)

def _get_refined_prompt(lang: str):
    # Refine Prompt
    if lang == "fr":
        CHAT_REFINE_PROMPT_TMPL_MSGS = [
            HumanMessagePromptTemplate.from_template("{query_str}"),
            AIMessagePromptTemplate.from_template("{existing_answer}"),
            HumanMessagePromptTemplate.from_template(
                "J'ai plus de contexte ci-dessous qui peut être utilisé"
                "(uniquement si nécessaire) pour mettre à jour votre réponse précédente.\n"
                "------------\n"
                "{context_msg}\n"
                "------------\n"
                "Compte tenu du nouveau contexte, mettre à jour la réponse précédente pour mieux"
                "répondez à ma question précédente."
                "Si la réponse précédente reste la même, répétez-la textuellement."
                "Ne référencez jamais directement le nouveau contexte ou ma requête précédente.",
            ),
        ]
    else:
        CHAT_REFINE_PROMPT_TMPL_MSGS = [
            HumanMessagePromptTemplate.from_template("{query_str}"),
            AIMessagePromptTemplate.from_template("{existing_answer}"),
            HumanMessagePromptTemplate.from_template(
                "I have more context below which can be used "
                "(only if needed) to update your previous answer.\n"
                "------------\n"
                "{context_msg}\n"
                "------------\n"
                "Given the new context, update the previous answer to better "
                "answer my previous query."
                "If the previous answer remains the same, repeat it verbatim. "
                "Never reference the new context or my previous query directly.",
            ),
        ]

    CHAT_REFINE_PROMPT_LC = ChatPromptTemplate.from_messages(CHAT_REFINE_PROMPT_TMPL_MSGS)
    return RefinePrompt.from_langchain_prompt(CHAT_REFINE_PROMPT_LC)

@app.route("/build", methods=["POST"])
def build_index():
    logging.info("Creating index...")
    container_name = "itsm"
    container_client = blob_service_client.get_container_client(container=container_name)

    #TODO: terrible way to do things, index should be generated elsewhere and simply loaded here.
    for blob in container_client.list_blobs():
        _download_blob_to_file(blob_service_client, container_name=container_name, blob_name=blob.name)
    
    SimpleDirectoryReader  = download_loader("SimpleDirectoryReader")
    documents = SimpleDirectoryReader(input_dir=_basepath, recursive=True).load_data()
    #logging.info("The documents are:" + ''.join(str(x.doc_id) for x in documents))

    fn = filename
    if fn.startswith("container/"):
        fn = fn.split("container/")[1]

    return {"filename": fn, "lastmodified": lastmod}

def _response_metadata(response: RESPONSE_TYPE) -> dict:
    metadata = {}
    scores = {}
    [print(str(node.score) + " and docid " + str(node.node.ref_doc_id)) for node in response.source_nodes]

    for node in response.source_nodes:
        print("docid " + str(node.node.ref_doc_id) + " and here is the current node score: " + str(node.score))
        if node.node.ref_doc_id not in metadata:
            scores[node.node.ref_doc_id] = [node.score]
            info = node.node.extra_info
            if 'filename' in info:
                metadata[node.node.ref_doc_id] = {"filename": info['filename']}
        else:
            scores[node.node.ref_doc_id].append(node.score)

    [v.update({"node_scores": scores[k]}) for k, v in metadata.items()]

    print("docid " + str(node.node.ref_doc_id) + " and here is the current node scores: " + str(metadata.get(node.node.ref_doc_id).get("node_scores")))

    return metadata
