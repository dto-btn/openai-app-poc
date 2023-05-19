import logging
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
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import (AIMessagePromptTemplate,
                                    ChatPromptTemplate,
                                    HumanMessagePromptTemplate)
from llama_index import (GPTVectorStoreIndex, LangchainEmbedding,
                         PromptHelper, QuestionAnswerPrompt,
                         ResponseSynthesizer, ServiceContext, StorageContext,
                         download_loader, load_index_from_storage)
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.llm_predictor import LLMPredictor
from llama_index.logger import LlamaLogger
from llama_index.prompts.prompts import RefinePrompt
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.storage.storage_context import DEFAULT_PERSIST_DIR

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
    index = ["unstructureddocs-all"] # default ..
    pretty = False # wether or not to pretty print medatada, used for the MS Teams chatbot ..

    if "query" not in body:
        return jsonify({"error":"Request body must contain a query."}), 400
    else:
        query = request.json["query"]

    if "index" in body:
        index = request.json["index"]
           
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

    service_context = _get_service_context(temperature)
    index = _get_index(service_context=service_context, storage_name=index[0])

    retriever = index.as_retriever(retriever_mode="embedding", similarity_top_k=k)
    # configure retriever
    # retriever = VectorIndexRetriever(
    #     index=index, 
    #     similarity_top_k=2,
    # )

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
        text_qa_template=_get_prompt_template(lang),
        refine_template=_get_refined_prompt(lang), 
    )

    response = query_engine.query(query)
    metadata = _response_metadata(response, pretty)

    r = {
            'query':query,
            'answer':str(response),
            'metadata': metadata
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
    # check if index file is present on fs ortherwise build it ...
    loc = os.path.join(storage_location, storage_name)
    if os.path.exists(loc):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=loc)
            return load_index_from_storage(service_context=service_context, storage_context=storage_context)
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
                            temperature=temperature)
    print(llm)
    # https://gist.github.com/csiebler/32f371470c4e717db84a61874e951fa4
    llm_predictor = LLMPredictor(llm=llm,)

    prompt_helper = PromptHelper(max_input_size=max_input_size, num_output=num_output, max_chunk_overlap=max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # limit is chunk size 1 atm
    embedding_llm = LangchainEmbedding(
        OpenAIEmbeddings(
            model="text-embedding-ada-002",
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
        f = ''.join(filename.split("sscplus", 1)).replace(".txt", ".html")
        url = f"https://plus.ssc-spc.gc.ca/{f}"

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