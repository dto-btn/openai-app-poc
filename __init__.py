import logging
import os
from typing import List

import openai
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index import (GPTVectorStoreIndex, LangchainEmbedding,
                        PromptHelper, QuestionAnswerPrompt,
                        ServiceContext, download_loader, StorageContext, load_index_from_storage)
from llama_index.llm_predictor.chatgpt import ChatGPTLLMPredictor
from llama_index.storage.storage_context import DEFAULT_PERSIST_DIR
from llama_index.logger import LlamaLogger
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

from llama_index.prompts.prompts import RefinePrompt

load_dotenv()

app = Flask(__name__)

#storage_account_name = os.environ["STORAGE_ACCNT_NAME"]
key_vault_name          = os.environ["KEY_VAULT_NAME"]
openai_endpoint_name    = os.environ["OPENAI_ENDPOINT_NAME"]
deployment_name         = os.environ["OPENAI_DEPLOYMENT_NAME"]
_basepath              = "/tmp/"
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
    temperature = 0.7 # default
    body = request.json
    debug = False
    lang = "en"

    if "query" in body:
        query = request.json["query"]
    else:
         return jsonify({"msg":"You must ask a question!"})
    
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

    # Query ChatGPT / embeddings deployment(s)
    index = get_index(service_context)
    query_engine = index.as_query_engine(mode="embedding", 
                                        text_qa_template=_get_prompt_template(lang), 
                                        similarity_top_k=k, 
                                        response_mode="compact", 
                                        refine_template=_get_refined_prompt(lang), service_context=service_context)
    response = query_engine.query(query)
    #return StreamingResponse(index.query(query, streaming=True).response_gen)
    #print(response.get_formatted_sources())
    #print(service_context.llama_logger.get_logs())

    if debug:
        return jsonify({'query':query,'answer':str(response),'nodes_score':[node.score for node in response.source_nodes], 'logs': service_context.llama_logger.get_logs()})
    else:
        return jsonify({'query':query,'answer':str(response),'nodes_score':[node.score for node in response.source_nodes]})
       
def get_index(service_context: ServiceContext) -> "GPTVectorStoreIndex":
    # check if index file is present on fs ortherwise build it ...
    if os.path.exists(DEFAULT_PERSIST_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=DEFAULT_PERSIST_DIR)
        return load_index_from_storage(storage_context)
    else:
        index = _build_index(service_context)
        index.storage_context.persist()
        return index

"""

TODO: Move the two functions below to their own application service


"""       
def _build_index(service_context: ServiceContext) -> "GPTVectorStoreIndex":
    logging.info("Creating index...")
    container_client = blob_service_client.get_container_client(container="unstructureddocs")

    #TODO: terrible way to do things, index should be generated elsewhere and simply loaded here.
    for blob in container_client.list_blobs():
        _download_blob_to_file(blob_service_client, container_name="unstructureddocs", blob_name=blob.name)
    
    SimpleDirectoryReader  = download_loader("SimpleDirectoryReader")
    documents = SimpleDirectoryReader(input_dir='/tmp/sscplus2', recursive=True).load_data()
    #logging.info("The documents are:" + ''.join(str(x.doc_id) for x in documents))

    return GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    
def _download_blob_to_file(blob_service_client: BlobServiceClient, container_name, blob_name):

    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    # azure function app only allows write to /tmp on the file system
    isExist = os.path.exists(_basepath + os.path.dirname(blob_name))
    if not isExist:
        os.makedirs(_basepath + os.path.dirname(blob_name))

    with open(file=_basepath + blob_name, mode="wb") as sample_blob:
        download_stream = blob_client.download_blob()
        sample_blob.write(download_stream.readall())

def _get_service_context(temperature: str) -> "ServiceContext":
    # Define prompt helper
    max_input_size = 4096
    num_output = 256 #hard limit
    chunk_size_limit = 1000 # token window size per document
    max_chunk_overlap = 20 # overlap for each token fragment

    # using same dep as model name because of an older bug in langchains lib (now fixed I believe)
    llm = AzureChatOpenAI(deployment_name=deployment_name, 
                          temperature=temperature, 
                          openai_api_version=openai_api_version)
    print(llm)
    # https://gist.github.com/csiebler/32f371470c4e717db84a61874e951fa4
    llm_predictor = ChatGPTLLMPredictor(llm=llm)

    prompt_helper = PromptHelper(max_input_size=max_input_size, num_output=num_output, max_chunk_overlap=max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # limit is chunk size 1 atm
    embedding_llm = LangchainEmbedding(OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1, openai_api_version=openai_api_version))

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