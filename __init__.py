import logging
import os
from typing import List
from flask import Flask, request, jsonify

import openai
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from llama_index import (GPTSimpleVectorIndex, LangchainEmbedding,
                         LLMPredictor, PromptHelper, QuestionAnswerPrompt, ServiceContext, download_loader)

from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

#storage_account_name = os.environ["STORAGE_ACCNT_NAME"]
key_vault_name          = os.environ["KEY_VAULT_NAME"]
openai_endpoint_name    = os.environ["OPENAI_ENDPOINT_NAME"]
deployment_name         = os.environ["OPENAI_DEPLOYMENT_NAME"]
index_location          = os.getenv("INDEX_LOCATION", "/tmp/vector_index.json")

kv_uri              = f"https://{key_vault_name}.vault.azure.net"
azure_openai_uri    = f"https://{openai_endpoint_name}.openai.azure.com"

credential  = DefaultAzureCredential()
client      = SecretClient(vault_url=kv_uri, credential=credential)

blob_service_client = BlobServiceClient.from_connection_string(client.get_secret("openai-storage-connection").value)

openai.api_type     = 'azure'
openai.api_base     = azure_openai_uri
openai.api_key      = client.get_secret("AzureOpenAIKey").value
openai.api_version  = '2023-03-15-preview' # this may change in the future

os.environ["OPENAI_API_TYPE"]   = "azure"
os.environ["OPENAI_API_BASE"]   = azure_openai_uri
os.environ["OPENAI_API_KEY"]    = client.get_secret("AzureOpenAIKey").value
os.environ["OPENAI_API_VERSION"] = '2023-03-15-preview' # this may change in the future

@app.route("/health", methods=["GET"])
def health(): 
    return jsonify({"msg":"Healthy"})

@app.route("/prompt", methods=["POST"])
def prompt():
    prompt = ""
    temperature = 0.7
    body = request.json
    if "prompt" in body:
        prompt = request.json["prompt"]
    if "temp" in body:
        temperature = float(request.json["temp"])

    # Define prompt helper
    max_input_size = 1024
    num_output = 256
    chunk_size_limit = 1000 # token window size per document
    max_chunk_overlap = 20 # overlap for each token fragment

    QA_PROMPT_TMPL = (
        "We have provided context information below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given this information, please answer the question: {query_str}\n"
    )
    prompt_template = QuestionAnswerPrompt(QA_PROMPT_TMPL)

    prompt_helper = PromptHelper(max_input_size=max_input_size, num_output=num_output, max_chunk_overlap=max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    
    # using same dep as model name because of an older bug in langchains lib (now fixed I believe)
    llm = AzureChatOpenAI(deployment_name=deployment_name, temperature=temperature, openai_api_version="2023-03-15-preview")
    print(llm)
    llm_predictor = LLMPredictor(llm=llm)
    #current limitation with Azure OpenAI, has to be chunk size of 1
    embedding_llm = LangchainEmbedding(OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper, embed_model=embedding_llm)

    index = get_index(temperature, service_context)
    answer = index.query(prompt, mode="embedding")

    #print(answer.response)

    if answer:
        return jsonify({'prompt':prompt,'answer':str(answer),'nodes_score':[node.score for node in answer.source_nodes]})
    else:
        # ideally return json..
        return jsonify({"msg":
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response."}
        )

def get_index(temperature: float, service_context: ServiceContext) -> "GPTSimpleVectorIndex":
    # check if index file is present on fs ortherwise build it ...
    if os.path.exists(index_location):
        return GPTSimpleVectorIndex.load_from_disk(save_path=index_location, service_context=service_context)
    else:
        index = build_index(temperature, service_context)
        index.save_to_disk(index_location)
        return index

"""

TODO: Move the two functions bellow to their own application service


"""       
def build_index(temperature: float, service_context: ServiceContext) -> "GPTSimpleVectorIndex":
    logging.info("Creating index...")
    container_client = blob_service_client.get_container_client(container="unstructureddocs")

    #TODO: terrible way to do things, index should be generated elsewhere and simply loaded here.
    for blob in container_client.list_blobs():
        download_blob_to_file(blob_service_client, container_name="unstructureddocs", blob_name=blob.name)
    
    SimpleDirectoryReader  = download_loader("SimpleDirectoryReader")
    documents = SimpleDirectoryReader(input_dir='/tmp/sscplus').load_data()
    #logging.info("The documents are:" + ''.join(str(x.doc_id) for x in documents))

    return GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
    
def download_blob_to_file(blob_service_client: BlobServiceClient, container_name, blob_name):
    basepath = "/tmp/"
    
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    # azure function app only allows write to /tmp on the file system
    isExist = os.path.exists(basepath + os.path.dirname(blob_name))
    if not isExist:
        os.makedirs(basepath + os.path.dirname(blob_name))

    with open(file=basepath + blob_name, mode="wb") as sample_blob:
        download_stream = blob_client.download_blob()
        sample_blob.write(download_stream.readall())
