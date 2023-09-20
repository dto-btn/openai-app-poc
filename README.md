# openai-app-poc
python flask app that uses llama index and langchain to build a vector index and query an LLM (Azure OpenAI).

## how-to

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --upgrade
```

Might need to run `Ctrl+Shift+P` in VSCode, type `Python: Create environment...` and follow instructions if needed.

To run the application simply do `flask --debug --app app run`

### sending a query

Only the `body.query` parameter is mandatory, other fields are optionals and have reasonable defaults.

```bash
curl --location 'http://127.0.0.1:5000/query' \
--header 'Content-Type: application/json' \
--data '{
    "query": "What is the ITSM training mailbox email address? I want the email address with the ampersand in it.",
    "temp": 0.7,
    "k": 3
}'
```

### build image and run it

```bash
docker build -t scdcciodtoopenaipoccontainerregistry.azurecr.io/app:3.0.2 .
docker push scdcciodtoopenaipoccontainerregistry.azurecr.io/app:3.0.2
# then you can run it via 
docker run -it --env-file .env scdcciodtoopenaipoccontainerregistry.azurecr.io/openai-app-poc:3.0.2
```

### troubleshooting

If you ever get an error like this one while building/loading the vector index: 

```log
2023-04-18T13:22:20.948654585Z     type = docstore_dict[TYPE_KEY]
2023-04-18T13:22:20.948658985Z KeyError: '__type__'
```

It is mostlikely because the index wasn't build with the same package version it is being loaded. Simply rebuild the index with the most up to date version of the packages and re-load it.

## documentation

* https://code.visualstudio.com/docs/python/tutorial-flask
* how `llama_index` [vector indices work](https://gpt-index.readthedocs.io/en/latest/guides/primer/index_guide.html#vector-store-index)

### docker for wsl

I use Windows 10 with WSL 2+. I installed `docker` following [the instructions on their site](https://docs.docker.com/desktop/windows/wsl/).

You will need JIT admin rights and since the install will put the admin account inside `docker-users` you will have to add your own domain account to that group after that.

To do so open powershell (as admin):

```bash
whoami
net localgroup "docker-users" "<your username>" /add
```
You should see something like `Command completed successfully` then logout and then you can start `Docker Desktop`.

### Using Azure Cognitive Search

https://github.com/Azure-Samples/azure-search-openai-demo/blob/main/app/backend/app.py

### building a chatbot

https://gpt-index.readthedocs.io/en/latest/guides/tutorials/building_a_chatbot.html

### Installing Python 3.11 on Ubuntu 22.04

https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv
python3 -V
```

## documentation

* [Improve search using Azure Cognitive search index](https://gpt-index.readthedocs.io/en/latest/examples/vector_stores/CognitiveSearchIndexDemo.html#basic-example)

