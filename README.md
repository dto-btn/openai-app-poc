# openai-app-poc
python flask app that uses an index and calls Azure OpenAI

## how-to

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --upgrade
```

Might need to run `Ctrl+Shift+P` in VSCode, type `Python: Create environment...` and follow instructions if needed.

To run the application simply do `flask --debug --app __init__.py run`

### build image and run it

```bash
docker build -t scdcciodtoopenaipoccontainerregistry.azurecr.io/app .
docker push scdcciodtoopenaipoccontainerregistry.azurecr.io/app
# then you can run it via 
docker run -it --env-file .env scdcciodtoopenaipoccontainerregistry.azurecr.io/app
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

## docker for wsl

I use Windows 10 with WSL 2+. I installed `docker` following [the instructions on their site](https://docs.docker.com/desktop/windows/wsl/).

You will need JIT admin rights and since the install will put the admin account inside `docker-users` you will have to add your own domain account to that group after that.

To do so open powershell (as admin):

```bash
whoami
net localgroup "docker-users" "<your username>" /add
```
You should see something like `Command completed successfully` then logout and then you can start `Docker Desktop`.