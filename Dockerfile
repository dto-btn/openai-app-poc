FROM python:3.9-slim

ENV TZ="Canada/Eastern"

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY __init__.py ./app.py
COPY prompts.py ./prompts.py

RUN addgroup --gid 1001 --system app && \
    adduser --no-create-home --shell /bin/false --disabled-password --uid 1001 --system --group app && \
    chown -R app /app

#TODO: fix config since app user via llama_index needs to write to /usr/local
#USER app

HEALTHCHECK CMD curl --fail http://localhost:5000 || exit 1
EXPOSE 5000

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]