FROM python:3.11.5

ENV TZ="Canada/Eastern"

WORKDIR /app

COPY requirements-freeze.txt requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
# https://github.com/mindsdb/mindsdb/pull/7155
RUN python3 -c 'import nltk; nltk.download("punkt"); nltk.download("stopwords");'

COPY app/ ./app/

HEALTHCHECK CMD curl --fail http://localhost:5000 || exit 1
EXPOSE 5000

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]