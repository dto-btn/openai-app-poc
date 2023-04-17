FROM python:3.9

ENV TZ="Canada/Eastern"

WORKDIR /app

ADD . /app

RUN useradd appuser && chown -R appuser /app
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "__init__:app"]