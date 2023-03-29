FROM python

WORKDIR /code

COPY . /code

RUN pip --no-cache-dir install -r requirements.txt
RUN pip install --upgrade pip

EXPOSE 8000

CMD [ "uvicorn", "main:app", "--host=0.0.0.0", "--port=8000" ]
