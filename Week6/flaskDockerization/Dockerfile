FROM python

WORKDIR /code

COPY . /code

RUN pip --no-cache-dir install -r requirements.txt
RUN pip install --upgrade pip

EXPOSE 5000

CMD [ "python", "app.py" ]
