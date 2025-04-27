FROM python:latest

COPY . /usr/src/app
RUN chmod +x /usr/src/app

WORKDIR /usr/src/app
RUN pip install -r requirements.txt

CMD ["python", "app.py"]
