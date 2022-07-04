FROM tensorflow/tensorflow:latest
EXPOSE 8080

# Upgrade pip 
RUN pip install -U pip
WORKDIR /root/app
COPY . .
RUN pip install -r ./requirements.txt

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]

