# Installs necessary dependencies.
FROM tensorflow/tensorflow:latest-gpu

# Installs necessary dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
         wget \
         curl \
    && rm -rf /var/lib/apt/lists/*

# Installs pip.
RUN pip install --upgrade pip && pip install setuptools

### Google CLoud SDK
# Copied from https://cloud.google.com/ai-platform/training/docs/train-ml-model-custom-containers#create_a_custom_container
# Installs cloudml-hypertune for hyperparameter tuning.
# It’s not needed if you don’t want to do hyperparameter tuning.
RUN pip install cloudml-hypertune
# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup
# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg
### Google CLoud SDK END

WORKDIR /root/app
# Copies the trainer code 
COPY script ./script
COPY Credentials .

RUN pip install --no-cache-dir -r ./docker_requirements.txt

ENTRYPOINT [ "python3", "./script/trainer.py"]
