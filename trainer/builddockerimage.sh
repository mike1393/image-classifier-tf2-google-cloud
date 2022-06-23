#!/bin/bash

############################################################

## Shell Script to Build Docker Image for Google AI Platform  

#############################################################
VERSION=6
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
IMAGE_REPO_NAME=food_image_classifier_trainer
TAG=food11_tf2_gpu$VERSION
HOSTNAME=asia.gcr.io
IMAGE_URI=$HOSTNAME/$PROJECT_ID/$IMAGE_REPO_NAME:$TAG
REGION=asia-east1
IMAGE_EXIST=$(docker images -q $IMAGE_URI 2> /dev/null)
############################################################

## Create Image 

#############################################################
if [[ "$IMAGE_EXIST" == "" ]]; then
    echo "Create Image URI:" $IMAGE_URI
    read -p "Do you want to build the image? [yY|nN]:" -n 1 -r
    echo    # (optional) move to a new line
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1
    fi

    docker build -f Dockerfile -t $IMAGE_URI .
    if [[ $? -ne 0 ]]; then
        echo "[FAIL] Image NOT Built"
        [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1
    else
        echo "[OK] Image Built"
    fi
else
    echo "$IMAGE_URI already built"
fi
############################################################

## Run image locally

#############################################################
read -p "Do you want to RUN the IMAGE LOCALLY? [yY|nN]:" -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker run --gpus all -it $IMAGE_URI \
        --bucket="kaggle-food-11-small-data" \
        --model_bucket="kaggle-food-11-saved-model" \
        --credential="./Credentials/sa-private-key.json"
    if [[ $? -ne 0 ]]; then
        echo "[FAIL] Error Occurs"
        [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1
    else
        echo "[OK]"
        read -p "Do you want to PRUNE the image: $IMAGE_URI? [yY|nN]:" -n 1 -r
        echo  # (optional) move to a new line
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1
        else
            docker container prune
        fi
    fi
fi
############################################################

## Pushing image to the cloud 

#############################################################

read -p "Do you want to PUSH the IMAGE to google cloud? [yY|nN]:" -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker push $IMAGE_URI
    if [[ $? -ne 0 ]]; then
        echo "[FAIL] Image NOT Pushed"
        [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1
    else
        echo "[OK] Image Pushed"
    fi
fi

############################################################

## Submit a job to the cloud 

#############################################################

read -p "Do you want to SUBMIT the JOB to google cloud with image: $IMAGE_URI? [yY|nN]:" -n 1 -r
echo  # (optional) move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1
fi
JOB_NAME=food_image_classifier_job_gpu_$(date +%Y%m%d_%H%M%S)
echo "Submitting job: " $JOB_NAME

gcloud ai-platform jobs submit training $JOB_NAME \
  --scale-tier BASIC_GPU \
  --region $REGION \
  --master-image-uri $IMAGE_URI \
  --config config.yaml \
  -- \
  --hyper_tune \
  --bucket="kaggle-food-11-bucket" \
  --model_bucket="kaggle-food-11-saved-model" \
  --credential="./sa-private-key.json" \
  --batch_size=64 \
  --dropout_rate=0.65 \
  --epochs=20

if [[ $? -ne 0 ]]; then
    echo "FAIL Job NOT submitted"
[[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1
else
    echo "OK Job Submitted"
fi




