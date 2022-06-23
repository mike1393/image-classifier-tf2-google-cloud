#!/bin/bash

############################################################

## Shell Script to Build Docker Image for Google AI Platform  

#############################################################
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
IMAGE_REPO_NAME=food_ingredient_predictor
HOSTNAME=asia.gcr.io
IMAGE_URI=$HOSTNAME/$PROJECT_ID/$IMAGE_REPO_NAME
REGION=asia-east1

############################################################

## Submit a job to the cloud 

#############################################################

read -p "Do you want to SUBMIT the Image to google cloud with image: $IMAGE_URI? [yY|nN]:" -n 1 -r
echo  # (optional) move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1
fi

echo "Submitting Image: " $IMAGE_URI

gcloud builds submit --tag $IMAGE_URI

if [[ $? -ne 0 ]]; then
    echo "[FAIL] Image NOT submitted"
[[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1
else
    echo "[OK] Image Submitted"
fi