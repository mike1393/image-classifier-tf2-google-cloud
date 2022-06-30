# Image Classifier with TensorFlow2 and Google Cloud Platform
## Goal
The goal of this repo is to learn two things:
1. How to deploy a neural network model.
2. How to serve the prediction on the cloud.

Hence, in this project, I will:
1. Create a training pipeline using transfer learning with custom dataset.
2. Train the model on Google Cloud AI Platform
3. Create a web app that interacts with the model prediction
4. Deploy the web app through Google Cloud Run.

## Data
For simplicity, the dataset I used for this project is food-11, which contains 16643 food images grouped in 11 major food categories. The dataset can be download at [Kaggle](https://www.kaggle.com/datasets/vermaavi/food11), which originates from [EPFL-Food Image Dataset](https://www.epfl.ch/labs/mmspg/downloads/food-image-datasets/).
