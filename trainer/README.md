# Trainer
## Building ML Model
The deep learning framework I used for this project is TensorFlow. If you are interested in how to use TensorFlow, feel free to check out my other [repo](https://github.com/mike1393/intro-to-tensorflow2.0-python).

Since training a model requires massive data, I used transfer learning. The base model I used is InceptionV3 and was set to train on the Imagenet dataset.

## Data Pipelining
The way I built my pipeline is heavily influenced by [Andrew Ng's article](https://cs230.stanford.edu/blog/datapipeline/#best-practices). Please check it out if you're interested.

Building a promising data pipeline is crucial for effective training. First, we want to ensure we kept all the GPU powers for training and CPU powers for fetching data. In doing so, we shortened the training time for each epoch since we could now fetch the data as we trained the model. Second, we can add data augmentation layers to the pipeline to prevent overfitting.

To verify my training pipeline, I trained the model locally with a small set of data with small epochs.
## Dockerize
Since dockerization aims to run the same app everywhere as locally, there cannot be any hard-coded file path. Hence, before I dockerize the training pipeline, I created argument groups for the trainer using the Argparse package.

The Dockerfile for the trainer can be found under the ```/trainer```folder. The base image I used was ```tensorflow/tensorflow:latest-gpu``` to ensure we set the docker environment correctly for TensorFlow. I also installed Google Cloud API packages for later use.
