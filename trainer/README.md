# Trainer
## Building ML Model
The deep learning framework I used for this project is TensorFlow. If you are interested in how to use TensorFlow, feel free to check out my other [repo](https://github.com/mike1393/intro-to-tensorflow2.0-python).

Since training a model requires huge amount of data, I used transfer learning. The base model I used is InceptionV3, which trained on Imagenet dataset.

## Data Pipelining
The way I built my pipeline is heavily influenced by [Andrew Ng's artical](https://cs230.stanford.edu/blog/datapipeline/#best-practices), please check it out if you're interested.

Building a good data pipeline is crusial for effective training. First, we want to make sure we kept all the GPU powers for training, and CPU powers for fetching data. By doing so, we shorten the training time for each epoch, since we could now fetch the data as we train the model. Second, to prevent overfitting, we can add data augmentation layer to the pipeline.

To verify my trainig pipeline, I trained the model locally with a small set of data with small epochs.
## Dockerize
Since the goal of dockerization is to run the same app everywhere as running locally, there cannot be any hard-coded file path. Hence, before I dockerize the training pipeline, I created argument groups for the trainer using Argparse package.

The Dockerfile for the trainer can be found under ```/trainer```folder. The base image I used was ```tensorflow/tensorflow:latest-gpu``` to make sure the docker environment was set correctly for TensorFlow. I also installed Google Cloud API packages for later use.
