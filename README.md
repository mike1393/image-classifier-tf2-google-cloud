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
For simplicity, the dataset I used for this project is food-11, which contains 16643 food images grouped in 11 major food categories. The dataset can be downloaded at [Kaggle](https://www.kaggle.com/datasets/vermaavi/food11), which originates from [EPFL-Food Image Dataset](https://www.epfl.ch/labs/mmspg/downloads/food-image-datasets/).

The dataset comes with three folders, Evaluate, Train and Validation. There are in total 11 categories. Each image file has a naming format of ```<class_id>_<img_id>.jpg```. For example, 0_1.jpg means the first image in class 0 (Bread). I further processed the dataset by grouping them in classess to simplify later operations.

<table>
<tr>
<th>Original Dataset</th>
<th>Preprocessed Dataset</th>
</tr>
<tr>
<td>

~~~text
/food-11
  |_/evaluate
  |   |_/0_1.jpg
  |     /0_2.jpg
  |   ...
  |_/training
  |_/validation
~~~
  
</td>
<td>

~~~text
/food-11
  |_/evaluate
  |   |_/Bread
  |       |_/0_1.jpg
  |         /0_2.jpg
  |   ...
  |_/training
  |_/validation
~~~

</td>
</tr>
</table>
