# Image Classifier with TensorFlow2 and Google Cloud Platform
## Goal
The goal of this repo is to learn two things:
1. How to train a neural network on the cloud.
2. How to serve the model on the cloud.

Hence, in this project, I will:
1. Create a training pipeline using transfer learning with a custom dataset.
2. Train the model on Google Cloud AI Platform
3. Create a web app that interacts with the model prediction
4. Deploy the web app through Google Cloud Run.

## File Structure
The trainer folder contains scripts and dockerfile for the training pipeline. For more detail, checkout [trainer/README.md](trainer).

The app folder contains the web app for classifing images using the model prediction. For more detail, checkout [app/README.md](app).

## Cloud Computing
![Project Flowchart](https://drive.google.com/uc?id=1ChATklAh-Kmp8yv_gIKH9d6g1JxK4vnT)

## Data
For simplicity, the dataset I used for this project is food-11, containing 16,643 food images grouped in 11 major food categories. You can download the dataset at [Kaggle](https://www.kaggle.com/datasets/vermaavi/food11), which originates from [EPFL-Food Image Dataset](https://www.epfl.ch/labs/mmspg/downloads/food-image-datasets/).

The dataset comes in three folders, Evaluate, Train, and Validation. There are, in total, 11 categories. Each image file has a naming format of ```<class_id>_<img_id>.jpg```. For example, 0_1.jpg means the first image in class 0 (Bread). I further processed the dataset by grouping them into classes to simplify later operations.

<table>
<tr>
<th>Original Dataset</th>
<th>Preprocessed Dataset</th>
</tr>
<tr>
<td>

```text
/food-11
  |_/evaluate
  |   |_/0_1.jpg
  |     /0_2.jpg
  |   ...
  |_/training
  |_/validation
```
  
</td>
<td>

```text
/food-11
  |_/evaluate
  |   |_/Bread
  |       |_/0_1.jpg
  |         /0_2.jpg
  |   ...
  |_/training
  |_/validation
```

</td>
</tr>
</table>


