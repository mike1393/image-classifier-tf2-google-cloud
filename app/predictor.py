#Third-party
from flask import Flask, request, redirect, render_template
from werkzeug.utils import secure_filename
import numpy as np
#Built-in
import os
#Local
from utils import parse_function, predict_json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/mike/project/image_classifier_tf2_google_cloud/image-classifier-tf2-google-cloud/trainer/Credentials/sa-private-key.json" # change for your GCP key
FOOD_CLASSES = [
    "Bread", "Dairy_product", "Dessert", "Egg", "Fried_food", 
    "Meat", "Noodles_or_Pasta", "Rice", "Seafood", "Soup", "Vegetable_or_Fruit"]
PROJECT = "cloud-image-classifier-tf2"
REGION = "asia-east1"
MODEL = "kaggle_food_11_savedmodel"

app = Flask(__name__, static_url_path="/static")
app.config["IMAGE_UPLOADS"] = "./static"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = [".JPEG", ".JPG", ".PNG"]



def is_allowed_file(filename):
    if filename=="":
        print("[ERROR] File path is empty.")
        return False
    ext = os.path.splitext(filename)[-1].upper()
    if ext in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        print("[DEBUG] File Allowed")
        return True
    print(f"[ERROR] File extension:{ext} NOT allowed.")
    return False

def parse_prediction(probability_list):
    predicted_result = {}
    predicted_result["result"] = FOOD_CLASSES[np.argmax(probability_list)]
    probability_list = list(np.round(probability_list*100, 2))[0]
    predicted_result["probabilities"] = dict(zip(FOOD_CLASSES, probability_list))
    predicted_result["result"] = FOOD_CLASSES[np.argmax(probability_list)]
    return predicted_result

def make_prediction(image_file, project, region, model):
    """
    Takes an image and uses model (a trained TensorFlow model) to make a
    prediction.
    Returns:
     image (preproccessed)
     pred_class (prediction class from class_names)
     pred_conf (model confidence)
    """
    image = parse_function(image_file)
    
    # image = tf.expand_dims(image, axis=0)
    preds = predict_json(project=project,
                         region=region,
                         model=model,
                         instances=image)
    return parse_prediction(preds)

@app.route("/",methods=["GET","POST"])
def upload_image():
    
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            if is_allowed_file(image.filename):
                filename = secure_filename(image.filename)
                image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))
                return redirect(f"/showing-image/{filename}")
            else:
                return redirect(request.url)
    
    return render_template("upload_image.html")

@app.route("/showing-image/<image_name>", methods=["GET","POST"])
def showing_image(image_name):
    if request.method == "POST":
        image_path = os.path.join(app.config["IMAGE_UPLOADS"], image_name)
        predicted_result = make_prediction(image_path,PROJECT,REGION,MODEL)
        print(f"[DEBUG] predicted_result: {predicted_result}")
        return render_template(
            "prediction_result.html",
             image_name=image_name,
             predicted_result=predicted_result["result"],
             probabilities = predicted_result["probabilities"])

    return render_template("showing_image.html", image_name=image_name)

if __name__=="__main__":
    app.run(host='0.0.0.0',port=int(os.environ.get('PORT',8080)), debug=True)