#Third-party
from flask import Flask, request, redirect, render_template
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
#Built-in
import os
#Local

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
FOOD_CLASSES = [
    "Bread", "Dairy_product", "Dessert", "Egg", "Fried_food", 
    "Meat", "Noodles_or_Pasta", "Rice", "Seafood", "Soup", "Vegetable_or_Fruit"]

app = Flask(__name__, static_url_path="/static")
app.config["IMAGE_UPLOADS"] = "./static"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = [".JPEG", ".JPG", ".PNG"]
food_prediction_model = tf.keras.models.load_model("./savedmodel")


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

def parse_function(filepath):
    img_string = tf.io.read_file(filepath)
    img = tf.image.decode_image(img_string,channels=3)
    img = tf.image.convert_image_dtype(img,tf.float32)
    img = tf.image.resize(img,[299,299])
    img = tf.expand_dims(img, axis=0) 
    return img

def get_prediction(image):
    predicted_result = {}
    probability_list = food_prediction_model.predict(image)
    predicted_result["result"] = FOOD_CLASSES[np.argmax(probability_list)]
    probability_list = list(np.round(probability_list*100, 2))[0]
    predicted_result["probabilities"] = dict(zip(FOOD_CLASSES, probability_list))
    predicted_result["result"] = FOOD_CLASSES[np.argmax(probability_list)]
    return predicted_result


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
        image = parse_function(image_path)
        predicted_result = get_prediction(image)
        print(f"[DEBUG] predicted_result: {predicted_result}")
        return render_template(
            "prediction_result.html",
             image_name=image_name,
             predicted_result=predicted_result["result"],
             probabilities = predicted_result["probabilities"])

    return render_template("showing_image.html", image_name=image_name)

if __name__=="__main__":
    app.run(host='0.0.0.0',port=int(os.environ.get('PORT',8080)), debug=True)