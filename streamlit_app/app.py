#Third Party
import streamlit as st
#Built-in
import os
import json
from argparse import ArgumentParser
#Local
from utils import parse_function, predict_json, parse_prediction
FOOD_CLASSES= [
    "Bread", "Dairy_product", "Dessert", "Egg", "Fried_food", 
    "Meat", "Noodles_or_Pasta", "Rice", "Seafood", "Soup", "Vegetable_or_Fruit"]
@st.cache
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
    return preds

if __name__=="__main__":
    # parser = ArgumentParser()
    # file_group = parser.add_argument_group("File | Folder flags", "Should be used accordling with every operation")
    # parser.add_argument('--config','-c',type=str, metavar="", help="sets environment config file", required=True)
    # args = parser.parse_args()
    # # config_file_path = "/home/mike/project/image_classifier_tf2_google_cloud/image-classifier-tf2-google-cloud/streamlit_app/config.json"
    # config_file_path = args.config
    with open("config.json", 'r') as f:
        os.environ.update(json.load(f))
    st.session_state.project = os.environ["PROJECT"]
    st.session_state.region = os.environ["REGION"]
    st.session_state.model = os.environ["MODEL"]
    st.title("Welcome to Food Classification🎉")
    st.header("Let's see what's on your plate! 👀")
    # Display info about classes
    if st.checkbox("Show classes"):
        st.write(f"These are the classes of food it can identify:\n", FOOD_CLASSES)
    # File uploader allows user to add their own image
    uploaded_file = st.file_uploader(label="Upload an image of food",
                                    type=["png", "jpeg", "jpg"])

    if not uploaded_file:
        st.warning("Please upload an image.")
        st.stop()
    st.session_state.uploaded_image = uploaded_file.read()
    st.image(st.session_state.uploaded_image, use_column_width=True)
    pred_button = st.button("Predict")

    # Did the user press the predict button?
    if pred_button:
        st.session_state.pred_button = True 
        st.session_state.preds = make_prediction(st.session_state.uploaded_image, st.session_state.project, st.session_state.region, st.session_state.model)
        st.session_state.result, st.session_state.confidence = parse_prediction(st.session_state.preds, FOOD_CLASSES)
        
        st.subheader(f"Prediction: {st.session_state.result}")
        for label, conf in st.session_state.confidence.items():
            st.caption(f"{label}: {conf}%")


    
