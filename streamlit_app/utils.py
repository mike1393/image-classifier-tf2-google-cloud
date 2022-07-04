import tensorflow as tf
import googleapiclient.discovery
from google.api_core.client_options import ClientOptions
import numpy as np

def parse_function(filepath):
    img = tf.image.decode_image(filepath,channels=3)
    img = tf.image.convert_image_dtype(img,tf.float32)
    img = tf.image.resize(img,[299,299])
    img = tf.expand_dims(img, axis=0)
    # Turn tensors into int16 (saves a lot of space, ML Engine has a limit of 1.5MB per request)
    img = tf.cast(img, tf.int16)
    return img

def parse_prediction(probability_list, classes):
    result = classes[np.argmax(probability_list)]
    probability_list = np.array(probability_list[0])
    print(type(probability_list))
    probability_list = np.round(probability_list*100, 2)
    confidence = dict(zip(classes, probability_list))
    return result, confidence

def predict_json(project, region, model, instances, version=None):
    """Send json data to a deployed model for prediction.
    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to Tensors.
        version (str): version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the 
            model.
    """
    # Create the ML Engine service object
    prefix = "{}-ml".format(region) if region else "ml"
    api_endpoint = "https://{}.googleapis.com".format(prefix)
    client_options = ClientOptions(api_endpoint=api_endpoint)

    # Setup model path
    model_path = "projects/{}/models/{}".format(project, model)
    if version is not None:
        model_path += "/versions/{}".format(version)

    # Create ML engine resource endpoint and input data
    ml_resource = googleapiclient.discovery.build(
        "ml", "v1", cache_discovery=False, client_options=client_options).projects()
    instances_list = instances.numpy().tolist() # turn input into list (ML Engine wants JSON)
    
    input_data_json = {"signature_name": "serving_default",
                       "instances": instances_list} 

    request = ml_resource.predict(name=model_path, body=input_data_json)
    response = request.execute()
    
    # # ALT: Create model api
    # model_api = api_endpoint + model_path + ":predict"
    # headers = {"Authorization": "Bearer " + token}
    # response = requests.post(model_api, json=input_data_json, headers=headers)

    if "error" in response:
        raise RuntimeError(response["error"])

    return response["predictions"]