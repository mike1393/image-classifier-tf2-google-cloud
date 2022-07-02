#Third-party package
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import hypertune
#Built-in package
import os
from argparse import ArgumentParser
import shutil
from datetime import datetime
#local package
from data_handler import download_to_local_folder, upload_data_to_bucket


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

FOOD_CLASSES = [
    "Bread", "Dairy_product", "Dessert", "Egg", "Fried_food", 
    "Meat", "Noodles_or_Pasta", "Rice", "Seafood", "Soup", "Vegetable_or_Fruit"]

def show_device_log():
    print("[INFO] TensorFlow is running on following devices: ")
    print(device_lib.list_local_devices())

def parse_function(filepath, label):
    img_string = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img_string,channels=3)
    img = tf.image.convert_image_dtype(img,tf.float32)
    img = tf.image.resize(img,[299,299])
    return img, label

def build_dataset_from_structured_file(batch_size:int, path_to_data:str, training:bool=False,verbose:bool=False):
    AUTOTUNE = tf.data.AUTOTUNE
    full_img_paths = []
    labels =[]
    number_of_classes = len(os.listdir(path_to_data))
    for r,d,f in os.walk(path_to_data):
        for file in f:
            pos = file.find("_")
            class_id = int(file[:pos])
            if pos !=-1:
                labels.append(class_id)
                full_img_paths.append(os.path.join(r,file))
    data_length = len(full_img_paths)
    labels = tf.keras.utils.to_categorical(labels, number_of_classes)
    
    dataset = tf.data.Dataset.from_tensor_slices((full_img_paths, labels))
    if verbose:
        print(f"[DEBUG] Dataset with {data_length} files have been created")
    
    if training:
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1,fill_mode='nearest'),
            layers.RandomTranslation(height_factor=0.2,width_factor=0.2,fill_mode='nearest'),
            layers.RandomZoom(height_factor=0.15, fill_mode='nearest')])
        dataset = dataset.shuffle(len(full_img_paths))
        dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)
        dataset = dataset.map(lambda x,y: (data_augmentation(x,training=True),y), num_parallel_calls=AUTOTUNE)
    else:
        dataset = dataset.map(parse_function,num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size = AUTOTUNE)
    return dataset
    
def get_datasets(batch_size:int, path_to_train:str, path_to_val:str, path_to_eval:str, verbose:bool=False):
    train_ds = build_dataset_from_structured_file(batch_size, path_to_train, True, verbose)
    val_ds = build_dataset_from_structured_file(batch_size, path_to_val, False,verbose)
    eval_ds = build_dataset_from_structured_file(batch_size, path_to_eval, False,verbose)
    return train_ds, val_ds, eval_ds

def build_model(base_model, dense_unit:int, dropout_rate:float):
    base_model.trainable=False
    input = layers.Input(shape=(299,299,3))
    x = base_model(input, training=False)
    x = layers.Flatten()(x)
    x = layers.Dense(dense_unit, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(len(FOOD_CLASSES), activation='softmax')(x)
    return Model(inputs=input, outputs=x)

def get_callbacks(path_to_save_model):
    callback_list = []
    early_stop = EarlyStopping(monitor="val_accuracy", patience=10)
    callback_list.append(early_stop)
    checkpoint_saver = ModelCheckpoint(
        filepath=path_to_save_model,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_freq="epoch",
        verbose=1)
    callback_list.append(checkpoint_saver)
    return callback_list

def zip_model(eval_loss, path_to_saved_model:str):
    current_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    zipped_model_name = f"loss_{eval_loss}_saved_model_{current_datetime}"
    shutil.make_archive(base_name=zipped_model_name, format='zip',root_dir=path_to_saved_model)
    dirname = os.path.dirname(path_to_saved_model)
    zipped_model_path = os.path.join(dirname, zipped_model_name + '.zip')
    print(f"zipped to {zipped_model_path}")
    return zipped_model_path, zipped_model_name

def train_model(
    model, path_to_data:str, batch_size:int, 
    epochs:int,learning_rate:float,
    model_bucket_name:str,path_to_credential:str,
    hyper_tune:bool=False,verbose:bool=False):
    """
    """
    path_to_train = os.path.join(path_to_data, "training")
    path_to_val = os.path.join(path_to_data, "validation")
    path_to_eval = os.path.join(path_to_data, "evaluation")
    train_ds, val_ds, eval_ds = get_datasets(batch_size, path_to_train, path_to_val, path_to_eval,verbose)
    # Compile the model
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer = opt, loss='categorical_crossentropy', metrics = ['accuracy'])
    
    print("[INFO] Training Model...")
    # Create callbacks
    path_to_saved_model = os.path.abspath("./saved_model")
    if not os.path.isdir(path_to_saved_model):
        os.makedirs(path_to_saved_model, exist_ok=True)
    callback_list = get_callbacks(path_to_saved_model)
    # Fit the model
    history = model.fit(
        train_ds,
        batch_size = batch_size,
        epochs = epochs,
        validation_data = val_ds,
        callbacks=callback_list,
        verbose = 2)

    print("[INFO] Constructing classification report...")
    pred = model.predict(eval_ds)
    pred_id = np.argmax(pred, axis=1)
    true_categories = tf.concat([y for x, y in eval_ds], axis=0)
    true_id = np.argmax(true_categories, axis=1)
    cls_report = classification_report(true_id,pred_id, target_names = FOOD_CLASSES)
    print(cls_report)
    print("[INFO] Constructing confusion matrix...")
    conf_mtx = confusion_matrix(true_id,pred_id)
    print(conf_mtx)
    print("[INFO] Evaluating Model...")
    result = model.evaluate(eval_ds)
    print(dict(zip(model.metrics_names, result)))
    #uploading saved model to bucket
    eval_loss = result[0]
    zipped_model_path, zipped_model_name = zip_model(eval_loss, path_to_saved_model)
    upload_data_to_bucket(model_bucket_name, path_to_credential, zipped_model_name, zipped_model_path)
    #update hyper tune metrics
    if hyper_tune:
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(hyperparameter_metric_tag="hpt_loss", metric_value=eval_loss, global_step=epochs)

    return history, cls_report, conf_mtx



if __name__=="__main__":
    parser = ArgumentParser()
    cloud_group = parser.add_argument_group("Cloud flags", "Should be used when accessing google buckets")
    cloud_group.add_argument('-b','--bucket',type=str, metavar="bucket name", help="bucket name", required=True)
    cloud_group.add_argument('-mb','--model_bucket',type=str, metavar="model bucket name", help="saved model bucket name", required=True)

    file_group = parser.add_argument_group("File | Folder flags", "Should be used accordling with every operation")
    file_group.add_argument('-c','--credential',type=str, metavar="credential file path", help="credential file path", required=True)
    file_group.add_argument('-dst','--destination',type=str,default="./download",metavar="folder path",help="destination folder path for downloaded data")
    
    hp_group = parser.add_argument_group("Hyperparameters")
    hp_group.add_argument('-hpt','--hyper_tune',action='store_true',default=False,help="[Model] If hyper tune is enabled")
    hp_group.add_argument('-du','--dense_unit',type=int,default=512, metavar=" dense unit",help="[Model] number of units in dense layer")
    hp_group.add_argument('-dpr','--dropout_rate',type=float,default=0.5, metavar=" dropout rate",help="[Model] Dropout rate after Dense layer")
    hp_group.add_argument('-bs','--batch_size',type=int,default=32, metavar=" batch size",help="[Training] number of batch size")
    hp_group.add_argument('-e','--epochs',type=int,default=2, metavar="number of epochs",help="[Training] number of epochs ")
    hp_group.add_argument('-lr','--learning_rate',type=float,default=1e-5, metavar="learning rate",help="[Training] optimizer learning rate ")
    
    optional_group = parser.add_argument_group("Optional flags")
    optional_group.add_argument('-ow','--overwrite',action='store_true',default=False, help="Overwrite downloaded data")
    optional_group.add_argument('-v','--verbose',action='store_true', default=False, help="Show all level of logs")
    args = parser.parse_args()

    bucket_name = args.bucket
    model_bucket_name = args.model_bucket
    path_to_credential = args.credential
    destination_folder_name = args.destination
    hyper_tune = args.hyper_tune
    dense_unit = args.dense_unit
    dropout_rate = args.dropout_rate
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    overwrite = args.overwrite
    verbose = args.verbose

    if verbose:
        show_device_log()
    path_to_local_data = download_to_local_folder(
        bucket_name = bucket_name,
        path_to_credential = path_to_credential,
        destination_folder_name=destination_folder_name,
        overwrite=overwrite)
    
    base_model = InceptionV3(include_top=False,weights='imagenet',input_shape = (299,299,3))
    model = build_model(
        base_model, 
        dense_unit=dense_unit,
        dropout_rate=dropout_rate)

    if verbose:
        model.summary()
        print(f"[DEBUG] Training data from {path_to_local_data}")
    print(f"\n[INFO] Building model with {base_model.name}")
    
    train_model(
        model,
        path_to_data=path_to_local_data,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        model_bucket_name=model_bucket_name,
        path_to_credential = path_to_credential,
        hyper_tune = hyper_tune,
        verbose=verbose)
    
