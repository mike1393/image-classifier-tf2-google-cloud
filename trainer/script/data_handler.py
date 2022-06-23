
"""
This data handler is dedicated for the food-11 dataset.
The dataset can be found in https://www.kaggle.com/datasets/vermaavi/food11
For more similar dataset, please visit https://mmspg.epfl.ch/downloads/food-image-datasets/
"""
# Third-party
from tabnanny import verbose
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import storage
# Built-in
import glob
import shutil
import os
import random
import json
from argparse import ArgumentParser
# Local

FOOD_CLASSES = [
    "Bread", "Dairy_product", "Dessert", "Egg", "Fried_food", 
    "Meat", "Noodles_or_Pasta", "Rice", "Seafood", "Soup", "Vegetable_or_Fruit"]

def arrange_data_into_classes(path_to_data:str):
    """ Reorganize the images into class folders

    Args:
        path_to_data: Path to the image folder
    """
    print(f"Processing folder: {path_to_data}")
    full_img_paths = glob.glob(os.path.join(path_to_data, "*.jpg"))
    data_length = len(full_img_paths)
    created_folder_count = 0
    for full_path in full_img_paths:
        filename = os.path.basename(full_path)
        pos = filename.find('_')
        if pos != -1:
            class_id = filename[:pos]
            target_path = os.path.join(path_to_data, FOOD_CLASSES[int(class_id)])
            if not os.path.isdir(target_path):
                created_folder_count+=1
                os.mkdir(target_path)
            shutil.move(full_path, target_path)
    print(f"Done! {data_length} files have been moved to {created_folder_count} class folders.")

def show_random_images(path_to_data:str, row:int, col:int):
    """Display random images with respective labels
    
    Use row and col to specify how many images to display.
    For example:
    show_random_images("foo/bar, 3,3")
    will display 9 images in a 3 by 3 fashion.

    Args:
        path_to_data: Path to the image folder
        row: Number of rows in the displayed table
        col: Number of columns in the displayed table
    """
    fig  = plt.figure()
    class_to_img_table = {folder:[] for folder in os.listdir(path_to_data)}
    for i in range(row*col):
        class_id = random.randint(0,10)
        class_dir = os.path.join(path_to_data, FOOD_CLASSES[class_id])
        if not class_to_img_table[FOOD_CLASSES[class_id]]:
            class_to_img_table[FOOD_CLASSES[class_id]] = [file for file in os.listdir(class_dir)]
        img_list = class_to_img_table[FOOD_CLASSES[class_id]]
        chosen_img = os.path.join(class_dir, img_list[random.randint(0,len(img_list)-1)])
        ax = fig.add_subplot(row,col,i+1)
        ax.title.set_text(FOOD_CLASSES[class_id])
        ax.imshow(Image.open(chosen_img))
        
    plt.tight_layout(pad=0.05)
    plt.show()

def get_image_size_info(path_to_data:str):
    """ Get Image width and height information from data folder

    Args:
        path_to_data: Path to the image folder
    Return:
        mean_height: The mean of all image heights in the folder
        mean_width: The mean of all image widths in the folder
        median_height: The median of all image heights in the folder
        median_width: The median of all image widths in the folder
    """
    print(f"Processing file: {path_to_data}")
    img_heights = []
    img_widths=[]
    for r,d,f in os.walk(path_to_data):
        for file in f:
            if file.endswith(".jpg"):
                full_file_path = os.path.join(r,file)
                img = Image.open(full_file_path)
                h,w = img.size
                img_heights.append(h)
                img_widths.append(w)
    mean_height = np.mean(img_heights)
    mean_width = np.mean(img_widths)
    median_height = np.median(img_heights)
    median_width = np.median(img_widths)
    print(f"Mean Height: {mean_height}, Mean Width:{mean_width}, Median Height: {median_height}, Median Width: {median_width}")
    return mean_height, mean_width, median_height, median_width

def copy_to_small_data_folder(path_to_data, dst, number_of_data):
    folder_type = os.path.basename(path_to_data)
    target_dir = os.path.join(dst,folder_type)
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    count=0
    for r,d,f in os.walk(path_to_data):
        if not f:
            continue
        class_name = os.path.basename(r)
        class_dir = os.path.join(target_dir,class_name)
        if not os.path.isdir(class_dir):
            os.mkdir(class_dir)
        for file in f:
            full_path = os.path.join(r,file)
            shutil.copy(full_path, class_dir)
            count+=1
            if count >=number_of_data:
                print(f"[INFO] Moved {count} files to {class_dir}")
                count=0
                break

def list_blobs(bucket_name, path_to_credential):
    # create client
    storage_client = storage.Client.from_service_account_json(path_to_credential)
    # get the list of blobs inside the buckets
    blobs = storage_client.list_blobs(bucket_name)
    return blobs


def download_to_local_folder(
    bucket_name:str, path_to_credential:str, 
    destination_folder_name:str, overwrite:bool=False, verbose=False):
    """
    """
    # create client
    storage_client = storage.Client.from_service_account_json(path_to_credential)
    # get the list of blobs inside the buckets
    blobs = storage_client.list_blobs(bucket_name)
    destination_folder_name = os.path.abspath(destination_folder_name)
    # Create the destination folder if not exist
    if not os.path.isdir(destination_folder_name):
        os.makedirs(destination_folder_name, exist_ok=True)
        overwrite = True
    
    metafile_path = os.path.join(destination_folder_name, "metadata.json")
    if not overwrite and os.path.isfile(metafile_path):
        with open(metafile_path) as f:
            metadata = json.load(f)
    else:
        if overwrite:
            print("[INFO] Overwriting folder and metadata")
        else:
            print(f"[INFO] metafile: {metafile_path} does not exist.")
        metadata = {}
        metadata["download_path"]=""
        print("[INFO] Creating new metadata")
    
    print(f"[INFO] Downloading files from {bucket_name}...")
    for blob in blobs:
        if "image" not in blob.content_type:
            continue
        if not overwrite and blob.name in metadata and metadata[blob.name] == str(blob.updated):
            continue

        # Download Procedure
        blob_dir = os.path.dirname(blob.name)
        if len(metadata["download_path"])==0:
            head = os.path.sep + [part for part in blob_dir.split(os.path.sep) if part][0]
            metadata["download_path"] = os.path.join(destination_folder_name + head)
            
        # prefix the destinateion folder to blob directory
        prefixed_dir = os.path.join(destination_folder_name, blob_dir)
        # create the prefixed folder if not exist
        if not os.path.isdir(prefixed_dir):
            os.makedirs(prefixed_dir, exist_ok=True)
        blob_filename = os.path.basename(blob.name)
        destination_file_name = os.path.join(prefixed_dir,blob_filename)
        blob.download_to_filename(destination_file_name)
        metadata[blob.name] = str(blob.updated)
        if verbose:
            print(f"[DEBUG] Downloaded storage object {blob.name} from bucket {bucket_name} to local file {destination_file_name}")
    
    print("[INFO] Download Completed")

    with open(metafile_path, "w") as f:
        if verbose:
            print(f"[DEBUG] Writing metadata to {metafile_path}")
        json.dump(metadata, f, indent = 4)
    download_path = metadata["download_path"]
    print(f"[INFO] Data has been stored in {download_path}")
    return download_path

def upload_data_to_bucket(
    bucket_name:str, path_to_credential:str, blob_name:str,
    source_filename:str, verbose=False):
    storage_client = storage.Client.from_service_account_json(path_to_credential)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(source_filename)
    print(f"Uploaded {source_filename} to gs:\\{bucket_name}")




if __name__ == "__main__":
    arrange_data_usage = "Arrange data by classes: -a [-src source file]\n"
    show_data_usage = "Display random image: -sh [-src source file] [Optional -r -cl]\n"
    data_info_usage = "Get Data Info: -i [-src source file]\n"
    split_data_usage = "Split data into smaller sample: -s [-src source file] [-dst destination file] [Optional -sa]\n"
    list_blob_usage = "List all blobs in the bucket: -lb [-b bucket name] [-c credential file]\n"
    download_usage = "Download data from google bucket: -dl [-b bucket_name] [-c credential file] [-dst destination folder] [Optional -ow]\n"
    
    parser = ArgumentParser()
    operation_group = parser.add_mutually_exclusive_group(required=True)
    operation_group.add_argument('-a','--arrange', action='store_true', help = arrange_data_usage)
    operation_group.add_argument('-sh','--show', action='store_true', help = show_data_usage)
    operation_group.add_argument('-i','--info', action='store_true',help=data_info_usage)
    operation_group.add_argument('-s','--split', action='store_true',help=split_data_usage)
    operation_group.add_argument('-lb','--listblob', action='store_true',help=list_blob_usage)
    operation_group.add_argument('-dl','--download', action='store_true',help=download_usage)
    file_group = parser.add_argument_group("File | Folder flags", "Should be used accordling with every operation")
    file_group.add_argument('-src','--source',type=str,metavar="")
    file_group.add_argument('-dst','--destination',type=str,metavar="")
    file_group.add_argument('-c','--credential',type=str, metavar="", help="credential file path")
    cloud_group = parser.add_argument_group("Cloud flags", "Should be used when accessing google buckets")
    cloud_group.add_argument('-b','--bucket',type=str, metavar="", help="bucket name")
    optional_group = parser.add_argument_group("Optional flags", "Used for special operations")
    optional_group.add_argument('-ow','--overwrite',action='store_true',default=False, help="Overwrite downloaded data")
    optional_group.add_argument('-r','--row',type=int,default=4, metavar="",help="rows for displaying images")
    optional_group.add_argument('-cl','--col',type=int,default=4, metavar="",help="cols for displaying images")
    optional_group.add_argument('-sa','--splitamount',type=int,default=32, metavar="",help="The amount to copy for evey classes")

    args = parser.parse_args()

    arrange_data_switch = args.arrange
    show_data_switch = args.show
    print_img_info_switch=args.info
    split_to_small_data = args.split
    list_blobs_switch = args.listblob
    download_switch = args.download

    if arrange_data_switch:
        if args.source is None:
            print("Argument -src or --source required, use -h or --help for more information.")
        else:
            file_source = args.source
            print(f"Arranging data in folder: {file_source} into classes")
            arrange_data_into_classes(file_source)
    if show_data_switch:
        if args.source is None:
            print("Argument -src or --source required, use -h or --help for more information.")
        else:
            file_source = args.source
            row,col = args.row, args.col
            print(f"Showing {row*col} random images from data :{file_source}")
            show_random_images(file_source,row,col)
    if print_img_info_switch:
        if args.source is None:
            print("Argument -src or --source required, use -h or --help for more information.")
        else:
            file_source = args.source
            print(f"Showing data info from folder: {file_source}")
            get_image_size_info(file_source)
    if split_to_small_data:
        if args.source is None or args.destination is None:
            print("Make sure -src and -dst is added, for more information see -h or --help")
        else:
            file_source = args.source
            file_destination = args.destination
            split_amount = args.splitamount
            if not os.path.isdir(file_destination):
                os.mkdir(file_destination)
            copy_to_small_data_folder(file_source, file_destination,split_amount)
        
    if list_blobs_switch:
        if args.bucket is None or args.credential is None:
            print("Make sure -b and -c is added, for more information see -h or --help")
        else:
            bucket_name = args.bucket
            credential_file = args.credential
            blobs = list_blobs(bucket_name,credential_file)
            for blob in blobs:
                print(blob.name)

    if download_switch:
        if args.bucket is None or args.credential is None or args.destination is None:
            print("Make sure -b, -c and -dst is added, for more information see -h or --help")
        else:
            bucket_name = args.bucket
            credential_file = args.credential
            destination_folder = args.destination
            overwrite = args.overwrite
            download_to_local_folder(
                bucket_name=bucket_name, destination_folder_name=destination_folder,
                path_to_credential=credential_file, overwrite = overwrite)