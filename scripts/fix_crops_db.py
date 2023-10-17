import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import config

from config import config
import sys
from pathlib import Path

from datasets import COCOImageDataset, SimpleDataset, DatasetSlice, COCOLeopardDataset, GroupDataset

from norppa_tools import compose_sequential, get_leopard_singletons, crop_label_step_sequential, load_pickle, curry, apply_pipeline_dataset, curry_sequential, print_topk_accuracy, save_pickle, print_step, resize_dataset
from reidentification.identify import identify, apply_geometric, encode_patches, encode_patches_single, getDISK, getKeyNetAffNetHardNet, getHessAffNetHardNet, extract_patches, extract_patches_single, encode_dataset, encode_single
from reidentification.find_matches import find_matches
from datetime import datetime
from skimage import color
import numpy as np

backend_folder = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(backend_folder))

from sql import *
from sql_adapter import *

def print_identity(x):
    print(x)
    return [x]

def crop_overwrite(input, conn):
    img, label = input
    labels = color.rgb2gray(np.asarray(img))
    labels = (labels > 0.1).astype(np.uint8)
    where = np.where(labels)
    y1, x1 = np.amin(where, axis=1)
    y2, x2 = np.amax(where, axis=1)

    if y1 <= 1 and x1 <= 1 and x2 >= img.width-1 and y2 >= img.height-1:
        print("No need for crop, skipping")
        return input
    
    print("crop bb: ", (x1, y1, x2, y2))
    img_cropped = img.crop((x1, y1, x2, y2))

    seal_id = label['class_id']
    img_path = label['file'].split("/")[-1]

    img_id = get_image(conn, seal_id, img_path)

    if img_id is not None:
        # print("Image exists")


        patch_ids, patch_coordinates = get_patch_coordinates(conn, img_id)
        c = conn.cursor()

        sql = """ UPDATE patches SET coordinates = %s WHERE patch_id = %s """
            


        for patch_id, patch_coordinate in zip(patch_ids, patch_coordinates):
            # print("old coordinate: ", patch_coordinate, [x*img.width for x in patch_coordinate])
            patch_coordinate[0] = (patch_coordinate[0]*img.width - x1)/img_cropped.width
            patch_coordinate[1] = (patch_coordinate[1]*img.width - y1)/img_cropped.width
            for i in range(2, 5):
                patch_coordinate[i] = patch_coordinate[i] * img.width/img_cropped.width
            
            c.execute(sql, (patch_coordinate, int(patch_id)))              
        conn.commit()

        c.close()
    
    # print("OVerwriting to ", label['file'])
    img_cropped.save(label['file'])
    return []
    


def create_pipeline(cfg):
    
    encode_pipeline = [
        compose_sequential(  
        curry(crop_overwrite, cfg["conn"])
        )
    ]
    return encode_pipeline
    

def process_datasets(datasets, topk=20):
    """
    Runs the tests on the specified datasets. 
    The dataset is specified by a 4-element tuple:
    - Dataset name
    - Dataset directory (or 2 directories for (database, query) division, otherwise leave-one-out is used on a single dataset)
    - A flag whether the images must be resized or used as is
    - Codebooks from which dataset to use. If None, the codebooks will be
    computed from the input dataset. Otherwise, will try to load the codebooks
    of the dataset specified here.

    For each dataset, runs complete pipeline (starting from feature extraction) and prints out re-identification accuracy.

    """
    for (dataset_name, dataset) in datasets:
        print()
        print(f"Dataset name: {dataset_name}")

        
        # print(f"Path to the dataset: {dataset.dataset_dir}")
        print(f"Dataset: {dataset}")
        print()

        cfg = {}

        ### TODO: Create psql connection here
        cfg["conn"] = create_connection()

        pipeline = create_pipeline(cfg)
            
        apply_pipeline_dataset(dataset, pipeline, True)
        

def main():
    ds = SimpleDataset("/app/mount/public/database/norppa")
    datasets = [  
                    ("norppa_database_segmented", 
                     DatasetSlice(ds, (6071, len(ds))))
                    #  ("norppa_database_pattern_unknown", 
                    #  SimpleDataset("/ekaterina/work/data/norppa_database_pattern_unknown"), 
                    #  None, 
                    #  "norppa_database_segmented_pattern")
                ]
    process_datasets(datasets)


if __name__ == "__main__":
    main()
