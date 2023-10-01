import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import config

from config import config

from sql import *
from datasets import COCOImageDataset, SimpleDataset, DatasetSlice, COCOLeopardDataset, GroupDataset

from norppa_tools import compose_sequential, get_leopard_singletons, crop_label_step_sequential, load_pickle, curry, apply_pipeline_dataset, curry_sequential, print_topk_accuracy, save_pickle, print_step, resize_dataset
from reidentification.identify import identify, apply_geometric, encode_patches, encode_patches_single, getDISK, getKeyNetAffNetHardNet, getHessAffNetHardNet, extract_patches, extract_patches_single, encode_dataset, encode_single
from reidentification.find_matches import find_matches
from datetime import datetime

def update_codebooks(input, cfg, save_path=None):
    """
    A pipeline step used after encode_dataset/encode_patches with compute_codebooks=True.
    Accepts a tuple of form (codebooks, encoded), updates config dict with provided 
    codebooks and returns encoded.
    If save_path is not None, saves the codebooks to the provided path
    """
    (codebooks, encoded) = input
    cfg["codebooks_path"] = None
    cfg["codebooks"] = codebooks
    if save_path is not None:
        save_pickle(codebooks, save_path)
    return encoded

def load_codebooks(input, cfg, load_path=None):
    """
    A pipeline step that loads codebooks and updates the config dict.
    Returns input without modification.
    """
    cfg["codebooks_path"] = None
    cfg["codebooks"] = load_pickle(load_path)
    return input

def update_coordinates(input, conn):
    if input is None:
        return [None]
    img, label = input

    seal_id = label['class_id'].split("_")[0].lower()
    seal_name = ""
    if len(label['class_id'].split("_")) > 1:
        seal_name = label['class_id'].split("_")[1]
    img_path = label['file'].split("/")[-1]
    viewpoints = {"right": False, "left": False, "up": False, "down": False}

    if label.get('viewpoint', 'unknown') != "unknown":
        viewpoints[label['viewpoint']] = True

    img_id = get_image(conn, seal_id, img_path)
    
    ids, coords = get_patch_coordinates(conn, img_id)

    for (id, coord) in zip(ids, coords):
        # print(type(coord))
        # print(coord)
        if coord[0] > 1:
            update_patch_coordinates(conn, id, np.array(coord)/img.width)
        
    return []

def print_identity(x):
    print(x)
    return [x]

def create_encode_pipeline(dataset_name, 
                    extractor_name, 
                    extractor,
                    cfg):
    
    encode_pipeline = [
        compose_sequential(
        # print_identity,
        # print_step(f"Extracting features using {extractor_name}..."),                   
        # curry(encode_single, init_apply=extractor, cfg=cfg),    

        curry(update_coordinates, cfg["conn"])
        )
    ]
    return encode_pipeline
    

def create_pipeline(dataset_name, 
                    extractor_name, 
                    extractor,
                    cfg, 
                    codebooks_dataset=None):
    """
    Create full pipeline. Considers several special cases:
    - Whether the codebooks need to be generated or loaded
    - Whether leave-one-out strategy or query/database split is used
    """
    if codebooks_dataset is not None:
        # Load the codebooks of the specified dataset
        cfg["codebooks"] = load_pickle(f"./codebooks/codebooks_{codebooks_dataset}_{extractor_name}.pickle")
    
    pipeline = create_encode_pipeline(dataset_name, 
                    extractor_name, 
                    extractor,
                    cfg)
    
    return pipeline

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
    for (dataset_name, dataset, preprocessing, codebooks_dataset) in datasets:
        print()
        print(f"Dataset name: {dataset_name}")

        
        # print(f"Path to the dataset: {dataset.dataset_dir}")
        print(f"Dataset: {dataset}")
        print()

        cfg = config()
        cfg["codebooks_path"] = None
        ### Adjust config parameters here
        cfg["topk"]=topk


        ### TODO: Create psql connection here
        cfg["conn"] = create_connection()

        ### Create dataset
        # dataset = SimpleDataset(dataset_dir)


        ### List of feature extractors to test
        extractors = [
            ("HessAffNetHardNet", getHessAffNetHardNet(cfg)),
            # ("DISK", getDISK()),
            # ("KeyNetAffNetHardNet", getKeyNetAffNetHardNet())
        ]

        for (extractor_name, extractor) in extractors:
            print()
            print(f"Testing extractor {extractor_name}...")
            print()
            pipeline = create_pipeline(dataset_name, 
                    extractor_name, 
                    extractor,
                    cfg, 
                    codebooks_dataset=codebooks_dataset)
            
            print()
            apply_pipeline_dataset(dataset, pipeline, True)
            print()

def main():
    smart_resize_size = 255
    smart_resize_preprocess = [print_step(f"Resizing dataset (max side is {smart_resize_size})..."),
                                curry_sequential(resize_dataset, smart_resize_size)]
    
    leopard_preprocess = [print_step("Removing singletons..."), 
                          get_leopard_singletons, 
                          print_step("Cropping bounding boxes..."),
                          crop_label_step_sequential(), 
                          *smart_resize_preprocess]
    datasets = [  
                    ("norppa_database_segmented_pattern", 
                     GroupDataset("/ekaterina/work/data/norppa_database_segmented_pattern", "viewpoint"), 
                     None, 
                     "norppa_database_segmented_pattern")
                    #  ("norppa_database_pattern_unknown", 
                    #  SimpleDataset("/ekaterina/work/data/norppa_database_pattern_unknown"), 
                    #  None, 
                    #  "norppa_database_segmented_pattern")
                ]
    process_datasets(datasets)


if __name__ == "__main__":
    main()
