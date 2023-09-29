import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import config

from config import config


from datasets import COCOImageDataset, SimpleDataset, DatasetSlice, COCOLeopardDataset, GroupDataset

from tools import get_leopard_singletons, crop_label_step_sequential, load_pickle, curry, apply_pipeline_dataset, curry_sequential, print_topk_accuracy, save_pickle, print_step, resize_dataset
from reidentification.identify import identify, apply_geometric, encode_patches, getDISK, getKeyNetAffNetHardNet, getHessAffNetHardNet, extract_patches
from reidentification.find_matches import find_matches

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

def upload_encoding_to_database(input, conn):
    fisher, labels = input
    label = labels['labels'][0]

    # fisher - fisher vector
    # label['class_id'] - class
    # label['viewpoint'] - viewpoint
    # label['file'] - img_path
    # label['features'] - n x 64 patch features (np.array)
    # label['ellipses'] - n x 5 patch ellipses ([np.array])
    # label['dataset_dir'] - dataset dir

    # Upload whatever using connection conn

    return []



def create_encode_pipeline(dataset_name, 
                    extractor_name, 
                    extractor,
                    cfg):
    
    encode_pipeline = [
        print_step(f"Extracting features using {extractor_name}..."),                   
        curry(extract_patches, init_apply=extractor, config=cfg),    

        print_step("Encoding fisher vectors..."),
        curry(encode_patches, cfg=cfg),

        # We can also save encoded images if we need to run further experiments later
        # curry(save_pickle, f"./output/identification_{dataset_name}_{extractor_name}.pickle"),
        curry_sequential(upload_encoding_to_database, cfg["conn"])
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
                    cfg, 
                    compute_codebooks=codebooks_dataset is None)
    
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

        
        print(f"Path to the dataset: {dataset.dataset_dir}")
        print(f"Dataset: {dataset}")
        print()

        cfg = config()
        cfg["codebooks_path"] = None
        ### Adjust config parameters here
        cfg["topk"]=topk


        ### TODO: Create psql connection here
        # cfg["conn"] = create_connection()

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
                    codebooks_dataset=codebooks_dataset, 
                    database_dataset=None)
            
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
                     GroupDataset("/ekaterina/work/data/norppa_database", "viewpoint"), 
                     None, 
                     "norppa_database_segmented_pattern")
                ]
    process_datasets(datasets)


if __name__ == "__main__":
    main()
