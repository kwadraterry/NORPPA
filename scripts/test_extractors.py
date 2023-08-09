import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import config

from config import config


from datasets import SimpleDataset, DatasetSlice

from tools import load_pickle, curry, apply_pipeline_dataset, curry_sequential, print_topk_accuracy, save_pickle, print_step, resize_dataset
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

def create_encode_pipeline(dataset_name, 
                    extractor_name, 
                    extractor,
                    cfg, 
                    compute_codebooks=False):
    if compute_codebooks:
        save_path = f"./codebooks/codebooks_{dataset_name}_{extractor_name}.pickle"
        codebooks_steps = [
            print_step("Updating config with new codebooks..."),
            curry(update_codebooks, cfg, save_path=save_path),
            print_step(f"The codebook is saved to {save_path}!"),
        ]
    else:
        codebooks_steps = []

    encode_pipeline = [
        print_step(f"Extracting features using {extractor_name}..."),                   
        curry(extract_patches, init_apply=extractor, config=cfg),    

        print_step("Encoding fisher vectors..."),
        curry(encode_patches, compute_codebooks=compute_codebooks, cfg=cfg),

        *codebooks_steps,
        # We can also save encoded images if we need to run further experiments later
        # curry(save_pickle, f"encoded_{dataset_name}_{extractor_name}.pickle"),
    ]
    return encode_pipeline
    

def create_pipeline(dataset_name, 
                    extractor_name, 
                    extractor,
                    cfg, 
                    codebooks_dataset=None, 
                    database_dataset=None):
    """
    Create full pipeline. Considers several special cases:
    - Whether the codebooks need to be generated or loaded
    - Whether leave-one-out strategy or query/database split is used
    """
    if codebooks_dataset is not None:
        # Load the codebooks of the specified dataset
        cfg["codebooks"] = load_pickle(f"./codebooks/codebooks_{codebooks_dataset}_{extractor_name}.pickle")
    
    encode_pipeline = create_encode_pipeline(dataset_name, 
                    extractor_name, 
                    extractor,
                    cfg, 
                    compute_codebooks=codebooks_dataset is None)

    if database_dataset is not None:
        print()
        print("Encoding database...")
        encoded_database = apply_pipeline_dataset(database_dataset, encode_pipeline, True)
        encode_pipeline = create_encode_pipeline(dataset_name, 
                    extractor_name, extractor,
                    cfg, compute_codebooks=False)
        print("Database encoded!")
        print()
    else:
        encoded_database = None
    
    pipeline = [
            *encode_pipeline,
            # We can also save encoded images if we need to run further experiments later
            # curry(save_pickle, f"encoded_{dataset_name}_{extractor_name}.pickle"),
        
            print_step("Starting identification..."),

            # We run testing with leave one out approach
            curry(identify, database=encoded_database, topk=cfg["topk"], leave_one_out=database_dataset is None),
            curry(print_topk_accuracy, label="Before geometric verification:"),

            print_step("Starting geometrical verification..."),
            curry_sequential(find_matches, cfg),
            curry_sequential(apply_geometric, cfg["geometric"]),
            curry(print_topk_accuracy, label="After geometric verification:")
                ]
    return pipeline

def process_datasets(datasets, smart_resize_size=256, topk=20):
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
    for (dataset_name, dataset, do_resize, codebooks_dataset) in datasets:
        print()
        print(f"Dataset name: {dataset_name}")

        leave_one_out = not isinstance(dataset, tuple)
        if leave_one_out:
            db_dataset = None
        else:
            (db_dataset, query_dataset) = dataset
            dataset = query_dataset
            print(f"Path to database the dataset: {db_dataset.dataset_dir}")
            print(f"Database dataset: {db_dataset}")
            print()

        print(f"Path to the dataset: {dataset.dataset_dir}")
        print(f"Dataset: {dataset}")
        print()

        cfg = config()
        cfg["codebooks_path"] = None
        ### Adjust config parameters here
        cfg["topk"]=topk

        ### Create dataset
        # dataset = SimpleDataset(dataset_dir)


        ### Resize dataset if necessary (i.e. when input images are too large)
        if do_resize:
            print(f"Resizing dataset (max side is {smart_resize_size})...")
            dataset = apply_pipeline_dataset(dataset, [curry_sequential(resize_dataset, smart_resize_size)])
            if not leave_one_out:
                db_dataset = apply_pipeline_dataset(db_dataset, [curry_sequential(resize_dataset, smart_resize_size)])

        ### List of feature extractors to test
        extractors = [
            ("HessAffNetHardNet", getHessAffNetHardNet(cfg)),
            ("DISK", getDISK()),
            ("KeyNetAffNetHardNet", getKeyNetAffNetHardNet())
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
                    database_dataset=db_dataset)
            
            print()
            apply_pipeline_dataset(dataset, pipeline, True)
            print()

def main():
    
    datasets = [  
                  ("norppa_pattern", 
                        (SimpleDataset("/ekaterina/work/data/dataset-0520/segmented_pattern_resized/database"), 
                         SimpleDataset("/ekaterina/work/data/dataset-0520/segmented_pattern_resized/query")), False, None),
                         
                  ("whaleshark_pattern_train", SimpleDataset("/ekaterina/work/data/whaleshark_norppa_pattern/train"), False, None),
                  ("whaleshark_base_train", SimpleDataset("/ekaterina/work/data/whaleshark_norppa/train"), True, None),
                  ("whaleshark_pattern_test", SimpleDataset("/ekaterina/work/data/whaleshark_norppa_pattern/test"), False, "whaleshark_pattern_train"),
                  ("whaleshark_base_test", SimpleDataset("/ekaterina/work/data/whaleshark_norppa/test"), True, "whaleshark_base_train"),
                ]
    process_datasets(datasets)


if __name__ == "__main__":
    main()
