import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import config

from config import config


from datasets import COCOImageDataset, SimpleDataset, DatasetSlice, COCOLeopardDataset

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
        # curry(save_pickle, f"./output/identification_{dataset_name}_{extractor_name}.pickle"),
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

            # curry(save_pickle, f"./output/identification_{dataset_name}_{extractor_name}.pickle"),

            curry(print_topk_accuracy, label="After geometric verification:")

                ]
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
        if preprocessing is not None:
            print(f"Preprocessing dataset...")
            if not leave_one_out:
                db_dataset = apply_pipeline_dataset(db_dataset, preprocessing)
            dataset = apply_pipeline_dataset(dataset, preprocessing)

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
    smart_resize_size = 255
    smart_resize_preprocess = [print_step(f"Resizing dataset (max side is {smart_resize_size})..."),
                                curry_sequential(resize_dataset, smart_resize_size)]
    
    leopard_preprocess = [print_step("Removing singletons..."), 
                          get_leopard_singletons, 
                          print_step("Cropping bounding boxes..."),
                          crop_label_step_sequential(), 
                          *smart_resize_preprocess]
    datasets = [  
                    # ("norppa_segmented", 
                    #      (SimpleDataset("/ekaterina/work/data/dataset-0520/segmented/database"), 
                    #       SimpleDataset("/ekaterina/work/data/dataset-0520/segmented/query")), smart_resize_preprocess, None),
                #   ("whaleshark_train", SimpleDataset("/ekaterina/work/data/whaleshark_norppa/train"), smart_resize_preprocess, "whaleshark_train_05"),
                #   ("whaleshark_test", SimpleDataset("/ekaterina/work/data/whaleshark_norppa/test"), smart_resize_preprocess, "whaleshark"),
                    # ("leopard", COCOLeopardDataset("/ekaterina/work/data/coco_leopard/images/test2023",
                    #                                 "/ekaterina/work/data/coco_leopard/annotations/instances_test2023.json"), leopard_preprocess, None)
                    ("whaleshark_pie_025_train", COCOImageDataset("/ekaterina/work/data/whaleshark_coco/", 
                                                            "/ekaterina/work/src/wildlife-embeddings_PIE2/data/coco_whaleshark_mixed_025.csv", 
                                                            "train", header=("image","name","set")), smart_resize_preprocess, None),
                    ("whaleshark_pie_025_test", COCOImageDataset("/ekaterina/work/data/whaleshark_coco/", 
                                                           "/ekaterina/work/src/wildlife-embeddings_PIE2/data/coco_whaleshark_mixed_025.csv", 
                                                           "test", header=("image","name","set")), smart_resize_preprocess, "whaleshark_pie_025_train"),
                    ("whaleshark_pie_05_train", COCOImageDataset("/ekaterina/work/data/whaleshark_coco/", 
                                                            "/ekaterina/work/src/wildlife-embeddings_PIE2/data/coco_whaleshark_mixed_05.csv", 
                                                            "train", header=("image","name","set")), smart_resize_preprocess, None),
                    ("whaleshark_pie_05_test", COCOImageDataset("/ekaterina/work/data/whaleshark_coco/", 
                                                           "/ekaterina/work/src/wildlife-embeddings_PIE2/data/coco_whaleshark_mixed_05.csv", 
                                                           "test", header=("image","name","set")), smart_resize_preprocess, "whaleshark_pie_05_train"),
                    ("whaleshark_pie_1_train", COCOImageDataset("/ekaterina/work/data/whaleshark_coco/", 
                                                            "/ekaterina/work/src/wildlife-embeddings_PIE2/data/coco_whaleshark_mixed_1.csv", 
                                                            "train", header=("image","name","set")), smart_resize_preprocess, None),
                    ("whaleshark_pie_1_test", COCOImageDataset("/ekaterina/work/data/whaleshark_coco/", 
                                                           "/ekaterina/work/src/wildlife-embeddings_PIE2/data/coco_whaleshark_mixed_1.csv", 
                                                           "test", header=("image","name","set")), smart_resize_preprocess, "whaleshark_pie_1_train"),
                    # ("sealid_pie_025_train", COCOImageDataset("/ekaterina/work/data/coco_sealid_pattern_resized/", 
                    #                                         "/ekaterina/work/src/wildlife-embeddings_PIE2/data/coco_sealid_025.csv", 
                    #                                         "train", header=("image","name","set")), None, None),
                    # ("sealid_pie_025_test", COCOImageDataset("/ekaterina/work/data/coco_sealid_pattern_resized/", 
                    #                                        "/ekaterina/work/src/wildlife-embeddings_PIE2/data/coco_sealid_025.csv", 
                    #                                        "test", header=("image","name","set")), None, "sealid_pie_025_train"),
                    
                    # ("sealid_pie_05_train", COCOImageDataset("/ekaterina/work/data/coco_sealid_pattern_resized/", 
                    #                                         "/ekaterina/work/src/wildlife-embeddings_PIE2/data/coco_sealid_05.csv", 
                    #                                         "train", header=("image","name","set")), None, None),
                    # ("sealid_pie_05_test", COCOImageDataset("/ekaterina/work/data/coco_sealid_pattern_resized/", 
                    #                                        "/ekaterina/work/src/wildlife-embeddings_PIE2/data/coco_sealid_05.csv", 
                    #                                        "test", header=("image","name","set")), None, "sealid_pie_05_train"),
                    
                    # ("sealid_pie_1_train", COCOImageDataset("/ekaterina/work/data/coco_sealid_pattern_resized/", 
                    #                                         "/ekaterina/work/src/wildlife-embeddings_PIE2/data/coco_sealid_1.csv", 
                    #                                         "train", header=("image","name","set")), None, None),
                    # ("sealid_pie_1_test", COCOImageDataset("/ekaterina/work/data/coco_sealid_pattern_resized/", 
                    #                                        "/ekaterina/work/src/wildlife-embeddings_PIE2/data/coco_sealid_1.csv", 
                    #                                        "test", header=("image","name","set")), None, "sealid_pie_1_train"),

                    # ("norppa_segmented", 
                    #      (SimpleDataset("/ekaterina/work/data/dataset-0520/segmented/database"), 
                    #       SimpleDataset("/ekaterina/work/data/dataset-0520/segmented/query")), smart_resize_preprocess, None),
                # ("norppa_pattern_train_025", 
                        # SimpleDataset("/ekaterina/work/data/dataset-0520/segmented_pattern_resized/database", per_class_limit=0.25), None, None),
                #    ("norppa_pattern", 
                #          (SimpleDataset("/ekaterina/work/data/dataset-0520/segmented_pattern_resized/database"), 
                #           SimpleDataset("/ekaterina/work/data/dataset-0520/segmented_pattern_resized/query")), None, None),

                #   ("whaleshark_train_025", SimpleDataset("/ekaterina/work/data/whaleshark_norppa/train", per_class_limit=0.25), smart_resize_preprocess, None),
                #   ("whaleshark_train", SimpleDataset("/ekaterina/work/data/whaleshark_norppa/train"), smart_resize_preprocess, "whaleshark_train_025"),
                #   ("whaleshark_test", SimpleDataset("/ekaterina/work/data/whaleshark_norppa/test"), smart_resize_preprocess, "whaleshark_train_025"),

                #   ("whaleshark_train_05", SimpleDataset("/ekaterina/work/data/whaleshark_norppa/train", per_class_limit=0.5), smart_resize_preprocess, None),
                #   ("whaleshark_train", SimpleDataset("/ekaterina/work/data/whaleshark_norppa/train"), smart_resize_preprocess, "whaleshark_train_05"),
                #   ("whaleshark_test", SimpleDataset("/ekaterina/work/data/whaleshark_norppa/test"), smart_resize_preprocess, "whaleshark_train_05")
                ]
    process_datasets(datasets)


if __name__ == "__main__":
    main()
