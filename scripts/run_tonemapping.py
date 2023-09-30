import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import config

from config import config


from datasets import SimpleDataset, DatasetSlice

from tonemapping.tonemapping import tonemap_step
from norppa_tools import compose_sequential, load_pickle, curry, apply_pipeline_dataset, apply_sequential, curry_sequential, print_topk_accuracy, save_pickle, print_step, resize_dataset, get_save_step
from reidentification.identify import identify, apply_geometric, encode_patches, getDISK, getKeyNetAffNetHardNet, getHessAffNetHardNet, extract_patches
from reidentification.find_matches import find_matches


def process_datasets(datasets):
    for (dataset_name, dataset, out_dataset) in datasets:
        print()
        print(f"Dataset name: {dataset_name}")

        print(f"Path to the dataset: {dataset.dataset_dir}")

        print(f"Creating dataset {out_dataset}...")
        print()

        pipeline = [compose_sequential(tonemap_step,
                    get_save_step(out_dataset))]

        apply_pipeline_dataset(dataset, pipeline, True)
        print()

def main():
    
    datasets = [  ("norppa_database", 
                        SimpleDataset("/ekaterina/work/data/norppa_database"), 
                        "/ekaterina/work/data/norppa_database_tonemapped"),]
    process_datasets(datasets)


if __name__ == "__main__":
    main()
