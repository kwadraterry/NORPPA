import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
# export CUDA_DEVICE_ORDER=PCI_BUS_ID & export CUDA_VISIBLE_DEVICES=2

# import config

from config import config


from datasets import SimpleDataset, DatasetSlice, GroupDataset

from segmentation.segmentation import segment
from tonemapping.tonemapping import tonemap_step
from tools import compose_sequential, get_smart_shrink_step, load_pickle, curry, apply_pipeline_dataset, apply_sequential, curry_sequential, print_topk_accuracy, save_pickle, print_step, resize_dataset, get_save_step
from reidentification.identify import identify, apply_geometric, encode_patches, getDISK, getKeyNetAffNetHardNet, getHessAffNetHardNet, extract_patches
from reidentification.find_matches import find_matches
from pattern_extraction.extract_pattern import extract_pattern


import torch
torch.cuda.empty_cache()

def debug_step(input):
    # denum = 1024 ** 2
    # t = torch.cuda.get_device_properties(0).total_memory / denum
    # r = torch.cuda.memory_reserved(0) / denum
    # a = torch.cuda.memory_allocated(0) / denum
    # f = r-a  # free inside reserved
    # print(f"Total: {t}")
    # print(f"Reserved: {r}")
    # print(f"Allocated: {a}")
    # print(f"Available: {f}")
    # print([x/denum for x in torch.cuda.mem_get_info()])
    # print()
    torch.cuda.empty_cache()
    return [input]

def clear_step(input):
    return []

def process_datasets(datasets):
    for (dataset_name, dataset, segment_dataset, pattern_dataset) in datasets:
        print()
        print(f"Dataset name: {dataset_name}")

        print(f"Path to the dataset: {dataset.dataset_dir}")

        print(f"Creating datasets {segment_dataset} and {pattern_dataset}...")

        print()

        cfg = config()

        os.makedirs(segment_dataset, exist_ok=True)
        os.makedirs(pattern_dataset, exist_ok=True)
        pipeline = [compose_sequential(
                    # debug_step,
                    # get_smart_shrink_step(128),
                    curry(segment, cfg["seem"], instance_segmentation=False),
                    get_save_step(segment_dataset),
                    curry(extract_pattern, model=cfg["unet"]),
                    get_save_step(pattern_dataset),
                    clear_step)
                    ]

        apply_pipeline_dataset(dataset, pipeline, True)
        print()

def main():
    
    datasets = [  ("norppa_database", 
                        GroupDataset("/ekaterina/work/data/norppa_database", "viewpoint"), 
                        "/ekaterina/work/data/norppa_database_segmented", 
                        "/ekaterina/work/data/norppa_database_segmented_pattern"),]
    process_datasets(datasets)


if __name__ == "__main__":
    main()
