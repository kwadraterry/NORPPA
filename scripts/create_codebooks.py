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
from reidentification.identify import identify, apply_geometric, encode_patches, getDISK, getKeyNetAffNetHardNet, getHessAffNetHardNet, extract_patches

import torch
torch.cuda.empty_cache()

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
    for (dataset_name, dataset) in datasets:
        print()
        print(f"Dataset name: {dataset_name}")

        print(f"Path to the dataset: {dataset.dataset_dir}")

        print()

        cfg = config()

        extractor_name, extractor = ("HessAffNetHardNet", getHessAffNetHardNet(cfg))

        save_path = f"./codebooks/codebooks_{dataset_name}_{extractor_name}.pickle"

        pipeline = [print_step(f"Extracting features using {extractor_name}..."),                   
                    curry(extract_patches, init_apply=extractor, config=cfg),    

                    print_step("Encoding fisher vectors..."),
                    curry(encode_patches, compute_codebooks=True, cfg=cfg),

                    curry(update_codebooks, cfg, save_path=save_path),
                    print_step(f"The codebook is saved to {save_path}!"),
                    apply_sequential(clear_step)
                    ]

        apply_pipeline_dataset(dataset, pipeline, True)
        print()

def main():
    
    datasets = [  ("norppa_database_segmented_pattern", 
                        GroupDataset("/ekaterina/work/data/norppa_database_segmented_pattern", "viewpoint")
                    ),]
    process_datasets(datasets)


if __name__ == "__main__":
    main()
