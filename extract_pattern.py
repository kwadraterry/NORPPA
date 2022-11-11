import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import config
from argparse import ArgumentParser
from pathlib import Path

from config import config
from pathlib import Path
import numpy as np
import tarfile, zipfile
import tensorflow as tf


from torchvision.datasets.utils import download_url
from datasets import COCOImageDataset, DatasetSlice, SimpleDataset

from tools import apply_pipeline, crop_step, curry, apply_pipeline_cocodataset, get_save_step
from tonemapping.tonemapping import tonemap, tonemap_step
from segmentation.segmentation import segment
from pattern_extraction.extract_pattern import extract_pattern
from reidentification.identify import encode_single, encode_pipeline, create_database, identify, identify_single
from reidentification.visualisation import visualise_match
from simple_database import SimpleDatabase
import pickle


def extract_pattern_dataset(input_dir, output_dir):
    cfg = config()
    extract_pattern_step = curry(extract_pattern, model=cfg["unet"])
    
    dataset = SimpleDataset(input_dir)
    pipeline = [
            crop_step,
            extract_pattern_step,
            get_save_step(output_dir)
            ]
    pattern_dataset = apply_pipeline_cocodataset(dataset, pipeline, True)
    

def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input",
                        dest="input",
                        required=False,
                        default="/ekaterina/work/data/unknown_dataset",
                        help="Path to the segmented images")
    parser.add_argument("-o", "--output",
                        dest="output",
                        required=False,
                        default="/ekaterina/work/data/unknown_dataset_pattern",
                        help="Path to the pattern images")
    args = parser.parse_args()
    
    extract_pattern_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
