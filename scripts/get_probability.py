
from config import config
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import zipfile
import wget
import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from torchvision.datasets.utils import download_url
from datasets import COCOImageDataset, DatasetSlice, SimpleDataset

from norppa_tools import load_pickle, save_pickle, curry, curry_sequential, apply_sequential, apply_pipeline_dataset, get_save_step, apply_sequential, compose_sequential, resize_dataset

from reidentification.identify import encode_patches, extract_patches

from reidentification.encoding_utils import calculate_dists


params = {
    "positive": {
        "shape": 0.21645254194860702,
        "loc": -1.0129360876962725,
        "scale": 0.15684373474506558,
    },
    "negative": {
        "shape": 21.939458901190136,
        "loc": -1.0330643527569894,
        "scale": 0.001826850341859507,
    },
    "prior": 0.005,
    "smart_resize_size": 255
}

cfg = config()
cfg["codebooks"] = load_pickle(f"./codebooks/codebooks_{codebooks_dataset}_{extractor_name}.pickle")
cfg["params"] = params


def get_likelihood(x, set="positive"):
    shape = params[set]["shape"]
    loc = params[set]["loc"]
    scale = params[set]["scale"]
    return gamma.pdf(-x, shape, loc=loc, scale=scale)



def probability_true(x, prior=0.5):
    true = get_likelihood(x, set="positive")
    false = get_likelihood(x, set="negative")
    prior = params["prior"]
    return (prior * true) / (prior * true + (1-prior) * false)




def get_probability(image1, image2):
    dataset = [(image1, {"file": "1"}), (image2, {"file": "2"})] 

    pipeline = [
        curry_sequential(resize_dataset, params["smart_resize_size"]),

        curry(extract_patches, init_apply=extractor, config=cfg), 
        curry(encode_patches, compute_codebooks=False, cfg=cfg),
    ]
    
    encoded = apply_pipeline_dataset(dataset, pipeline, verbose=True)

    if len(encoded) < 2:
        return None
    
    image1_features = encoded[0][0][np.newaxis,...]
    image2_features = encoded[1][0][np.newaxis,...]

    dist = calculate_dists(image1_features, image2_features)


    return probability_true(dist)
