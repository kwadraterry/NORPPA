"""
The script peforms tone mapping.

Usage:
python correct.py -s <source_dir_path> -d <dest_dir_path>

There is no need to create the directory for the results manually since it will
be generated automatically preseving the structure of the source directory.

"""

from argparse import ArgumentParser
import json
import os
from norppa_tools import is_image
import subprocess

from pathlib import Path
import numpy as np

import skimage.color
import matplotlib.pyplot as plt
from PIL import Image
from subprocess import *

def decode(input):
    sep = b"ENDH"
    endh = input.find(sep)
    header = input[:endh].decode()
    lines = header.split('\n')
    size = lines[1].split(' ')
    channels = int(lines[2])
    width = int(size[0])
    height = int(size[1])
    
    image = input[endh+len(sep):]
    vals = np.frombuffer(image, dtype=np.float32)
    image = np.reshape(vals, (channels, height, width)).transpose((1, 2, 0))
    image = skimage.color.xyz2rgb(image)
    image = Image.fromarray(np.uint8(image*255))
    return image

def encode(image, label):
    header_start = "PFS1"
    width, height = image.size
    channels = len(image.split())
    luminance = "RELATIVE"
    WHITE_Y = 1
    tags = 4
    BITDEPTH=8
    X = 0
    Y = 0
    Z = 0
    header_end = "ENDH"

    header_string = f'{header_start}\n{width} {height}\n{channels}\n{tags}\nLUMINANCE={luminance}\nWHITE_Y={WHITE_Y}\nFILE_NAME={label}\nBITDEPTH={BITDEPTH}\nX\n{X}\nY\n{Y}\nZ\n{Z}\n{header_end}'

    image = skimage.color.rgb2xyz(image)
    image_array = np.array(image, dtype=np.float32).transpose((2, 0, 1))
    vals = header_string.encode() + np.ndarray.tobytes(image_array)
    return  vals

def call_mantiuk(img_bytes):
    p = Popen(['pfstmo_mantiuk06', '--quiet'],   stdin=PIPE,      stdout=PIPE)
    stdout, _ = p.communicate(input=img_bytes) 
    return stdout


def tonemap(image, label=""):
    image = encode(image, label)
    image = call_mantiuk(image)
    return decode(image)

def get_path(label):
    if type(label) is str:
        return label
    elif type(label) is dict:
        return label['file']
    else:
        return ""

def tonemap_step(input):
    image, img_label = input
    if image is None:
        return []
    image = encode(image, get_path(img_label))
    image = call_mantiuk(image)
    result_image = decode(image)

    return [(result_image, img_label)]
