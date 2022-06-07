
from datetime import datetime
import random
import string
import os
import sys
import shutil
from zipfile import ZipFile
from skimage import color
import numpy as np
from datetime import datetime
from six.moves import urllib

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from PIL import Image
import json
import gc


def flatten(t):
    return [item for sublist in t for item in sublist]

def apply_step(input, step, rest):
    result = step(input)
    if rest:
        result = flatten((apply_step(x, rest[0], rest[1:]) for x in result))
    return result

def apply_pipeline(image, label, pipeline):
    return apply_step((image, label), pipeline[0], pipeline[1:])


def apply_pipeline_cocodataset(cocodataset, pipeline, verbose=False):
    result = []
    for (i, (img, data)) in enumerate(cocodataset):
        result.extend(apply_pipeline(img, data, pipeline))
        if verbose:
            print(f"Completed {i+1}/{len(cocodataset)} images")
    return result

def process(SOURCE_DIR, pipeline):
    result = []
    num_files = sum([len(files) for r, d, files in os.walk(SOURCE_DIR)])
    for root, _, files in os.walk(SOURCE_DIR):
        for f in files:
            num_files -= 1
            result.extend(apply_pipeline(os.path.join(root, f), SOURCE_DIR, pipeline))
            gc.collect(0)
    return result

def curry(func, *params, **kw):
    return lambda x: func(x, *params, **kw)


def save_id_result(result, source_dir, dest_path):
    res_json_path = os.path.join(dest_path, 'result.json')
    data = []
    
    for _, res in result:
        res["query"] = os.path.join(source_dir, res["query"])
        data.append(res)
        with open(res_json_path, mode='w') as outfile:
            json.dump(data, outfile)

def save_upload_result(result, dest_path):
    res_json_path = os.path.join(dest_path, 'result.json')
    data = []
    
    for image, label in result:
        if image is None:
            res = "Fail"
        else:
            res = "Success"
        data.append((label,res))
        with open(res_json_path, mode='w') as outfile:
            json.dump(data, outfile)



def randomString(stringLength=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def generate_task_id(task_type):
    now = datetime.now()

    date_time = now.strftime("%d%m%y_%H%M%S")

    return f"{task_type[:4]}_{date_time}"


def clean_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def to_zip_file(file, zip_file):
    if os.path.isfile(file):
        zip_file.write(file)
    else:
        addFolderToZip(zip_file, file, file)

def toZip(file, filename):
    zip_file = ZipFile(filename, 'w')
    to_zip_file(file, zip_file)
    zip_file.close()

def addFolderToZip(zip_file, folder, base_path): 
    for file in os.listdir(folder):
        full_path = os.path.join(folder, file)
        if os.path.isfile(full_path):
            # print('File added: ' + str(full_path))
            zip_file.write(full_path, os.path.relpath(full_path, start=base_path))
            # print(full_path, base_path)
        elif os.path.isdir(full_path):
            # print('Entering folder: ' + str(full_path))
            addFolderToZip(zip_file, full_path, base_path)

def is_image(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', 'cr2', 'pef', 'nef'))

def is_raw_image(filename):
    return filename.lower().endswith(('cr2', 'pef', 'nef'))

def crop_to_bb(img, return_bb=False):
    labels = color.rgb2gray(np.asarray(img))
    labels = (labels > 0).astype(np.uint8)
    where = np.where(labels)
    y1, x1 = np.amin(where, axis=1)
    y2, x2 = np.amax(where, axis=1)
    img_cropped = img.crop((x1, y1, x2, y2))
    if return_bb:
        return (img_cropped, (x1,y1,x2-x1,y2-y1))
    else:
        return img_cropped



def crop_imgs_in_dir(src):
    for dirpath, _, filenames in os.walk(src): 
        for filename in filenames:
            source = os.path.join(dirpath, filename)
            img = Image.open(source)
            img = img.convert('RGB')
            img_cropped = crop_to_bb(img)
            img_cropped.save(source)


def crop_step(input):
    image, label = input 
    if image is not None:
        image, bb = crop_to_bb(image, True)
        if type(label) is dict:
            label["bb"] = bb
    return [(image, label)]

def change_dir(path, new_dir):
    name = os.path.basename(path)
    return os.path.join(new_dir, name)

def get_save_step(dest_dir):
    return lambda x: save_step(x, dest_dir)

def save_step(input, dest_dir, new_path_name=None):
    image, label = input
    if image is not None:
        os.makedirs(dest_dir, exist_ok=True)
        if type(label) is dict and 'dataset_dir' in label:
            new_path = os.path.join(dest_dir, os.path.relpath(label['file'], label['dataset_dir']))
        else:
            new_path = change_dir(label['file'], dest_dir)
        os.makedirs(os.path.dirname(new_path), exist_ok = True) 
        image.save(new_path)
        if type(label) is dict and new_path_name is not None:
            label[new_path_name] = new_path
    return [(image, label)]

def download_url(url, dst):

    print('* url="{}"'.format(url))
    print('* destination="{}"'.format(dst))

    def _reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(
            '\r...%d%%, %d MB, %d KB/s, %d seconds passed'
            % (percent, progress_size / (1024 * 1024), speed, duration)
        )
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dst, _reporthook)
    sys.stdout.write('\n')


def read_image(path):
    if not os.path.exists(path):
        raise IOError(f'"{path}" does not exist')
    try:
        img = Image.open(path).convert('RGB')
    except IOError:
        print(f'IOError when reading "{path}"')
    return img