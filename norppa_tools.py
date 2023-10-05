
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
from timeit import default_timer as timer
from pattern_extraction.utils import thickness_resize
from pattern_extraction.extract_pattern import smart_resize

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from PIL import Image
import json
import gc

from tqdm import tqdm

import pickle
import copy

def get_leopard_singletons(db, threshold=1):
    classes_nums = {class_id:0 for class_id in db.classes}
    for ci in [items[1]["class_id"] for items in db.data]:
        classes_nums[ci] += 1
    filt = [i for (i, (_, label)) in enumerate(db.data) if classes_nums[label["class_id"]] > threshold]
    db_new = copy.copy(db)
    db_new.data = [db_new.data[i] for i in filt]
    db_new.classes = list(db_new._get_classes(db_new.data))
    return db_new

def get_smart_shrink_step(max_size):
    return curry(smart_shrink_step, max_size)

def smart_shrink_step(input, max_size):
    img, label = input
    print(label)
    return [(smart_shrink(img, max_size), label)]

def smart_shrink(img, max_size, return_ratio=False):
    ratio = min(1, max_size / max(img.size))
    if ratio < 1:
        result = img.resize(tuple(int(i * ratio) for i in img.size), Image.LANCZOS).convert("RGB")
    else:
        result = img.convert("RGB")
    print(result.size)
    if return_ratio:
        return (result, ratio)
    else:
        return result

def save_pickle(x, file):
    with open(file, 'wb') as f_file:
        pickle.dump(x, f_file, protocol=4)
    return x
        
def load_pickle(file):
    with open(file, 'rb') as f_file:
        result = pickle.load(f_file)
    return result

def load_json(file):
    with open(file, 'rb') as f_file:
        result = json.load(f_file)
    return result

def print_step(text):
    def identity_print(x):
        print(text)
        return x
    return identity_print

def flatten(t):
    return [item for sublist in t for item in sublist]

def apply_step(input, step, rest):
    result = step(input)
    if rest:
        result = flatten((apply_step(x, rest[0], rest[1:]) for x in result))
    return result

def apply_pipeline(image, pipeline, verbose=False):
    return apply_pipeline_dataset([image], pipeline, verbose=verbose)

def print_step_progress(i, n):
    print(f"Completed {i}/{n} steps")

def apply_pipeline_dataset(dataset, pipeline, verbose=False, verbose_action=print_step_progress):
    result = dataset
    for (i,step) in enumerate(pipeline):
        result = step(result)
        if verbose:
            verbose_action(i+1, len(pipeline))
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

def cat(array):
    return [y for x in array for y in x]

def apply_sequential(func):
    return lambda dataset: cat([func(x) for x in tqdm(dataset)])

def curry_sequential(func, *params, **kw):
    return apply_sequential(curry(func, *params, **kw))

def compose(*funcs):
    return curry(apply_step, step=funcs[0], rest=funcs[1:])

def compose_sequential(*funcs):
    return apply_sequential(compose(*funcs))

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

def bbox_to_coords(bbox):
    (l, t, w, h) = bbox
    return (l, t, l+w, t+h)

def crop_label_step(input, bb_field="bbox"):
    image, label = input
    if image is not None:
        image = image.crop(bbox_to_coords(label[bb_field]))
    return [(image, label)]

def crop_label_step_sequential(bb_field="bbox"):
    return curry_sequential(crop_label_step, bb_field=bb_field)

def crop_step(input):
    image, label = input 
    if image is not None:
        image, bb = crop_to_bb(image, True)
        if type(label) is dict:
            label["bb"] = bb
    return [(image, label)]

def crop_step_sequential(input):
    return apply_sequential(crop_step)(input)

def thickness_resize_step(input):
    image, label = input 
    if image is not None:
        image, ratio = thickness_resize(image, return_ratio=True)
        if type(label) is dict:
            label["resize_ratio"] = ratio
    return [(image, label)]

def change_dir(path, new_dir):
    name = os.path.basename(path)
    return os.path.join(new_dir, name)

# def get_save_step(dest_dir):
#     return lambda x: save_step(x, dest_dir)

def get_save_step_sequential(dest_dir, new_path_name=None, verbose=False):
    return apply_sequential(get_save_step(dest_dir, new_path_name=new_path_name, verbose=verbose))

def get_save_step(dest_dir, new_path_name=None, verbose=False):
    return curry(save_step, dest_dir, new_path_name=new_path_name, verbose=verbose)

def test_save_step(dest_dir):
    return lambda x: test_step(x, dest_dir)

def test_step(input, dest_dir):
    image, label = input
    if image is not None:
        if type(label) is dict and 'dataset_dir' in label:
            new_path = os.path.join(dest_dir, os.path.relpath(label['file'], label['dataset_dir']))
        else:
            new_path = change_dir(label['file'], dest_dir)
        if os.path.exists(new_path):
            return []
        
    return [(image, label)]

def save_step(input, dest_dir, new_path_name=None, verbose=False, clear=False):
    image, label = input
    if image is not None:
        os.makedirs(dest_dir, exist_ok=True)
        if type(label) is dict and 'dataset_dir' in label:
            new_path = os.path.join(dest_dir, os.path.relpath(label['file'], label['dataset_dir']))
        else:
            new_path = change_dir(label['file'], dest_dir)
        os.makedirs(os.path.dirname(new_path), exist_ok = True) 
        image.save(new_path, format="png")
        if verbose:
            print(f"Saved image to path {new_path}")
        if type(label) is dict and new_path_name is not None:
            label[new_path_name] = new_path

    if clear:
        return []
    else:
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

def get_topk_matrix(identification_result):
    result = []
    for (db_labels, q_labels) in (identification_result):
        q_class = q_labels['class_id']
        
        q_ln = len(q_labels['labels'])
        
        
        result.extend([[db_label['db_label']['class_id']==q_class for db_label in db_labels]]*q_ln)
    topk = len(identification_result[0][0])
    return (np.asarray(result), topk)

def calculate_accuracy(result, max_topk=None):

    topk_matrix, topk = get_topk_matrix(result)

    topk = max_topk if max_topk is not None else topk
    print(f'TOP-k={topk}')
    hits = topk_matrix 
    # hits = (db_labels.T == q_labels).T
    print([sum((np.sum(hits[:, :j+1], axis=1) > 0)) / len(topk_matrix)
            for j in range(topk)])
    
def get_topk_matrix(identification_result):
    result = []
    for (db_labels, q_labels) in identification_result:
        q_class = q_labels['class_id']
        q_ln = len(q_labels['labels'])
        result.append([db_label['db_label']['class_id']==q_class for db_label in db_labels]*q_ln)
    return np.asarray(result)


def get_topk_accuracy(identification_result):
    result = []
    for (db_labels, q_labels) in identification_result:
        q_class = q_labels['class_id']
        q_ln = len(q_labels['labels'])
        hits = [db_label['db_label']['class_id']==q_class for db_label in db_labels]
        for _ in range(q_ln):
            result.append(hits)
    result = np.asarray(result)
    return [sum((np.sum(result[:, :j+1], axis=1) > 0)) / len(result) for j in range(result.shape[1])]

def print_topk_accuracy(identification_result, label=""):
    topk_acc = get_topk_accuracy(identification_result)
    print(label)
    for (i, acc) in enumerate(topk_acc):
        print(f"Top-{i+1} accuracy: {acc*100}%")
    return identification_result


def resize_dataset(input, size):
    image, img_label = input
    if image is None:
        return [input]

    result, ratio = smart_resize(image, size, return_ratio=True)
    img_label["resize_ratio"] = ratio
    return [(result, img_label)]


def update_codebooks(input, cfg):
    codebooks, encoded = input
    cfg["codebooks"] = codebooks
    return encoded



class StopwatchPrint(object):
    def __init__(self,
                 start_print="Start stopwatch",
                 final_print="Elapsed time %2.4f"):
        self.start_print = start_print
        self.final_print = final_print

    def __enter__(self):
        self.tic = timer()
        print(self.start_print)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.toc = timer()
        print(self.final_print % (self.toc-self.tic))


def overwrite_step(input):
    image, label = input
    if image is not None:
        image.save(label['file'])
    return [input]