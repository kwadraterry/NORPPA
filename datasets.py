import os
from pathlib import Path
from norppa_tools import read_image
import csv
import numpy as np
import math

from torch.utils.data import Dataset
import os
import json

class DatasetSlice(Dataset):
    def __init__(self, dataset, slice=None):
        self.dataset = dataset
        self.slice = (0, len(self.dataset)) if slice is None else slice
        if type(self.slice) is tuple:
            self.slice = range(*self.slice)
        if hasattr(dataset,'imgs'):
            self.imgs = [dataset.imgs[i] for i in self.slice]
        if hasattr(dataset,'classes'):
            self.classes = dataset.classes
    def __getitem__(self, index):
        return self.dataset[self.slice[index]]
    def __len__(self):
        return len(self.slice)


class COCOLeopardDataset(Dataset):

    def __init__(self, 
                dataset_dir,
                annotation):
        
        self.dataset_dir = dataset_dir
        self.annotation = annotation

        self.data = self._get_data(self.annotation)
        self.classes = list(self._get_classes(self.data))


    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = read_image(img_path)
        return img, label

    def __len__(self):
        return len(self.data)

    def _get_classes(self, data):
        classes = set([items[1]["class_id"] for items in data])
        return classes

    def _get_data(self, annotation):
        """ Get database from COCO annotations """
        
        result = []
        class_ids = []
        with open(annotation, mode='r') as json_file:
            json_dict = json.load(json_file)
        for (img, annot) in zip(json_dict["images"], json_dict["annotations"]):
            image_path = self._get_image_path(img["file_name"])
            class_id = len(class_ids)
            for i in range(len(class_ids)):
                if annot["id"] in class_ids[i]:
                    class_id = i
                    break
            if class_id == len(class_ids):
                class_ids.append(annot["individual_ids"])
            label = annot.copy()
            label["class_id"] = f"{class_id}_{annot['viewpoint']}"
            label["file"] = image_path
            result.append((image_path,label))
        return result

    def _get_image_path(self, filename):
        return os.path.join(self.dataset_dir, filename)


class COCOImageDataset(Dataset):

    def __init__(self, 
                dataset_dir,
                annotation,
                split=None,
                header= ("file","class_id","reid_split")):
                # ("image","name","set")):
        
        self.split = split
        self.dataset_dir = dataset_dir
        self.annotation = annotation
        self.header = header

        self.data = self._get_data(split, self.annotation)
        self.classes = list(self._get_classes(self.data))


    def __getitem__(self, index):
        img_path, pid = self.data[index]
        img = read_image(img_path) if os.path.isfile(img_path) else None
        return img, {'class_id': pid, 'file': img_path}

    def __len__(self):
        return len(self.data)

    def _get_classes(self, data):
        classes = set([items[1] for items in data])
        return classes

    def _get_data(self, split, annotation):
        """ Get database from COCO annotations """
        
        result = []
        
        with open(annotation, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if split is None or (row[self.header[2]] == split):
                    image_path = self._get_image_path(row[self.header[0]])
                    image_pid = row[self.header[1]]
                    result.append((image_path,image_pid))
        return result

    def _get_image_path(self, filename):
        return os.path.join(self.dataset_dir, filename)

    

class SimpleDataset(Dataset):
    def __init__(self, 
                dataset_dir, per_class_limit=None):
        
        self.per_class_limit = per_class_limit
        self.dataset_dir = dataset_dir
        self.data = self._get_data(dataset_dir)
        self.classes = list(self._get_classes(self.data))

    def __getitem__(self, index):
        img_path, pid = self.data[index]
        img = read_image(img_path)
        return img, {'class_id': pid, 'file': img_path, 'dataset_dir':self.dataset_dir}

    def __len__(self):
        return len(self.data)

    def _get_classes(self, data):
        classes = set([items[1] for items in data])
        return classes

    def _check_limit(self, class_iter):
        if self.per_class_limit is None:
            return len(class_iter)
        elif self.per_class_limit < 1:
            return round(len(class_iter) * self.per_class_limit)
        else:
            return self.per_class_limit
        

    def _get_data(self, dataset_dir):
        dataset_dir = Path(dataset_dir)
        result = []
        for class_dir in [x for x in dataset_dir.iterdir() if x.is_dir()]:
            imgs = list(class_dir.iterdir())
            lim = self._check_limit(imgs)
            for img in imgs[:lim]:
                result.append((str(img), class_dir.name))
        return result
    
    def get_labels(self):
        labels = [items[1] for items in self.data]
        return labels


class QueryDataset(Dataset):
    def __init__(self, 
                dataset_dir, class_name="unknown"):
        
        self.dataset_dir = dataset_dir
        self.data = self._get_data(dataset_dir)
        self.class_name = class_name
        self.classes = [self.class_name]

    def __getitem__(self, index):
        img_path = self.data[index]
        img = read_image(img_path)
        return img, {'class_id': self.class_name, 'file': img_path, 'dataset_dir':self.dataset_dir}

    def __len__(self):
        return len(self.data)

    def _get_data(self, dataset_dir):
        dataset_dir = Path(dataset_dir)
        result = []
        for img in dataset_dir.iterdir():
            result.append(str(img))
        return result
    
    def get_labels(self):
        labels = [items[1] for items in self.data]
        return labels

    
class GroupDataset(Dataset):
    def __init__(self, 
                dataset_dir, group_name):
        
        self.dataset_dir = dataset_dir
        self.group_name = group_name
        self.data = self._get_data(dataset_dir)
        self.classes = list(self._get_classes(self.data))

    def __getitem__(self, index):
        if isinstance(index, slice):
            start = 0 if index.start is None else index.start
            stop = len(self) if index.stop is None else index.stop
            step = 1 if index.step is None else index.step
            return (self[i] for i in range(start, stop, step))
        img_path, pid, group = self.data[index]
        img = read_image(img_path)
        return img, {'class_id': pid, self.group_name: group, 'dataset_dir':self.dataset_dir, 'file':img_path}

    def __len__(self):
        return len(self.data)

    def _get_classes(self, data):
        classes = set([items[1] for items in data])
        return classes
    
    def _get_group(self, data):
        groups = np.unique([items[2] for items in data])
        return groups
    
    def get_group_ids(self, k=None):
        data = self.data
        groups = self._get_groups(data)
        counts = [0] *len(groups)
        res = []
#         res = [np.where(groups==items[2])[0][0] for items in data]
        for items in data:
            seq_id = np.where(groups==items[2])[0][0]
            if k is None or counts[seq_id] < k:
                counts[seq_id]+=1
            else:
                seq_id = -1
            res.append(seq_id)
        return res
    
    def get_group_labels(self):
        data = self.data
        groups = self._get_groups(data)
        res = []
        for group in groups:
            ind = next(i for (i,x) in enumerate(data) if x[2] == group)
            res.append(data[ind][1])
        return res
    
    def get_labels(self):
        labels = [items[1] for items in self.data]
        return labels
    
    def get_group_files(self):
        data = self.data
        groups = self._get_groups(data)
        res = []
        for group in groups:
            ind = next(i for (i,x) in enumerate(data) if x[2] == group)
            res.append(data[ind][0])
        return res
    
    def get_group_lengths(self):
        data = self.data
        groups = self._get_groups(data)
        res = []
        for group in groups:
            ln = len(list(filter(lambda x: x[2] == group, data)))
            res.append(ln)
        return res

    def _get_data(self, dataset_dir):
        dataset_dir = Path(dataset_dir)
        result = []
        
        for class_dir in [x for x in dataset_dir.iterdir() if x.is_dir()]:
            for group in class_dir.iterdir():
                images = []
                for img in group.iterdir():
                    result.append((str(img), class_dir.name, group.name))
        return result

   

    
class DBDataset:
    def __init__(self, seal_type, cfg):
        self.ids = get_db_ids(cfg["conn"], "norppa")
        self.conn = cfg["conn"]
        self.cfg = cfg
        self._fisher_vectors = None
        self._fisher_vectors_seq = None


    def get_ids(self):
        return self.ids
    
    def get_label(self, i):
        return get_label(self.conn, self.ids[i])[0]
    
    def get_label_seq(self, i):
        return self._get_sequences()[i]
    
    def get_labels(self):
        result = []
        for (i, el) in enumerate(self.ids):
            result.append(self.get_label(i))
        return result
    
    def _get_sequences(self):
        seqs = np.unique(self.get_labels())
        return seqs
    
    def get_sequence_ids(self):
        data = self.ids
        seqs = self._get_sequences()
        res = [np.where(seqs==self.get_label(i))[0][0] for (i,item) in enumerate(data)]
        return res

    def get_fisher_vector(self, i):
        if self._fisher_vectors is not None:
            return self._fisher_vectors[i]
        patch_features = get_patch_features(self.conn, self.ids[i])
        encoded = fisher_single(patch_features, self.cfg)
        return encoded

    def get_fisher_vectors(self):
        if self._fisher_vectors is None:
            db_ids, db_features = get_patch_features_multiple_ids(self.conn, self.ids)
            encoded = fisher_multiple(db_features, db_ids, self.cfg)
            self._fisher_vectors = encoded
        return self._fisher_vectors


    def get_fisher_vectors_seq(self):
        if self._fisher_vectors_seq is None:
            db_ids, db_features = get_patch_features_multiple_ids(self.conn, self.ids)
            sequence_ids = self.get_sequence_ids()
            seqs = np.array(sequence_ids).squeeze()

            db_ids = [seqs[self.ids.index(db_id)] for db_id in db_ids]
            encoded = fisher_multiple(db_features, db_ids, self.cfg)
            self._fisher_vectors_seq = encoded
        return self._fisher_vectors_seq

    def get_patches(self, i):
        return get_patches(self.conn, i)
    
    def get_seq_img_ids(self, seq_id):
        data = self.get_sequence_ids()
        ids = [i for (i,x) in enumerate(data) if x == seq_id]
        res = [self.ids[ind] for ind in ids]
        return res
    
    def get_patches_seq(self, seq_id):
        ids = self.get_seq_img_ids(seq_id)
        return get_patches_multiple(self.conn, ids)