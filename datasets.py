import os
from pathlib import Path
from tools import read_image
import csv

from torch.utils.data import Dataset
import os

class DatasetSlice(Dataset):
    def __init__(self, dataset, slice=None):
        self.dataset = dataset
        self.slice = (0, len(self.dataset)) if slice is None else slice
        if hasattr(dataset,'imgs'):
            self.imgs = [dataset.imgs[i] for i in slice]
        if hasattr(dataset,'classes'):
            self.classes = dataset.classes
    def __getitem__(self, index):
        return self.dataset[self.slice[index]]
    def __len__(self):
        return len(self.slice)

class COCOImageDataset(Dataset):

    def __init__(self, 
                dataset_dir,
                annotation,
                split):
        
        self.split = split
        self.dataset_dir = dataset_dir
        self.annotation = annotation

        self.data = self._get_data(split, self.annotation)
        self.classes = list(self._get_classes(self.data))


    def __getitem__(self, index):
        img_path, pid = self.data[index]
        img = read_image(img_path)
        return img, {'class_id': pid, 'file': img_path}

    def __len__(self):
        return len(self.data)

    def _get_classes(self, data):
        classes = set([items[1] for items in data])
        return classes

    def _get_data(self, split, annotation):
        """ Get database from COCO anntations """
        
        result = []
        
        with open(annotation, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if (row["reid_split"] == split):
                    image_path = self._get_image_path(row["file"])
                    image_pid = row["class_id"]
                    result.append((image_path,image_pid))
        return result

    def _get_image_path(self, filename):
        return os.path.join(self.dataset_dir, filename)

    

   
class SimpleDataset(Dataset):
    def __init__(self, 
                dataset_dir):
        
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

    def _get_data(self, dataset_dir):
        dataset_dir = Path(dataset_dir)
        result = []
        
        for class_dir in [x for x in dataset_dir.iterdir() if x.is_dir()]:
            for img in class_dir.iterdir():
                result.append((str(img), class_dir.name))
        return result

   

    
