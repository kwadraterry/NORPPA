import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# export CUDA_DEVICE_ORDER=PCI_BUS_ID & export CUDA_VISIBLE_DEVICES=2

# import config

from config import config


from datasets import SimpleDataset, DatasetSlice, GroupDataset, COCOWBIADataset

from segmentation.segmentation import segment
from tonemapping.tonemapping import tonemap_step
from norppa_tools import compose_sequential, get_smart_shrink_step, load_pickle, curry, apply_pipeline_dataset, apply_sequential, curry_sequential, print_topk_accuracy, save_pickle, print_step, crop_step, noise_step, crop_label_step, get_save_step
from reidentification.identify import identify, apply_geometric, encode_patches, getDISK, getKeyNetAffNetHardNet, getHessAffNetHardNet, extract_patches
from reidentification.find_matches import find_matches
from pattern_extraction.extract_pattern import extract_pattern
from temp.save_subimages import create_bg_step
from temp.put_bg import add_bg_step

import json

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

def change_dir(path, new_dir):
    name = os.path.basename(path)
    return os.path.join(new_dir, name)

def check_exist_step(input, dest_dir):
    image, label = input
    path = change_dir(label['file'], dest_dir)
    
    if os.path.isfile(path):
        return []   
    else:
        return [(image, label)]

def clear_step(input):
    return []

def update_annotation(input, in_annotation_file, out_annotation_file, update_key="bbox", key='uuid'):
    with open(in_annotation_file, mode='r') as json_file:
        json_dict = json.load(json_file)
    annots = {}
    for annot in json_dict["annotations"]:
        annots[annot[key]] = annot
    for (_, label) in input:
        uuid = label[key]
        # print("Old bbox: ", annots[uuid][update_key])
        if uuid in annots:
            if (update_key in label) and (label[update_key] is not None):
                annots[uuid][update_key] = list(label[update_key])
        else:
            annots[uuid] = label
        # print("New bbox: ", annots[uuid][update_key], label[update_key])
    json_dict["annotations"] = list(annots.values())
    with open(out_annotation_file, mode='w') as json_file:
        json.dump(json_dict, json_file, default=int)



def process_datasets(datasets):
    for (dataset_name, dataset, segment_dataset, pattern_dataset) in datasets:
        print()
        print(f"Dataset name: {dataset_name}")

        print(f"Path to the dataset: {dataset.dataset_dir}")

        print(f"Creating datasets {segment_dataset} and {pattern_dataset}...")

        print()

        # out_annotation_file = "/ekaterina/work/data/miewid__norppa/export_annotations_070424/instances_train2024_new.json"
        cfg = config()
        # cfg["seem"]["species"] = 'giraffe'

        if segment_dataset is not None:
            os.makedirs(segment_dataset, exist_ok=True)
        # os.makedirs(pattern_dataset, exist_ok=True)
        pipeline = [
                    # lambda x: [x[i] for i in range(0, 3)],
                    compose_sequential(
                    # curry(check_exist_step, segment_dataset),
                    # curry(tonemap_step),
                    # debug_step,
                    crop_label_step,
                    # get_smart_shrink_step(480),
                    curry(segment, cfg["seem"], instance_segmentation=False, non_segmented_dir="/ekaterina/work/data/miewid__norppa/images_original_not_segmented"),
                    crop_step,
                    # curry(add_bg_step, background_dir='/ekaterina/work/data/coco_leopard/bgs'),
                    get_save_step(segment_dataset),
                    # curry(extract_pattern, model=cfg["unet"]),
                    # get_save_step(pattern_dataset),
                    # clear_step
                    ),
                    curry(update_annotation, dataset.annotation, dataset.annotation, update_key="bbox", key='uuid')
                    ]

        apply_pipeline_dataset(dataset, pipeline, True)
        print()

def main():
    
    # datasets = [  ("norppa_database", 
    #                     GroupDataset("/ekaterina/work/data/norppa_database", "viewpoint"), 
    #                     "/ekaterina/work/data/norppa_database_segmented", 
    #                     "/ekaterina/work/data/norppa_database_segmented_pattern"),]
    datasets = [  
        # ("miewid_norppa_train", 
        # COCOWBIADataset("/ekaterina/work/data/miewid__norppa/images", "/ekaterina/work/data/miewid__norppa/annotations/left/instances_train2023.json"), 
        # None,
        # "/ekaterina/work/data/miewid__norppa/images_pattern"),
        # ("miewid_norppa_test", 
        # COCOWBIADataset("/ekaterina/work/data/miewid__norppa/images", "/ekaterina/work/data/miewid__norppa/annotations/left/instances_test2023.json"), 
        # None,
        # "/ekaterina/work/data/miewid__norppa/images_pattern"),
        # ("miewid_norppa_val", 
        # COCOWBIADataset("/ekaterina/work/data/miewid__norppa/images", "/ekaterina/work/data/miewid__norppa/annotations/left/instances_val2023.json"), 
        # None,
        # "/ekaterina/work/data/miewid__norppa/images_pattern")
        # ("zebra_train", 
        # COCOWBIADataset("/ekaterina/work/data/zebra/images_segmented_cropped", "/ekaterina/work/data/zebra/annotations/instances_train2024.json"), 
        # "/ekaterina/work/data/zebra/images_segmented_fake_bg",
        # None),
        # ("zebra_test", 
        # COCOWBIADataset("/ekaterina/work/data/zebra/images_segmented_cropped", "/ekaterina/work/data/zebra/annotations/instances_test2024.json"), 
        # "/ekaterina/work/data/zebra/images_segmented_fake_bg",
        # None),
        # ("zebra_val", 
        # COCOWBIADataset("/ekaterina/work/data/zebra/images_segmented_cropped", "/ekaterina/work/data/zebra/annotations/instances_val2024.json"),  
        # "/ekaterina/work/data/zebra/images_segmented_fake_bg",
        # None),
        # ("leo_train", 
        # COCOWBIADataset("/ekaterina/work/data/coco_leopard/images_segmented", "/ekaterina/work/src/MiewID/wbia_miew_id/annotations/leopard/instances_train2023.json"), 
        # "/ekaterina/work/data/coco_leopard/images_fake_bg",
        # None),
        # ("leo_test", 
        # COCOWBIADataset("/ekaterina/work/data/coco_leopard/images_segmented", "/ekaterina/work/src/MiewID/wbia_miew_id/annotations/leopard/instances_test2023.json"), 
        # "/ekaterina/work/data/coco_leopard/images_fake_bg",
        # None),
        # ("leo_val", 
        # COCOWBIADataset("/ekaterina/work/data/coco_leopard/images_segmented", "/ekaterina/work/src/MiewID/wbia_miew_id/annotations/leopard/instances_val2023.json"),  
        # "/ekaterina/work/data/coco_leopard/images_fake_bg",
        # None),
        # ("giraffe_train", 
        # COCOWBIADataset("/ekaterina/work/data/giraffe_tamil/coco/images/test2023", "/ekaterina/work/data/giraffe_tamil/coco/annotations/instances_train2023.json"), 
        # "/ekaterina/work/data/giraffe_tamil/coco/images_segmented",
        # None),
        # ("giraffe_test", 
        # COCOWBIADataset("/ekaterina/work/data/giraffe_tamil/coco/images/test2023", "/ekaterina/work/data/giraffe_tamil/coco/annotations/instances_val2023.json"), 
        # "/ekaterina/work/data/giraffe_tamil/coco/images_segmented",
        # None),
        # ("norppa_original", 
        # COCOWBIADataset("/ekaterina/work/data/miewid__norppa/images_original", "/ekaterina/work/data/miewid__norppa/export_annotations_070424/instances_train2024.json"), 
        # "/ekaterina/work/data/miewid__norppa/images_original_segmented", None),
        # ("norppa_original", 
        # COCOWBIADataset("/ekaterina/work/data/miewid__norppa/images_original", "/ekaterina/work/data/miewid__norppa/export_annotations_070424/instances_val2024.json"), 
        # "/ekaterina/work/data/miewid__norppa/images_original_segmented", None),
        # ("norppa_original", 
        # COCOWBIADataset("/ekaterina/work/data/miewid__norppa/images_original", "/ekaterina/work/data/miewid__norppa/export_annotations_070424/instances_test2024.json"), 
        # "/ekaterina/work/data/miewid__norppa/images_original_segmented", None),     
        ("ringedseals_test", 
        COCOWBIADataset("/ekaterina/work/data/ringedseals_wildme/images_tonemapped", "/ekaterina/work/data/ringedseals_wildme/annotations/instances_test2024.json"), 
        "/ekaterina/work/data/ringedseals_wildme/images_segmented", None),   
        ]
    process_datasets(datasets)


if __name__ == "__main__":
    main()
