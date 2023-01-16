# import config
from argparse import ArgumentParser
from pathlib import Path
from segmentation.train_dataset import create_dataset_json
import pickle


def main():
    parser = ArgumentParser()
    parser.add_argument("-fp", "--full_path",
                        dest="full_path",
                        required=False,
                        default="/ekaterina/work/src/NORPPA/data/full images/source_query",
                        help="Path to the full images")
    parser.add_argument("-sp", "--segmented_path",
                        dest="segmented_path",
                        required=False,
                        default="/ekaterina/work/src/NORPPA/data/full images/segmented_query",
                        help="Path to the segmented images")
    parser.add_argument("-o", "--output",
                        dest="output_path",
                        required=False,
                        default="dataset_test.pickle",
                        help="Path to the output pickle file")
    args = parser.parse_args()
    
    for d in ["train", "val"]:
        DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
        MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
        balloon_metadata = MetadataCatalog.get("balloon_train")


if __name__ == "__main__":
    main()


