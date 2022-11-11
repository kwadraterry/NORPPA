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
                        default="/ekaterina/work/data/Norppa_orig_tonemapped", 
                        # default="/ekaterina/work/data/bad_orig"
                        help="Path to the full images")
    parser.add_argument("-sp", "--segmented_path",
                        dest="segmented_path",
                        required=False,
                        default="/ekaterina/work/data/Norppa_dataset_segmented",
                        # default="/ekaterina/work/data/bad_segmented"
                        help="Path to the segmented images")
    parser.add_argument("-k", "--keyword",
                        dest="keyword",
                        required=False,
                        default="",
                        help="Path to the segmented images")
    parser.add_argument("-o", "--output",
                        dest="output_path",
                        required=False,
                        default="dataset.pickle",
                        help="Path to the output pickle file")
    args = parser.parse_args()
    
    dataset_info = create_dataset_json(args.full_path, args.segmented_path)
    print(dataset_info)
    with open(args.output_path, 'wb') as f:
        pickle.dump(dataset_info, f)


if __name__ == "__main__":
    main()


