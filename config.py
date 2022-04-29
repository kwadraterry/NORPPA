    
import sys
from pathlib import Path

from pattern_extraction.extract_pattern import extract_pattern

file_folder = Path(__file__).resolve().parent
sys.path.append(str(file_folder / "reidentification/hesaff_pytorch"))


from HessianAffinePatches import init_affnet, init_orinet, init_hardnet
from segmentation.detectron_segment import create_predictor
from pattern_extraction.extract_pattern import create_unet
from torchvision.datasets.utils import download_url

def init_file(path, url, allow_download=True):
    if Path(path).exists():
        return path
    elif allow_download:
        download_url(url, Path(path).parent, Path(path).name)
        return path
    else:
        raise Exception("The file {path} is not found!")

def config(use_cuda=True, allow_download=True):   

    config =  {}     
    base_dir = Path(__file__).resolve().parent

    config["net"] = init_hardnet(init_file(base_dir/"models/HardNet++.pth", 
                                            "https://github.com/kwadraterry/NORPPA/raw/models/models/HardNet++.pth", 
                                            allow_download=allow_download), 
                                use_cuda=use_cuda)
    affnet = init_affnet(init_file(base_dir/"models/AffNet.pth", 
                                            "https://github.com/kwadraterry/NORPPA/raw/models/models/AffNet.pth", 
                                            allow_download=allow_download), 
                                use_cuda=use_cuda)
    orinet = init_orinet(init_file(base_dir/"models/OriNet.pth",  
                                            "https://github.com/kwadraterry/NORPPA/raw/models/models/OriNet.pth", 
                                            allow_download=allow_download), 
                                use_cuda=use_cuda)
    codebooks_path = init_file(base_dir/'codebooks/codebooks.pickle',
                               "https://github.com/kwadraterry/NORPPA/raw/models/codebooks/codebooks.pickle", 
                               allow_download=allow_download)
    config["codebooks_path"] = codebooks_path 
    config["codebooks"] = None
    config["hesaff_args"] = {'init_sigma': 1.3213713243956968, 
                            'mrSize': 9.348280997446642, 
                            'nlevels': 10, 
                            'num_features': 480, 
                            'unsharp_amount': 6.80631647207343, 
                            'unsharp_radius': None,
                            'use_cuda' :use_cuda}
    config["detectron_predictor"] = create_predictor(init_file(base_dir/"models/R-101-FPN_150ims.pth",  
                                            "https://github.com/kwadraterry/NORPPA/raw/models/models/R-101-FPN_150ims.pth", 
                                            allow_download=allow_download),
                                not use_cuda)
    config["unet"] = create_unet(init_file(base_dir/"models/unet_seals_512.hdf5",  
                                            "https://github.com/kwadraterry/NORPPA/raw/models/models/unet_seals_512.hdf5", 
                                            allow_download=allow_download))
    config["hesaff_args"]["AffNet"] = affnet
    config["hesaff_args"]["OriNet"] = orinet

    config["hesaff_args"]["patch_size"] = 32


    config["use_hesaff"] = True
    

    config["pca"] = 64
    config["use_kpca"] = False

    config["n_clusters"] = 1400
    config["features_shape"] = 64
    config["topk"] = 5

    config["kernel"] = "rbf"
    config["use_cuda"] = use_cuda
    config["dataset_dir"] = base_dir/'data'

    config["batch_size"] = 256

    return config