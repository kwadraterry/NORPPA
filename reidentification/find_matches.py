from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from reidentification.identify import fisher_single, do_matching
from reidentification.encoding_utils import calculate_dists


def get_fisher(label, cfg):
    if "fisher" in label:
        return label["fisher"]
    else:
        return fisher_single(label['features'], cfg)

def find_matches(identification_result, cfg):
    if identification_result is None:
        return [identification_result]
    matches, query_labels = identification_result
    query_images = query_labels["labels"]
    if len(query_images) == 1:
        query_fishers = [query_labels["fisher"]]
    else:
        query_fishers = [get_fisher(query_image,cfg)  for query_image in query_images]
    db_encodings = []
    for (j,match) in enumerate(matches):
        db_images = match["db_label"]["labels"]
        if len(db_images)==1:
            db_fishers = [match["db_label"]["fisher"]]
        else:
            db_fishers = [get_fisher(db_image,cfg)  for db_image in db_images]
        
        (dists, _) = calculate_dists(query_fishers, db_fishers)
        dists = np.nan_to_num(dists, nan=2)
        ind1, ind2 = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
 

        query_patch_features = query_images[ind1]['features']
        db_patch_features = db_images[ind2]['features']
        
        query_ellipses = query_images[ind1]['ellipses']
        db_ellipses = db_images[ind2]['ellipses']

        (filt, sorted_inds, similarity) = do_matching(query_patch_features, db_patch_features)
        matches[j]["db_ind"] = ind2
        matches[j]["query_ind"] = ind1
        matches[j]["patches"] = [
            [query_ellipses[k].tolist() for k in filt],
            [db_ellipses[k].tolist() for k in sorted_inds],
            similarity.tolist()
        ]
        matches[j]["inds"] = [filt, sorted_inds]
    return [(matches, query_labels)]