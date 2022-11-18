from sql import *
from reidentification.identify import fisher_single, fisher_multiple
from cyvlfeat.fisher import fisher


class SimpleDatabase:
    def __init__(self, fisher_vectors, labels, patches):
        self.fisher_vectors = fisher_vectors
        self.labels = labels
        self.patches = patches

    def __init__(self, features, labels, patch_features, all_ells):
        self.fisher_vectors = features
        self.labels = labels
        self.patches = list(zip(patch_features, all_ells))

    def get_label(self, i):
        return self.labels[i]

    def get_fisher_vector(self, i):
        return self.fisher_vectors[i]

    def get_fisher_vectors(self):
        return self.fisher_vectors

    def get_patches(self, i):
        return self.patches[i]
    
    
    
class DBDatabase:
    def __init__(self, seal_type, conn, cfg):
        self.ids = get_db_ids(conn, "norppa")
        self.conn = conn
        self.cfg = cfg
#         self.fisher_vectors = fisher_vectors
#         self.labels = labels
#         self.patches = patches

#     def __init__(self, features, labels, patch_features, all_ells):
#         self.fisher_vectors = features
#         self.labels = labels
#         self.patches = list(zip(patch_features, all_ells))

    def get_ids(self):
        return self.ids
    
    def get_label(self, i):
        return get_label(self.conn, self.ids[i])

    def get_fisher_vector(self, i):
        patch_features = get_patch_features(self.conn, self.ids[i])
        encoded = fisher_single(patch_features, self.cfg)
        
        return encoded

    def get_fisher_vectors(self):
        db_ids, db_features = get_patch_features_multiple_ids(self.conn, self.ids)
        encoded = fisher_multiple(db_features, db_ids, self.cfg)
        return encoded

    def get_patches(self, i):
        return get_patches(self.conn, i)
