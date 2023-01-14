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
    
    
    
    
