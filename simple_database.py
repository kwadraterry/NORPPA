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
