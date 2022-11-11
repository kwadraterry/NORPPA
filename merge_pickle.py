import pickle

dataset_1="./dataset_train.pickle"
dataset_2="./bad_dataset.pickle"
result = "./full_dataset.pickle"


with open(dataset_1, 'rb') as f:
    dicts_1 = pickle.load(f)
with open(dataset_2, 'rb') as f:
    dicts_2 = pickle.load(f)

dicts_final = dicts_1+dicts_2

with open(result, 'wb') as f:
    pickle.dump(dicts_final,f)

