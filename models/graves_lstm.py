import tensorflow as tf
import librosa as lr
import pickle
import glob
from data.dataset import TIMITDataset

LOAD_PICKLE = False

def create_model():
    pass

def train_model():
    if LOAD_PICKLE == False:
        # create the dataset
        MyTimitDataset = TIMITDataset()
        MyTimitDataset.load()
        MyTimitDataset.to_file()
    else:
        filename = glob.glob("timit*.pickle")[0]
        with open(filename, "rb") as dataset_file:
            timit_dataset = pickle.load(dataset_file)

def evaluate_model():
    pass

if __name__ == "__main__":
    create_model()
    train_model()
    evaluate_model()
