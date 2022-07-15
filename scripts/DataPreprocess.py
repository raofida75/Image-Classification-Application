from tensorflow.keras.utils import to_categorical
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# reading features and the labels
def read_file(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    X = np.array(data[0])
    y = np.array(data[1])
    return X,y


# normalize x vals between 0 and 1, divide them by 255
def normalize(X): 
    X = X / 255
    return X 


# one hot encoding
def one_hot_encoding(y):
    classes = len(np.unique(y))
    y = to_categorical(y, classes)
    return y


def shuffle_dataset(X, y):
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    return X[p], y[p]


# combining all the above functions to read all the data and preprocess it.
def load_dataset():
    
    X_train, y_train = read_file('../data/train.pkl')
    X_test, y_test = read_file('../data/test.pkl')
    
    X_train, y_train = shuffle_dataset(X_train, y_train)
    X_test, y_test = shuffle_dataset(X_test, y_test)

    scaled_train = normalize(X_train)
    scaled_test = normalize(X_test)
    
    one_hot_train = one_hot_encoding(y_train)
    one_hot_test = one_hot_encoding(y_test)
    
    scaled_train,scaled_valid,one_hot_train,one_hot_valid = train_test_split(scaled_train, one_hot_train, test_size=0.2, random_state=0)
    
    signnames = pd.read_csv('signnames.csv').iloc[:,1]
    
    return signnames, scaled_train, one_hot_train, scaled_valid, one_hot_valid, scaled_test, one_hot_test
