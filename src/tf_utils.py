from PIL import Image
import h5py
import numpy as np
import tensorflow as tf
import math
import os
import random

def partition_dataset(data_sources, training_percentage, cv_percentage, test_percentage):
    class_id = 0
    dataset = {"train_set_x": np.array([]),
        "train_set_y": np.array([]),
        "cv_set_x": np.array([]),
        "cv_set_y": np.array([]),
        "test_set_x": np.array([]),
        "test_set_y": np.array([]),
        "classes": np.array([])}

    for data_source in data_sources:
        im_array = np.array( [np.array(Image.open(data_source + "\\" + img), 'f') for img in os.listdir(data_source)] )
        print("im_array shape: " + str(im_array.shape))
        im_array_len = im_array.shape[0]
        indexes = [i for i in range(im_array_len)]
        random.shuffle(indexes)

        training_index = int(im_array_len * training_percentage)
        train_set_y = np.empty([1,training_index])
        train_set_y.fill(class_id)
        if len(dataset["train_set_x"]) == 0:
            dataset["train_set_x"] = im_array[indexes[:training_index]]
            dataset["train_set_y"] = train_set_y
        else:
            dataset["train_set_x"] = np.concatenate((dataset["train_set_x"], im_array[indexes[:training_index]]))
            dataset["train_set_y"] = np.concatenate((dataset["train_set_y"], train_set_y), axis=-1)
        assert(dataset["train_set_x"].shape[0] == dataset["train_set_y"].shape[1])

        cv_index = int(im_array_len * (training_percentage + cv_percentage))
        if cv_index > training_index:
            cv_set_y = np.empty([1, cv_index - training_index])
            cv_set_y.fill(class_id)
            if len(dataset["cv_set_x"]) == 0:
                dataset["cv_set_x"] = im_array[indexes[training_index + 1:cv_index]]
                dataset["cv_set_y"] = cv_set_y
            else:
                dataset["cv_set_x"] = np.concatenate((dataset["cv_set_x"], im_array[indexes[training_index + 1:cv_index]]))
                dataset["cv_set_y"] = np.concatenate((dataset["cv_set_y"], cv_set_y), axis=-1)
            assert(dataset["cv_set_x"].shape[0] == dataset["cv_set_y"].shape[1])
        
        if im_array_len > cv_index + 1:        
            test_set_y = np.empty([1, im_array_len - cv_index - 1])
            test_set_y.fill(class_id)                
            if len(dataset["test_set_x"]) == 0:
                dataset["test_set_x"] = im_array[indexes[cv_index + 1:]]
                dataset["test_set_y"] = test_set_y
            else:
                dataset["test_set_x"] = np.concatenate((dataset["test_set_x"], im_array[indexes[cv_index + 1:]]))
                dataset["test_set_y"] = np.concatenate((dataset["test_set_y"], test_set_y), axis=-1)
            assert(dataset["test_set_x"].shape[0] == dataset["test_set_y"].shape[1])

        class_id = class_id + 1
    
    dataset["classes"] = np.array([i for i in range(class_id)])
        
    return dataset

def save_dataset_h5(dataset, target):
    f = h5py.File(target, "w")
    for dname, values in dataset.items():
        f.create_dataset(name=dname, data=values)

def load_dataset_h5(source):
    dataset = h5py.File(source, "r")
    train_set_x = np.array(dataset["train_set_x"][:]) # your train set features
    train_set_y = np.array(dataset["train_set_y"][:]) # your train set labels

    cv_set_x = np.array(dataset["cv_set_x"][:]) # your cross validation set features
    cv_set_y = np.array(dataset["cv_set_y"][:]) # your cross validation set labels

    test_set_x = np.array(dataset["test_set_x"][:]) # your test set features
    test_set_y = np.array(dataset["test_set_y"][:]) # your test set labels

    classes = np.array(dataset["classes"][:]) # the list of classes
    
    return train_set_x, train_set_y, cv_set_x, cv_set_y, test_set_x, test_set_y, classes

def load_dataset():
    train_dataset = h5py.File('E:/Projects/Hach2019/CourseraData/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('E:\\Projects\\Hach2019\\CourseraData\\test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if class A, 1 if non-class A), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def predict(X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder("float", [12288, 1])
    
    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction

def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3'] 
                                                           # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3
    
    return Z3
    