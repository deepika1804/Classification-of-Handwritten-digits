import numpy as np
import pandas as pd
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
from decimal import *
import gzip
import os
from PIL import Image
import glob

IMAGE_SIZE = 28
FEATURE_SIZE = IMAGE_SIZE * IMAGE_SIZE + 1
NUM_CLASS = 10
LAMBDA = 0.001

# extract the data from the file and insert 1 for the bias term
def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float16)
        data = data.reshape(num_images, IMAGE_SIZE * IMAGE_SIZE)
        # normalise data
        data = normalise_data(data)
        # add 1 in the beginning for bias term
        data = np.insert(data, 0, 1, axis=1)
    return data

# normalise the image data to lie between 0 to 1
def normalise_data(data):
    arr = data.reshape(data.shape[0] * data.shape[1],1)
    max = np.max(arr)
    min = np.min(arr)
    diff = max - min;
    x = arr / diff
    return x.reshape(data.shape)

# data normalisation for USPS data 
# the background is white for USPS
# the background is black for MNIST
def normalise_data_abs(data):
    arr = data.reshape(data.shape[0] * data.shape[1],1)
    max = np.max(arr)
    min = np.min(arr)
    diff = max - min;
    x = abs(arr - max) / diff
    return x.reshape(data.shape)

# extract the labels of the input data from input file
def extract_label(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int16)
    return labels

# extract USPS data
def extract_usps_data():
    files = [f for f in glob.glob("/Users/manpreetdhanjal/Downloads/proj3_images/Numerals/*[0-9]")]

    data =np.empty((1, 784))
    usps_labels_training = []
    for f in files:
        img =  glob.glob(f +"/*.png")
    
        for i in img:
            arr = get_image_data(i)
            data = np.append(data,arr,axis=0)
            usps_labels_training.append(int(f.split("/")[-1]))
            print('data is',(data).shape)

    data = data[1:,:]
    data = normalise_data_abs(data)
    data = np.insert(data, 0, 1, axis=1)
    val = usps_labels_training
    usps_labels_training = np.reshape(usps_labels_training,(len(usps_labels_training),1))

    return (data, usps_labels_training)

def get_image_data(filename):
    imgData = (Image.open(filename).convert('L'))
    imgData = imgData.resize((28,28),Image.ANTIALIAS)
    imgData = np.asarray(imgData)
    s = 28 * 28
    img_wide = imgData.reshape(1, s)
    return img_wide

# applies the softmax and returns array with probabilities of all classes
def softmax(prediction):
    exp_vec = np.exp(prediction)
    res_exp_vec = np.zeros((prediction.shape[0],prediction.shape[1]))
    for i in range(prediction.shape[0]):
        res_exp_vec[i,:] = exp_vec[i,:]/np.sum(exp_vec[i,:])
    return res_exp_vec

# convert the input labels to Nx10 vector representing the probability for each class
def format_labels(labels):
    result = np.zeros((labels.size,NUM_CLASS))
    result[np.arange(result.shape[0]), labels] = 1
    return result

# tunes the hyperparameters for neural networks
# trains the NN for various values of eta, lambda and number of nodes
# tests on validation dataset
def tune_parameters(train_input, train_label):   
    # grid search
    eta_arr = [0.03, 0.1, 0.3]
    lambda_arr = [0.01, 0.03, 0.1, 0.3]
    hidden_node_num_arr = [100, 300, 500, 700]
    minibatch_size = 10000
    
    # optimal accuracy
    opt_accuracy = 0
    
    # divide into training & validation set
    validation_start = np.round(int(train_input.shape[0] * 0.9))
    training_input = train_input[0:validation_start-1,:]
    training_output = train_label[0:validation_start-1]
    validation_input = train_input[validation_start:train_input.shape[0],:]
    validation_output = train_label[validation_start:train_input.shape[0]]
    
    accuracy_arr = np.zeros((64,1))
    i=0
    for learning_param in eta_arr:
        for lambda_param in lambda_arr:
            for num_hidden_nodes in hidden_node_num_arr:
                # train the NN for given parameters
                [weight_1, weight_2] = train_neural_network(training_input, training_output, minibatch_size, 
                                                            learning_param, lambda_param, num_hidden_nodes)
                
                # test on validation set
                accuracy = test_neural_network(weight_1, weight_2, validation_input, validation_output)
                print(accuracy)
                accuracy_arr[i,0] = accuracy
                i=i+1
                if accuracy > opt_accuracy:
                    eta_opt = learning_param
                    lambda_opt = lambda_param
                    num_nodes_opt = num_hidden_nodes
                    
    return (eta_opt, lambda_opt, num_nodes_opt, accuracy_arr)
    
# train neural network
def train_neural_network(training_input, train_labels, minibatch_size, learning_param, lambda_param, num_hidden_nodes):
    
    # array to save the accuracy for each iteration
    accuracy = np.zeros((1000, 1))
    # format labels to Nx10 matrix
    training_output = format_labels(train_labels)
    
    # parameters
    total_iterations = 1000
    i = 0
    output_nodes = 10
    
    # randomly initiate the weights for the two layers
    weight_1 = np.random.normal(loc=0, scale=0.3, size = (FEATURE_SIZE, num_hidden_nodes))
    weight_2 = np.random.normal(loc=0, scale=0.3, size = (num_hidden_nodes + 1, output_nodes))
    
    N = training_input.shape[0]
    
    while i < total_iterations:
        # train in minibatches - SGD
        for j in range(int(N/minibatch_size)):
            # get the limits for the input and outputs
            lower_bound = j * minibatch_size 
            upper_bound = min((j+1)*minibatch_size, N)
            
            train_batch_input = training_input[lower_bound:upper_bound,:]
            train_batch_output = training_output[lower_bound:upper_bound,:]
            
            #60000x785 785x50 = 60000x50
            layer_1_prediction = np.matmul(train_batch_input, weight_1)
            # applying activation function to it, 60000x50
            layer_1_value = np.tanh(layer_1_prediction)
            #layer_1_value = relu(layer_1_prediction)
            # insert 1 for bias 60000x51
            layer_1_value_bias = np.insert(layer_1_value, 0, 1, axis=1)

            # 60000x51 51x10 = 60000x10
            layer_2_prediction = np.matmul(layer_1_value_bias, weight_2)
            exp_vec = softmax(layer_2_prediction)

            # 60000x10
            del_2 = exp_vec - train_batch_output
            # 60000x51 60000x10 = 51x10
            E_2 = np.matmul(layer_1_value_bias.T, del_2)
            temp = LAMBDA * weight_2;
            temp[0,:] = 0;
            weight_2 = weight_2 - learning_param/N * (E_2 + temp)

            # 60000x50 60000x10 51x10
            del_1 = (1 - np.square(layer_1_value)) * (np.matmul(del_2, weight_2[1:,:].T))
            # 60000x785 60000x50 = 785x50
            E_1 = np.matmul(train_batch_input.T, del_1)
            temp = LAMBDA * weight_1;
            temp[0,:] = 0;
            weight_1 = weight_1 - learning_param/N * (E_1 + temp)
        
        # calculate and save accuracy for each iteration 
        accuracy[i,0] = test_neural_network(weight_1, weight_2, training_input, train_labels)
        print(accuracy[i,0])
        i=i+1

    return (weight_1, weight_2, accuracy)

# returns accuracy of the trained model on given data
def test_neural_network(weight_1, weight_2, test_input, test_output):
    layer_1_prediction = np.tanh(np.matmul(test_input, weight_1))
    layer_1_value_bias = np.insert(layer_1_prediction, 0, 1, axis=1)
    layer_2_prediction = softmax(np.matmul(layer_1_value_bias, weight_2))
    predicted_output = np.argmax(layer_2_prediction, axis=1)
    corr = np.sum(np.equal(predicted_output,test_output))
    return corr/test_output.shape[0]

# returns accuracy on test and usps data
def print_all_accuracies(weight_1, weight_2, train_input, train_label, test_input, test_label, usps_input, usps_label):
    # accuracy on train data
    train_acc = test_neural_network(weight_1, weight_2, train_input, train_label)
    # accuracy on test data
    test_acc = test_neural_network(weight_1, weight_2, test_input, test_label)
    # accuracy on usps data
    usps_acc = test_neural_network(weight_1, weight_2, usps_input, usps_label)
    
    print("Accuracy on train data:")
    print(train_acc)
    print("Accuracy on test data:")
    print(test_acc)
    print("Accuracy on USPS data:")
    print(usps_acc)
    
    return train_acc, test_acc, usps_acc

# run SNN
def SNN():
    # extract MNIST data 
    test_input = extract_data('/Users/manpreetdhanjal/Downloads/t10k-images-idx3-ubyte.gz', 10000)
    test_label = extract_label('/Users/manpreetdhanjal/Downloads/t10k-labels-idx1-ubyte.gz', 10000)
    train_input = extract_data('/Users/manpreetdhanjal/Downloads/train-images-idx3-ubyte.gz', 60000)
    train_label = extract_label('/Users/manpreetdhanjal/Downloads/train-labels-idx1-ubyte.gz', 60000)

    # extract USPS data
    usps_input, usps_label = extract_usps_data()

    # tune hyper parameters using grid search 
    [eta_opt, lambda_opt, num_nodes_opt, accuracy_arr] = tune_parameters(train_input, train_label)

    mini_batch_size = 10000
    #directly train neural network by providing the hyper parameter values
    train_neural_network(train_input, train_label, mini_batch_size, eta_opt, lambda_opt, num_nodes_opt)

    print_all_accuracies(weight_1, weight_2, train_input, train_label, test_input, test_label)

def main():
    SNN()

main()