"""

@author: dhanjal(50248990),d25(50248725)

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from numpy.random import randn
import math

IMAGE_SIZE = 28
NumOfmodels = 10
stopItr = False
np.seterr(divide='ignore', invalid='ignore')

# Extracts the images of size 28 x 28 and combines it into shape N X 784
# Extracts the labels and reshapes it into N X 1.
def extract_files(filename, num_images,isImg):
    print('Extracting', filename)
    with open(filename,'rb') as bytestream:
        bytestream.read(8)
        if isImg:
            buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        else:
            buf = bytestream.read(1 * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        if isImg:
            data = data.reshape(num_images, IMAGE_SIZE * IMAGE_SIZE)
    return data


# Modifies the labels of shape N X 1 to N X 10 thus represting each sample in their respective models.
def modify_label(labels):
    new_labels = np.zeros((labels.shape[0],NumOfmodels))
    new_labels[np.arange(new_labels.shape[0]), labels.astype(int)] = 1
    return new_labels;


# Normalizes the images data between 0 to 1.
def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return np.round_(out,decimals=4)  
    

# updates the hyper parameters w_map and S_N(variance)
def hyper_para_bayes_logistic(m_0, S_0, Theta, y):
    w_map = m_0
    S_N = np.linalg.inv(S_0)
    Theta = Theta.T
    for i in range(Theta.shape[0]):
        S_N = S_N + y[i]*(1-y[i])*np.matmul(Theta[i].T, Theta[i])
    return w_map, S_N

#prediction of data with updated w_map and S_N
#mu_a = W_map.T  X theta
#var_a = theta x S_N X theta.T
# returns predicted output using formula 1/(1+exp(-x))
def pred_bayes_logistic(w_map, S_N, theta):
    mu_a = np.dot(theta,w_map.T) # N X 785 * 785 X 1 = N X 1
    var_a = np.dot(np.dot(theta, S_N), theta.T) # N X 785 * 785 X 785 * 785 X N = N X N
    kappa_var = np.power(abs(1 + math.pi*var_a/8),(-0.5))
    x = np.matmul(kappa_var,mu_a) # N X N * N X 1 = N X 1
    maxVal = x.max(0)
    sigmoid = np.exp(x - maxVal) #normalization for exp
    
    return sigmoid/(1 + sigmoid)

#computes the predicted output and compares it with our original labels classification given.
#if accuracy is low update the weight and recompute
def compute_regression(inputArr,org_labels,labels_t,weight,variance,etaVal):
    numOfFeature = inputArr.shape[0]
    output_a = pred_bayes_logistic(weight, variance, inputArr)
    org_labels = org_labels.reshape(org_labels.shape[0],1)
    
    error = np.matmul(inputArr.T,(output_a - org_labels))
    weight_lambda = weight
    weight_lambda[0,0] = 0    
    weight = weight - etaVal * ((error.T + np.matmul(weight_lambda,variance))/numOfFeature)
    return (weight,accuracy)
  

#divides data into mini batches of 10000 and sends data to compute_regression for calculation of prediction,accuracy and updated hyperparameters
def process_input(Theta,y,org_labels,weight,noOfItr,Variance,etaVal):
    inputdata = im2double(Theta)
    inputdata = np.insert(inputdata, 0, 1, axis=1)
    N = inputdata.shape[0]
    minibatch_size = min(10000,N)
    num_epochs = noOfItr
    accuracy_arr = []
    labels = y
    for epoch in range(num_epochs):
        for i in range(int(N / minibatch_size)):
            lower_bound = i * minibatch_size
            upper_bound = min((i+1)*minibatch_size, N)
            newinputdata = inputdata[lower_bound:upper_bound,:]
            newinputdataLabels = org_labels[lower_bound:upper_bound]
            newlabels = labels[lower_bound:upper_bound]
            print('Iteration No is:',epoch)
            weight,accuracy = compute_regression(newinputdata,newinputdataLabels,newlabels,weight,Variance,etaVal)
            print('Accuracy is',accuracy)
            accuracy_arr.append(accuracy)
    return weight,accuracy,accuracy_arr 

#main class for training and prediction
class logistic_sigmoid_model:
    def _init(self):
        self.hyper_para_bayes_logistic = hyper_para_bayes_logistic
        self.pred_bayes_logistic = pred_bayes_logistic
    def training(self, Theta, y,org_labels):
        self.w0 = np.random.normal(0, 1,(1,(Theta.shape[1]+1)))
        self.S0 = np.diag(np.random.normal(0, 1, (Theta.shape[1]+1)))
        variance = np.linalg.inv(self.S0)
        # Theta n*m (n samples, m features), y n*1
        # variance mxm
        # self.w0 1 X 785
        self.w0,accuracy,_  = process_input(Theta,y,org_labels,self.w0,10,variance,0.9)
        inputdata = im2double(Theta)
        inputdata = np.insert(inputdata, 0, 1, axis=1)
        inputdata = inputdata[1:10000,:]
        final_y = pred_bayes_logistic(self.w0, variance, inputdata)
        
        self.w_map, self.S_N = self.hyper_para_bayes_logistic(self.w0, self.S0,inputdata, final_y)
    def predict(self, theta):
        theta = np.insert(theta, 0, 1, axis=1)
        return self.pred_bayes_logistic(self.w_map, self.S_N, theta)

# checks the accuracy of our predicted data and original data
def check_accuracy(isEqual,total_num):
    numOfmatch = np.count_nonzero(isEqual == True)
    print('num of matches',numOfmatch)
    return (numOfmatch/total_num)

# checks the accuracy of our predicted data and original data
def check_prediction_equality(org_labels,predicted_labels,isUsps):
    predicted_labels_idx = np.argmax(predicted_labels,axis=1)
    if isUsps:
        predicted_labels_idx = predicted_labels_idx.reshape(predicted_labels_idx.shape[0],1)
        org_labels = org_labels.reshape(org_labels.shape[0],1)
    isEqual = np.equal(org_labels,predicted_labels_idx);
    total_num = predicted_labels.shape[0]
    accuracy = check_accuracy(isEqual,total_num)
    return accuracy

# trains the data for each model seperately    
def multiclass_sigmoid_logistic(Theta, Y,org_labels):
    n_class = Y.shape[1]
    models = []
    for i in range(n_class):
        models.append(logistic_sigmoid_model())
        models[i]._init()
        models[i].training(Theta, Y[:, i],org_labels)
        
    return models

#classifies the test data based on hyperparameters obtained after training
def pred_multiclass_sigmoid_logistic(theta, Theta, Y,org_labels):
    models = multiclass_sigmoid_logistic(Theta, Y,org_labels)
    props = np.zeros((10000,10))
    for i in range(len(models)):
        props[:,i] = (models[i].predict(theta)).reshape(10000)
    accuracy = check_prediction_equality(org_labels,props,0)
    return accuracy

#starts the application
#divides the data into training data,validation data and test data
def main():
    inputData = extract_files('/Users/deepikachaudhary/Documents/Sem-1_UB/ML/proj3_images/mnist/train-images-idx3-ubyte',60000,1)
    labelData = extract_files('/Users/deepikachaudhary/Documents/Sem-1_UB/ML/proj3_images/mnist/train-labels-idx1-ubyte',60000,0)
    train_data_filename = inputData[1:55000,:]
    train_labels_filename = labelData[1:55000]
    val_data_filename = inputData[55000:60000,:]
    val_labels_filename =labelData[55000:60000]
    test_data_filename = extract_files('/Users/deepikachaudhary/Documents/Sem-1_UB/ML/proj3_images/mnist/t10k-images-idx3-ubyte',10000,1)
    test_labels_filename = extract_files('/Users/deepikachaudhary/Documents/Sem-1_UB/ML/proj3_images/mnist/t10k-labels-idx1-ubyte',10000,0)
    Y = modify_label(train_labels_filename)
    Theta = train_data_filename
    theta = test_data_filename
    accuracy = pred_multiclass_sigmoid_logistic(theta, Theta, Y,train_labels_filename)

main()