import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from numpy.random import randn
from PIL import Image
import glob
import gzip

####################################### code for Logistic Regression starts here #######################################

IMAGE_SIZE = 28
NumOfmodels = 10
stopItr = False
np.seterr(divide='ignore', invalid='ignore')
FEATURE_SIZE = IMAGE_SIZE * IMAGE_SIZE + 1
NUM_CLASS = 10
LAMBDA = 0.001

weight = np.random.rand(784, 10);
weight = np.insert(weight, 0, 0.01, axis=0)

accuracy_arr = []

#helper function to get the images 
def get_image_data(filename):
    imgData = (Image.open(filename))
    imgData = imgData.resize((28,28),Image.ANTIALIAS)
    #imgData = map(list, imgData)
    imgData = np.asarray(imgData)
    s = 28 * 28
    img_wide = imgData.reshape(1, s)
    return img_wide

# imports the USPS data use isTest = 1 for test folder and 0 for numerals
# Extracts the labels and reshapes it into N X 1.
def extract_usps_files(filesPaths,isTest):
    usps_data =np.empty((1, 784))
    usps_data_labels = []
    print(filesPaths)
    j = 9
    target = 150
    for f in filesPaths:
        img =  glob.glob(f +"/*.png")
        count = 0
        for i in img:
            count = count + 1
            arr = get_image_data(i)
            usps_data = np.append(usps_data,arr,axis=0)
            if isTest:
                usps_data_labels.append(j)
                if count == target:
                    target = target + 150
                    j = j-1
            else:
                usps_data_labels.append(int(f.split("/")[-1]))
            
    usps_data = usps_data[1:usps_data.shape[0],:]     
    usps_data_labels = np.reshape(usps_data_labels,(len(usps_data_labels),1))
    return usps_data,usps_data_labels

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
# checks the accuracy of our predicted data and original data
def check_accuracy(isEqual,total_num):
    numOfitr = np.count_nonzero(isEqual == True)
    return (numOfitr/total_num)

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

#computes the predicted output and compares it with our original labels classification given.
#if accuracy is low update the weight and recompute   
def compute_regression(inputArr,org_labels,labels_t,weight,isTraining,lambdaVal,etaVal,isUsps):
    numOfFeature = inputArr.shape[0]
    lambda_t = lambdaVal
    output_a = np.exp(np.matmul(inputArr,weight))
    output_y = np.round_(output_a,decimals=4)
    sumOfrows = np.sum(output_y, axis=1)
    sumOfrows = np.round_(sumOfrows.reshape(sumOfrows.shape[0],1),decimals = 4)
    result_mat = np.divide(output_y ,sumOfrows);
    accuracy = check_prediction_equality(org_labels,result_mat,isUsps);
    if accuracy  > 1:
        #stop the itr
        return (weight,True,accuracy)
    elif isTraining:
        error = np.matmul(inputArr.T,(result_mat - labels_t))
        weight_lambda = weight
        weight_lambda[0,:] = 0
        weight = (weight - etaVal * (error + lambda_t*weight_lambda)/numOfFeature) 
        return (weight,False,accuracy)
    else:
        return (weight,False,accuracy)

# Normalizes the images data between 0 to 1.
def im2double(im,isUsps):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    if isUsps == 1:
        out = (im.astype('float') - max_val) / (max_val - min_val)
    else:
        out = (im.astype('float') - min_val) / (max_val - min_val)
    
    return np.round_(out,decimals=4)  


#divides data into mini batches of 10000 and sends data to compute_regression for calculation of prediction,accuracy and updated hyperparameters
def process_input(inputdata,inputdataLabels,weight,isTraining,noOfItr,lambdaVal,etaVal,isUsps):
    inputdata = im2double(inputdata,isUsps)
    inputdata = np.insert(inputdata, 0, 1, axis=1)
    labels = modify_label(inputdataLabels);
    stopItr = False
    N = inputdata.shape[0]
    minibatch_size = min(10000,N)
    num_epochs = noOfItr
    accuracy_arr = []
    for epoch in range(num_epochs):
        for i in range(int(N / minibatch_size)):
            lower_bound = i * minibatch_size
            upper_bound = min((i+1)*minibatch_size, N)
            newinputdata = inputdata[lower_bound:upper_bound,:]
            newinputdataLabels = inputdataLabels[lower_bound:upper_bound]
            newlabels = labels[lower_bound:upper_bound,:]
            weight,stopItr,accuracy = compute_regression(newinputdata,newinputdataLabels,newlabels,weight,isTraining,lambdaVal,etaVal,isUsps)
            accuracy_arr.append(accuracy)
    return weight,accuracy,accuracy_arr     

#used for validation data inorder to select best eta and lambda
def validate_hyperParams(lambda_arr,eta_arr,weight,valInput,valOutput):        
    isTraining = True
    final_accuracy = 0.1
    noOfItr = 500
    for eta in eta_arr:
        for L in lambda_arr:
            val_weight,accuracy,_ = process_input(valInput,valOutput,weight,isTraining,noOfItr,L,eta,0)
            if accuracy > final_accuracy:
                final_accuracy = accuracy
                final_eta = eta
                final_lambda = L
    return (final_eta,final_lambda)
#main function for LR
def LR():
	inputData = extract_files('/Users/deepikachaudhary/Documents/Sem-1_UB/ML/proj3_images/mnist/train-images-idx3-ubyte',60000,1)
	labelData = extract_files('/Users/deepikachaudhary/Documents/Sem-1_UB/ML/proj3_images/mnist/train-labels-idx1-ubyte',60000,0)
	test_data_filename = extract_files('/Users/deepikachaudhary/Documents/Sem-1_UB/ML/proj3_images/mnist/t10k-images-idx3-ubyte',10000,1)
	test_labels_filename = extract_files('/Users/deepikachaudhary/Documents/Sem-1_UB/ML/proj3_images/mnist/t10k-labels-idx1-ubyte',10000,0)

	train_data_filename = inputData[1:55000,:]
	train_labels_filename = labelData[1:55000]
	val_data_filename = inputData[55000:60000,:]
	val_labels_filename =labelData[55000:60000]
	
	accuracy_arr = []
	
	final_accuracy = 0.1
	final_eta = 0.1
	final_lambda = 0.01

	# training
	isTraining = True
	noOfItr = 500

	trained_weight,accuracy,_ = process_input(train_data_filename,train_labels_filename,weight,isTraining,noOfItr,final_lambda,final_eta,0)

	#validate
	lambda_arr = [0.001,0.01,0.1]
	eta_arr = [0.01,0.03,0.09,0.3,0.9]
	final_eta,final_lambda = validate_hyperParams(lambda_arr,eta_arr,trained_weight,val_data_filename,val_labels_filename)
	print('Final eta is',final_eta)
	print('Final lambda is',final_lambda)

	#training 
	isTraining = True
	noOfItr = 500
	trained_weight,accuracy,accuracy_arr = process_input(train_data_filename,train_labels_filename,weight,isTraining,noOfItr,final_lambda,final_eta,0)
	print('Training accuracy is')
	print(accuracy)


	# # testing
	isTraining = False
	noOfItr = 1
	_,accuracy,_ = process_input(test_data_filename,test_labels_filename,trained_weight,isTraining,noOfItr,final_lambda,final_eta,0)
	print('Testing accuracy is')
	print(accuracy)

	#USPS DATA TESTING
	files = [f for f in glob.glob("/Users/deepikachaudhary/Documents/Sem-1_UB/ML/proj3_images/Numerals/*[0-9]")]

	isTraining = False
	noOfItr = 1
	usps_data,usps_labels_training = extract_usps_files(files,0)
	_,accuracy,_ = process_input(usps_data,usps_labels_training,trained_weight,isTraining,noOfItr,final_lambda,final_eta,1)

	print("USPS Training data accuracy is")
	print(accuracy)

	files_test = [f for f in glob.glob("/Users/deepikachaudhary/Documents/Sem-1_UB/ML/proj3_images/Test/")]
	usps_data_test,usps_labels_test = extract_usps_files(files_test,1)
	_,accuracy,_ = process_input(usps_data_test,usps_labels_test,trained_weight,isTraining,noOfItr,final_lambda,final_eta,1)

	print("USPS Testing data accuracy is")
	print(accuracy)

#######################################code for Logistic Regression ends here #######################################

####################################### code for Single layer neural network starts here #######################################

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

####################################### code for Single layer neural network ends here #######################################

####################################### code for CNN starts here #######################################
# normally distributed weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) 
    return tf.Variable(initial)

# random bias
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# generate convolutional layer 
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# generate pool layer
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# train CNN
def train_CNN():
	# extract mnist data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	# input
	x = tf.placeholder(tf.float32, [None, 784])
	# parameter
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))
	# softmax regression
	y = tf.nn.softmax(tf.matmul(x, W) + b)
	# cross entropy
	y_ = tf.placeholder(tf.float32, [None, 10])

	# initiate weights for layer 1, convolution layer 1 and pool layer 1
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])
	x_image = tf.reshape(x, [-1, 28, 28, 1])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) 
	h_pool1 = max_pool_2x2(h_conv1)

	# initiate weights for layer 2, convolution layer 2 and pool layer 2
	W_conv2 = weight_variable([5, 5, 32, 64]) 
	b_conv2 = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	# initiate weights for layer 3 and fully connected layer
	W_fc1 = weight_variable([7 * 7 * 64, 1024]) 
	b_fc1 = bias_variable([1024])
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	# specify drop value
	keep_prob = tf.placeholder(tf.float32) 
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	# weights for output layer
	W_fc2 = weight_variable([1024, 10]) 
	b_fc2 = bias_variable([10])
	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	# declare the function which should be minimised
	cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	# use optimiser for gradient descent
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) 
	# method for checking the prediction
	correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1)) 
	# method to calculate accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# start the session
	with tf.Session() as sess:
		# initiate global variables
	    sess.run(tf.global_variables_initializer())
	    # run epochs
	    for i in range(20000):
	    	# use minibatch
	        batch = mnist.train.next_batch(50) 
	        # print accuracy after every 100 iterations
	        if i % 100 == 0:
	            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
	            print('step %d, training accuracy %g' % (i, train_accuracy)) 
	        
	        # train the model
	        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob:0.5})
	        # print the test accuracy
	        print('test accuracy %g' % accuracy.eval(feed_dict={
	        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

####################################### code for CNN ends here #######################################
def main():
	# uncomment below method to run the CNN part
	train_CNN()
	# uncomment below method to run logistic regression
	LR()
	# uncomment below method to run SNN part
	SNN()
main()