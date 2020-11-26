#!/usr/bin/env python3
"""
student.py
UNSW COMP9444 Neural Networks and Deep Learning
You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.
You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.
The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.
You may only use GloVe 6B word vectors as found in the torchtext package.
"""

"""
This code was written by Edward Li (z5162357) and Kevin Zhu (z5205927)

The neural network consists of an LSTM followed by three linear layers with ReLu as activation functions. There are then two seperate linear layers, which are used 
to handle rating classification and category classification. The use of a bidirectional LSTM layer was chosen over others due to the need to consider long term dependencies, as this
is a review classification task and there may be keywords towards the beginning/end of the review which are important. The bidirectional nature of the LSTM layer allowed
the network to preserve information from the beginning and ending of the review The linear layers were chosen to be the fully connected layers of this neural net - that 
is to map the outputs of the hidden LSTM and ReLu layers to the target dimensions of the rating and category classification output.

For the loss functions, we used BCEWithLogitsLoss to compute a loss for the rating, and CrossEntropyLoss for category. BCEWithLogitsLoss was chosen as it excels
for multi-binary classification, which in this case we are classifying as either a 0 or 1 for rating. It also applies a sigmoid activation internally making it more stable 
than using a sigmoid layer manually alongside BCE. For the category, we use CrossEntropyLoss which is more suited for classification problems with a number of classes. 
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
import torch.nn.functional as F
from torchtext.vocab import GloVe
# import numpy as np
# import sklearn
import re

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """

    processed = sample.split()

    return processed

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """

    #Removing punctuation, filtering out stop words
    output = []
    for i in range(len(sample)):
        if sample[i] not in stopWords:
            word = re.sub(r'[^a-z]', "", sample[i])
            output.append(word)
    
    return output

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch

stopWords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours',
             'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
             'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
             'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
             'while', 'of', 'at', 'by', 'for', 'with', 'about', 'between', 'into', 'through', 'during', 'before', 'after', 'to', 'from',
             'up', 'down', 'in', 'out', 'on', 'off', 'again', 'further', 'then', 'once', 'when', 'where', 'why', 'how', 'all',
             'any', 'both', 'each', 'few', 'other', 'some', 'such', 'no', 'nor', 'same', 'so', 'than', 'too', 's', 't',
             'can', 'will', 'just', 'don', 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
             'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'needn',
             'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
wordVectors = GloVe(name='6B', dim=300)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """

    #Convert the tensors into their expected values of 0/1 for rating, and 0,1,2,3,4 for business
    #based on whichever index has the highest value

    categoryOutput = categoryOutput.argmax(1).flatten()
    ratingOutput = ratingOutput.argmax(1).flatten()

    return ratingOutput, categoryOutput

################################################################################
###################### The following determines the model ######################
################################################################################

input_size = 300
sequence_size = 100
num_layers = 1
hidden_size = 128

# num_features = 2 #rating, business category


class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self):
        super(network, self).__init__()

        #Bidirectional LSTM Layer
        self.lstm = tnn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True, bidirectional=True)
        
        #Three linear layers with ReLu activations
        self.layer_one = tnn.Linear(num_layers*hidden_size*2, hidden_size*2)
        self.activation_one = tnn.ReLU()
        self.layer_two = tnn.Linear(hidden_size*2, hidden_size*2)
        self.activation_two = tnn.ReLU()
        self.layer_three = tnn.Linear(hidden_size*2, hidden_size)
        self.activation_three = tnn.ReLU()
        
        # self.fc = tnn.Linear(hidden_size, num_features)
        self.linearForRating = tnn.Linear(hidden_size, 2)
        self.linearForCategories = tnn.Linear(hidden_size, 5)
        
        self.LogSoftMax = tnn.LogSoftmax(1)

    def forward(self, input, length):

        #Initialising the hidden state to pass into LSTM
        h0 = torch.zeros(num_layers*2, len(length), hidden_size).to(device)
        c0 = torch.zeros(num_layers*2, len(length), hidden_size).to(device)
        output, (h_n, c_n) = self.lstm(input, (h0, c0))

        #Getting a slice of the hidden output
        output = output[:, -1, :]

        #Pass output through linear layers and activation functions
        output = self.layer_one(output)
        output = self.activation_one(output)
        output = self.layer_two(output)
        output = self.activation_two(output)
        output = self.layer_three(output)
        output = self.activation_three(output)

        #Pass output into two seperate linear layers for rating and category classification
        ratingOutput = self.linearForRating(output)
        categoryOutput = self.linearForCategories(output)
        
        return ratingOutput, categoryOutput


class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.BCEWithLogitsLoss = tnn.BCEWithLogitsLoss()
        self.CrossEntropyLoss = tnn.CrossEntropyLoss()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):

        #Convert rating target into one hot vector for input into BCEWithLogitsLoss
        ratingTarget = F.one_hot(ratingTarget).float()

        categoryLoss = self.CrossEntropyLoss(categoryOutput, categoryTarget)
        ratingLoss = self.BCEWithLogitsLoss(ratingOutput, ratingTarget)

        #Sum the two losses to compute overal loss
        final_loss = categoryLoss + ratingLoss
        return final_loss
    

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 1 
batchSize = 64 
epochs = 10 
#optimiser = toptim.SGD(net.parameters(), lr=0.01)
optimiser = torch.optim.Adam(net.parameters(),eps=0.000001,lr=0.01,
                             betas=(0.9,0.999),weight_decay=0.0001)