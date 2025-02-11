# -*- coding: utf-8 -*-
"""layer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1On0IB61chz87hRfUHxBz9G_BSMZEFHVp
"""

#*********************************
# 1-Dimensional Convolutional Layer Class
# Author: Manuel Serna-Aguilera
#*********************************
import numpy as np

class Conv1D():
    #---------------------------------
    # Constructor
    '''
    NOTE: descriptions of params taken from Pytorch
        in_channel: number of channels in input image
        out_channel: number of channels produced by convolution
        kernel_size: size of convolving kernel
        stride: stride of convolution (default=1)
    '''
    #---------------------------------
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, W_init=None, b_init=None):
        # Define input params
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        # Store layer input, its dims will be set when forward is called
        self.x = None

        # Initialize weights and biases
        self.W = np.random.randn(out_channel, in_channel, kernel_size)
        self.b = np.zeros(out_channel)

        # Zero-initialize gradients for W and b
        self.dW, self.db = np.zeros(self.W.shape), np.zeros(self.b.shape)

        if W_init: self.W = W_init
        if b_init: self.b = b_init
    
    #-----------------------------
    # Call method--call forward
    '''
    Input:
        x: input for the convolutional layer of shape (batch_size, in_channel, in_width)
    Return:
        forward: output y of convolutional layer (batch_size, out_channel, out_width)
    '''
    #-----------------------------
    def __call__(self, x):
        return self.forward(x)
    
    #---------------------------------
    # Forward Pass on convolutional layer
    '''
    Adapted from 2d cnn forward in DL lec 13 slides (slide #43)

    * NOTE 1: 'in width' is an abritary (+) int,
       while 'out_width' is determined by in_width, kernel size, and stride.
    * NOTE 2: outputs of indv filters/kernels are are "channels"
    * NOTE 3: the output of this method will immediately be passed on to an activation function

    Input:
        x: (np array) input for the convolutional layer of shape (batch_size, in_channel, in_width)
          Where...
            batch_size: first dim refers to the # of samples we have
            in_channel: second dim refers to channels in each sample (like RGB channels)
            in_width: the length/width of the sample
    Return:
        y: output of convolutional layer (batch_size, out_channel, out_width), which is just the weighted sum term Z
    '''
    #---------------------------------
    def forward(self, x):
        batch_size = np.shape(x)[0] # get number of samples in batch
        in_width = np.shape(x)[2] # get width of input signal
        out_width = int((in_width-self.kernel_size)/self.stride)+1 # compute width of output feature map
        y = np.zeros((batch_size, self.out_channel, out_width)) # output np array y

        # Store layer input so we can calc dW in backprop
        self.x = x

        # Convolve 2D signal via batch sample, then output channel length, then input channel length, and then stride through input signal width/length
        for sample in range(0, batch_size):
            for j in range(0, self.out_channel):
                for i in range(0, self.in_channel):
                    for w in range(0, in_width-self.kernel_size, self.stride):
                        segment = x[sample, i, w:(w+self.kernel_size)]
                        y[sample][j][int(w/self.stride)] += self.W[j][i].dot(segment) + self.b[j]
        return y

    #---------------------------------
    # Backprop
    '''
    Input:
        delta: derivative of loss wrt output of current layer
            Where shape is (batch_size, out_channel, out_width) same shape as output of forward
             batch_size:  # of samples
             out_channel: channels of each (output) sample
             out_width:   width of output
    Return:
        dx: derivative of loss wrt input
            Where shape is (batch_size, in_channel, in_width) same as layer input shape
             batch_size: # of samples
             in_channel: channels in input sample
             in_width:   width/length of 1D input to layer
    '''
    #---------------------------------
    def backward(self, delta):
        batch_size = delta.shape[0]
        # in_channel given by object instance
        out_width = delta.shape[2]
        in_width = self.stride*(out_width-1) + self.kernel_size + 1 # reverse eq for getting output len

        # Init derivative of loss wrt layer input
        dx = np.zeros((batch_size, self.in_channel, in_width))
        
        # Backpropagate and update dW, db, and return derivative of loss wrt input
        for sample in range(0, batch_size):
            for j in range(0, self.out_channel):
                for i in range(0, self.in_channel):
                    w_out = 0 # keep counter for getting delta values
                    weight = self.W[j][i][:]
                    for w_in in range(0, in_width, self.stride):
                        if w_in+self.kernel_size < in_width:
                            # Update dx
                            d = delta[sample][j][w_out]
                            dx[sample][i][w_in:w_in+self.kernel_size] += d*weight
                            
                            # Update dW
                            self.dW[j,i,:] += d*self.x[sample, i, w_in:w_in+self.kernel_size]

                            # Move on to next index from output
                            w_out += 1
                self.db[j] = np.sum(delta[:,j,:]) # db = sum of values of out channel
        return dx

# Test my conv1D implementation against pytorch.nn.Conv1d
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

## initialize your layer and PyTorch layer
net1 = Conv1D(8, 12, 3, 2) ## Your own convolution
net2 = torch.nn.Conv1d(8, 12, 3, 2) ## generated by torch

## initialize the inputs
x1 = np.random.rand(3, 8, 20) # my input
x2 = Variable(torch.tensor(x1),requires_grad=True) # pytorch's input

## Copy the parameters from the Conv1D class to PyTorch layer
net2.weight = nn.Parameter(torch.tensor(net1.W))
net2.bias = nn.Parameter(torch.tensor(net1.b))

## Your forward and backward – Network 1
y1 = net1(x1) ## Call your forward
#print(y1) # print my result of forward--DONE

b, c, w = y1.shape
delta = np.random.randn(b,c,w)
dx = net1.backward(delta) # Call your backward
#print(dx) # print my result of backward--DONE

## PyTorch forward and backward – Network 2
y2 = net2(x2) ## Call pytorch's forward

#print(y2) # print pytorch's result of forward--DONE
delta = torch.tensor(delta)
y2.backward(delta) # Call pytorch's backward
#print(x2.grad)--# print dx for pytorch--DONE

# Compare gradients for weights
#print(net1.dW)# mine--DONE
#print(net2.weight.grad)# pytorch--DONE

# Compare gradients of bias
#print(net1.db)# mine--DONE
#print(net2.bias.grad)# pytorch--DONE