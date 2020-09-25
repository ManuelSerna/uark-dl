#*********************************
'''
    MLP 
     Objective: Build MLP with one hidden layer to classify data within or outside a unit circle centered at the origin (0,0).

    Author: Manuel Serna-Aguilera
'''
#*********************************

import numpy as np
import matplotlib.pyplot as plt



#=================================
# MLP Class
# NOTE TODO: adapt the code I already made for the mnist neural network in my ml-notes repo
#=================================
class mlp():
    def __init__(self):
        # TODO: init params W and b here
    
    #-----------------------------
    '''
    Fit Data
     Input:
        x: n*2 numpy array of training points
        y: labels for the corresponding point x[i]
    '''
    #-----------------------------
    def fit(self, x, y):
        # call forward pass, backward pass, and update methods here
        
    def forward_prop(self):
        # TODO
    
    def back_prop(self):
        # TODO
    
    def update(self):
        # TODO



#---------------------------------
'''
    Function to generate data
     Input:   n (points to generate)
     Returns:
        x: is a n*2 numpy array of randomly generated points
        y: labels for the corresponding point x[i]
'''
#---------------------------------
def sample_points(n):
    radius = np.random.uniform(low=0, high=2, size=n).reshape(-1, 1) # get distances for points from origin
    angle = np.random.uniform(low=0, high=2*np.pi, size=n).reshape(-1, 1) # get angles of points in circle
    
    # Generate points
    x1 = radius*np.cos(angle)
    x2 = radius*np.sin(angle)
    
    y = (radius < 1).astype(int).reshape(-1) # assign labels
    x = np.concatenate([x1, x2], axis=1)
    return x,y

#---------------------------------
'''
    Plot set of points
    Input:
        x: n*2 numpy array of points
        y: labels for corresponding x
    Returns: NA
'''
#---------------------------------
def plot_data(x, y):
    for i in range(len(x)):
        if y[i] == 0:
            plt.plot(x[i, 0], x[i, 1], 'bo')
        else:
            plt.plot(x[i, 0], x[i, 1], 'ro')
    plt.show()



# Generate training data
n_train = 10000
x_train, y_train = sample_points(n_train)
plot_data(x_train, y_train)

# Generate validation data
n_val = 2000
x_val, y_val = sample_points(n_val)

# Generate testing data
n_test = 2000
x_test, y_test = sample_points(n_test)

# Other setup
lr = 0.01 # pre-determined learning rate

'''
TODO: build MLP with specifications
    - input layer: two neurons for x[i, 0] and x[i, 1]
    - hidden layer: TODO: figure out how many linear classifiers I want to use to encapsulate the circular region. Consider
        > 6, so a hexagon-shaped region, makes it simple, start with this first maybe
        > 8, maybe a step up if time allows
    - output layer: one activation (NOTE: use different activation?)

TODO: write network on paper first where each perceptron computes: Sigma(w[i]*x[i]) + bias => 0

NOTE: use cross-entropy loss
NOTE: activation function: ReLU, which is just max(x,0)
'''
