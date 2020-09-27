#*********************************
'''
MLP 
    Objective: Build MLP with one hidden layer to classify data within or outside a unit circle centered at the origin (0,0).

Author: Manuel Serna-Aguilera

TASKS:
 DONE: Create MLP--adapted from MNIST NN
 DONE: Use cross-entropy loss in back prop
 DONE: Report training loss
 TODO, done if correct: Report validation loss (NOTE: DOUBLE CHECK--do I just iterate over the validation set and only compute the cross entropy loss or do I have to adjust the hyperparameters as well, for this assignment?)
 DONE: Report testing accuracy via a %
 TODO: plot testing accuracy in matplotlib fig
 TODO: train mlp with 10 iterations and output train and val loss, and test acc
 TODO: train mlp with 100 iterations and output train and val loss, and test acc
 TODO: train mlp with 1000 iterations and output train and val loss, and test acc
'''
#*********************************

import numpy as np
import matplotlib.pyplot as plt



#=================================
# MLP Class
#=================================
class mlp():
    #-----------------------------
    # Constructor
    '''
     Input:
        epochs: how many iterations to train on
        s: list where each element represents the number of activations in each layer l
        lr: learning rate
    '''
    #-----------------------------
    def __init__(self, epochs, s, lr):
        self.W = [] # weights
        self.b = [] # biases
        self.epochs = epochs
        self.s = s
        self.lr = lr
        
        # Initialize parameters
        for l in range(len(self.s)-1):
            self.W.append(np.random.normal(0.0, 1.0, (self.s[l+1], self.s[l])))
            self.b.append(np.random.normal(0.0, 1.0, (self.s[l+1], 1)))
    
    #-----------------------------
    # Sigmoid function
    #  Input:  weigted sum matrix z
    #  Return: function result
    #-----------------------------
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    #-----------------------------
    # Derivative of sigmoid
    #  Input: weighted sum matrix z
    #  Return: derivative function result
    #-----------------------------
    def d_sigmoid(self, z):
        return self.sigmoid(z) * (1-self.sigmoid(z))
    
    #-----------------------------
    # Cost function: Cross entropy loss
    '''
     Input:
        a: network output (single number)
        y: true/desired output (single number)
     Return: function result
     NOTE: numpy.log is natural log
    '''
    #-----------------------------
    def cross_entropy(self, a, y):
        return -y*np.log(a) - (1-y)*np.log(1-a)
    
    #-----------------------------
    # Derivative of cross entropy loss
    '''
     Input:
        a: network output (single number)
        y: true/desired output (single number)
     Return: derivative function result
    '''
    #-----------------------------
    def d_cross_entropy(self, a, y):
        np.seterr(divide='ignore', invalid='ignore')
        return -1*(y/a) + ((1-y)/(1-a))
    
    #-----------------------------
    # Fit data and report training and validation loss
    '''
     Input:
        x_train: n_train*2 numpy array of training points
        y_train: labels for the corresponding point x_train[i]
        x_val: n_val*2 numpy array of training points
        y_val: labels for the corresponding point x_val[i]
     Return: NA
    '''
    #-----------------------------
    def fit(self, x_train, y_train, x_val, y_val):
        print('Training ({} epochs)...'.format(self.epochs))
        l_train = 0 # training loss
        l_val = 0 # validation loss
        n_train = len(y_train) # number of training samples
        n_val = len(y_val) # number of validation samples
        
        for epoch in range(self.epochs):
            print(' epoch {}'.format(epoch+1)) # DEBUG msg
            for i in range(n_train):
                #print('traninig sample {}'.format(i)) # DEBUG msg
                
                # Insert one training sample and label at a time
                x = np.zeros((2, 1)) # need column vector of data instance
                for j in range(2):
                    x[j, 0] = x_train[i][j]
                y = y_train[i] # single true label
                
                Z, A = self.forward_prop(x) # forward propagate on one instance of train data
                dW, db = self.back_prop(Z, A, x, y)
                self.update(dW, db)
                
                # Add loss for sample i
                l_train += self.cross_entropy(A[-1][0], y) # pass last val in A (single activation, so this is fine)
            
            # Compute: training loss after epoch
            l_train = (1/n_train)*l_train
            print('  Training Loss: {}'.format(l_train))
            
            # Compute: validation loss after epoch on entire validation set
            for l in range(n_val):
                # Insert one training sample and label at a time
                x = np.zeros((2, 1)) # need column vector of data instance
                for j in range(2):
                    x[j, 0] = x_val[l][j]
                y = y_val[l] # single true label
                
                Z, A = self.forward_prop(x) # forward propagate on one instance of val data
                l_val += self.cross_entropy(A[-1][0], y)
            
            l_val = (1/n_val)*l_val
            print('  Validation Loss: {}'.format(l_val))
    
    #-----------------------------
    # Forward propagate
    '''
     Input:
        x: list (column vector) of single instance of data (e.g. one (x,y) point or one image)
     Return:
        Z: list of weighted sums
        A: list of activations
    '''
    #-----------------------------
    def forward_prop(self, x):
        Z = []
        A = []
        
        n = len(self.s)
        
        for l in range(n-1):
            if l == 0: # if at first layer, use inputs x
                z = self.W[0].dot(x) + self.b[0]
                a = self.sigmoid(z)
            else:
                z = self.W[l].dot(A[l-1]) + self.b[l]
                a = self.sigmoid(z)
            Z.append(z)
            A.append(a)
        
        return Z, A
    
    #-----------------------------
    # Back propagate
    '''
     Input:
        Z: list of weighted sums
        A: list of activations
        x: single training input (2*1 numpy array)
        y: single training label for input point (number)
     Return:
        dW: calculated change for W
        db: calculated change for b
    '''
    #-----------------------------
    def back_prop(self, Z, A, x, y):
        L = len(self.s) - 1 # theoretically the index of the last layer
        
        # Initialize gradient lists
        dW = []
        db = []
        dA = []
        
        # Zero-init gradient storage
        for l in range(L):
            dW.append(np.zeros(self.W[l].shape))
            db.append(np.zeros(self.b[l].shape))
            dA.append(np.zeros(A[l].shape))
        
        # Back-propagate from output layer
        for l in range(L-1, -1, -1):
            for i in range(len(dW[l])):
                # Pre-compute: partial der of activation func for current layer l
                dz = self.d_sigmoid(Z[l][i])
                
                # Compute: partial der of activation
                if l == L-1: # if starting at last layer, use der of cost function
                    dA[l][i] = self.d_cross_entropy(A[l][i], y) # get cross entropy val with network output at A[l=2][i=0]
                else: # else at hidden layer l
                    for k in range(len(dW[l+1])): # get pre-computed values of activations of l+1
                        dA[l][i] += self.W[l+1][k][i] * self.d_sigmoid(Z[l+1][k]) * dA[l+1][k]
                
                # Compute: partial der for bias
                db[l][i] = dz * dA[l][i]
                
                # Compute: partial der for weights
                for j in range(len(dW[l][i])):
                    a = 0
                    if l == 0:
                        a = x[j]
                    else:
                        a = A[l-1][j]
                    dW[l][i][j] = a * dz * dA[l][i]
        return dW, db
    
    #---------------------------------
    # Update weights and biases
    '''
     Input:
        dW: calculated change for W
        db: calculated change for b
     Return: NA
    '''
    #---------------------------------
    def update(self, dW, db):
        layers = len(self.W)
        
        for l in range(layers):
            self.W[l] = self.W[l] - self.lr*dW[l]
            self.b[l] = self.b[l] - self.lr*db[l]
    
    #-----------------------------
    # Predict label on test input
    '''
     Input: x: single input (2*1) array
     Return: predicted label y (output of last layer)
    '''
    #-----------------------------
    def predict(self, x):
        # Get network output
        Z, A = self.forward_prop(x)
        y = A[-1]
        
        # Make output binary for comparison with labels
        if y >= 0.5:
            y = 1
        else:
            y = 0
        
        return y
    
    #-----------------------------
    # Evaluate model for accuracy
    '''
     Input:
        x_test: test data (n_test*2)
        y_test: test labels (n_test*1)
    '''
    # TODO: plot figure in this method after eval along with printing accuracy
    #-----------------------------
    def evaluate(self, x_test, y_test):
        print('Evaluating...')
        d = 2 # dimensionality of data x
        n = len(y_test)
        tot_correct = 0
        
        # Make predictions on test data
        for i in range(n):
            x = np.zeros((d, 1)) # need to make a 'column numpy array' for one data sample
            
            for j in range(d):
                x[j, 0] = x_test[i][j] # copy one data sample
            
            y = self.predict(x)
            #print(y_test[i], y)# DEBUG msg
            
            if y_test[i] == y:
                tot_correct += 1
        
        # Print accuracy to console
        print('  Testing Accuracy: {}/{} = {}%'.format(tot_correct, n, tot_correct/n * 100))



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
#plot_data(x_train, y_train)

# Generate validation data
n_val = 2000
x_val, y_val = sample_points(n_val)

# Generate testing data
n_test = 2000
x_test, y_test = sample_points(n_test)

# Other setup
iterations = 10 # a.k.a. epochs
s = [2, 6, 1] # encapsulate circle with s[1]-sided shape
lr = 0.01 # pre-determined learning rate

# Fit model and output results
model = mlp(iterations, s, lr)
model.fit(x_train, y_train, x_val, y_val)
model.evaluate(x_test, y_test)
