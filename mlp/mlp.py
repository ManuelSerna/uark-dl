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
    #-----------------------------
    # Constructor
    '''
     Input: TODO: describe parameters (copy comments for attributes below and then delete them
    '''
    #-----------------------------
    def __init__(self, epochs, s, lr):
        self.W = [] # weights
        self.b = [] # biases
        self.epochs = epochs # how many times to train over data
        self.s = s  # list where each element represents the number of activations in each layer l
        self.lr = lr # learning rate of model
        
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
    # Fit data
    '''
     Input:
        x_in: n*2 numpy array of training points
        y_in: labels for the corresponding point x_in[i]
     MAYBE?: Return:
     MAYBE?:   model: Python dictionary containing the optimized W and b
    '''
    #-----------------------------
    def fit(self, x_in, y_in):
        #W, b = initialize(s)
        n = len(y_in)
        
        for epoch in range(self.epochs):
            print(' epoch {}'.format(epoch+1))
            for i in range(n):
                #print('traninig sample {}'.format(i))
                
                # Insert one training sample and label at a time
                x = np.zeros((2, 1)) # need column vector of data instance
                for j in range(2):
                    x[j, 0] = x_in[i][j]
                
                y = y_in[i]
                
                Z, A = self.forward_prop(x) # forward propagate on one instance of train data
                dW, db = self.back_prop(Z, A, x, y)
                self.update(dW, db)
        
        #model = {}
        #model['W'] = self.W
        #model['b'] = self.b
    
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
        x: single input (2*1 numpy array)
        y: single input label (1*1 numpy array)
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
        
        # Y will contain the desired labels of every output activation given a specific training
        '''
        # Y is just y for this mlp, maybe include this code again? (If so, change y to Y 13 lines down)
        Y = []
        for i in range(s[L]):
            if i == y:
                Y.append(1.0)
            else:
                Y.append(0.0)
        '''
        # Back-propagate from output layer
        for l in range(L-1, -1, -1):
            for i in range(len(dW[l])):
                # Pre-compute: partial der of activation func for current layer l
                dz = self.sigmoid(Z[l][i]) * (1 - self.sigmoid(Z[l][i]))
                
                # Compute: partial der of activation
                if l == L-1: # if starting at last layer
                    dA[l][i] = 2 * (A[l][i] - y)# NOTE: in old code: Y[i]
                else: # else at hidden layer l
                    for k in range(len(dW[l+1])): # get pre-computed values of activations of l+1
                        dA[l][i] += self.W[l+1][k][i] * self.sigmoid(Z[l+1][k]) * (1 - self.sigmoid(Z[l+1][k])) * dA[l+1][k]
                
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
     Return: predicted label (output of last layer)
    '''
    #-----------------------------
    def predict(self, x):
        Z, A = forward_prop(x)
        
        '''
        max_prob = -1.0
        max_index = -1
        
        for k in range(len(A[-1])):
            if max_prob < A[-1][k]:
                max_prob = A[-1][k]
                max_index = k
        '''
        # NOTE: A[0] should be single label 0 or 1
        return A[0]
    
    #-----------------------------
    # Evaluate model for accuracy
    '''
     Input:
        x_test: test data (n_test*2)
        y_test: test labels (n_test*1)
    '''
    #-----------------------------
    def evaluate(self, x_test, y_test):
        n = len(y_test)
        tot_correct = 0
        
        # Make predictions on test data
        for i in range(n):
            x = np.zeros((784, 1)) # need to make a 'column array'
            
            for j in range(784):
                x[j, 0] = x_test[i][j]
            
            y = self.predict(x)
            #print(y_test[i], y)# DEBUG

            if y_test[i] == y:
                tot_correct += 1
        
        # Print accuracy to console
        print('Accuracy: {}/{} = {}%'.format(tot_correct, n, tot_correct/n * 100))



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

# TODO: change squared diff to cross entropy loss
# TODO: calculate loss wrt to new function
# TODO: call model and test
model = mlp(iterations, s, lr)
model.fit(x_train, y_train)
print('everything seems ok if this shows')
