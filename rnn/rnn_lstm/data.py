#*********************************
# Custom file to import MNIST data from subdirectory 'data'
# Author: Manuel Serna-Aguilera
#*********************************

# Imports
import idx2numpy # used for extracting mnist data
import matplotlib.pyplot as plt

'''
NOTE: To reshape image into col vector:
e.g.

    #...

    # Testing: reshaping
    n = 28
    img = x_train[0].reshape(-1)    
    img = img.reshape(n,n)
    plt.imshow(img)
    plt.show()
'''

def get_mnist():
    # Define file names
    file_train_x = 'data/train-images-idx3-ubyte'
    file_train_y = 'data/train-labels-idx1-ubyte'
    file_test_x = 'data/t10k-images-idx3-ubyte'
    file_test_y = 'data/t10k-labels-idx1-ubyte'

    # Extract data into arrays
    # NOTE 1: All samples are already shuffled
    # NOTE 2: All image samples are numpy arrays of shape (28,28)
    x_train = idx2numpy.convert_from_file(file_train_x)
    y_train = idx2numpy.convert_from_file(file_train_y)
    x_test = idx2numpy.convert_from_file(file_test_x)
    y_test = idx2numpy.convert_from_file(file_test_y)
    
    return x_train, y_train, x_test, y_test
