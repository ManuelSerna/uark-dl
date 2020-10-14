#*********************************
# CNN Model
'''
Input:
    num_classes: number of classes to categorize
Return:
    Keras implementation of LeNet CNN
'''
#*********************************
def get_lenet(num_classes):
    model = Sequential() # define model using sequential class
	# Start by adding a layer to the model, in this step, insert a convolutional layer.
    '''
    Arg 1: num filters, the more you have, the more computing power needed 
    Arg 2: filter size (of 5x5 for the 32x32 images)
    Arg 3: input will be fed an image that is 32x32 with 3 channels (RGB)
    Arg 4: activation function, use the ReLU function

    The image will be reduced to 30 feature maps, each 24x24.

    Padding works to preserve the spatial dimensionality of the image.
        * Same padding: making the output matrix size the same as the input.
            - Allows to extract low-level features.
    '''
    model.add(Conv2D(30, (5,5), input_shape=(28,28,1), activation='relu')) # use 30 filters of size 5x5

    # Add pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2))) # only 1 arg: size of pooling element

    # Add another convolutional layer. Use smaller filter to extract features.
    model.add(Conv2D(15, (3,3), activation='relu')) # have 4,065 parameters, use 15 filters of size 3x3

    # Add another pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2))) # produce a 5x5 image with a depth of 50

    # Take convoluted data and feed into the fully connected layer
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))

    '''
    Use a single dropout layer.
    Although more can be used, and in different places, they are used in between layers that have a high number of parameters, these are more likely to overfit.

    Arg 1: fraction rate, the amount of input nodes that the dropout layer drops during each update.
    0 = no nodes dropped.
    1 = all nodes dropped.
    RECOMMENDED = 0.5
    '''
    model.add(Dropout(0.5))

    # Define output layer
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    return model



# Test driver code
# TODO: must import appropriate data
'''
# Train and test lenet model
model1 = get_lenet(num_classes)
#print(model1.summary())
history = model.fit(x_train, y_train, epochs=10, validation_split=0.1, batch_size=64, verbose=1, shuffle=1)

# Get accuracy of model for current letter set
score = model.evaluate(x_test, y_test, verbose=0)
print(type(score))
print('Test accuracy for leNet:', score[1])
'''
