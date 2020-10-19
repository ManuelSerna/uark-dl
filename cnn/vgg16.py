#---------------------------------
# VGG-16 CNN
# Image input shape: 224x224x3 (RGB images)
#---------------------------------
def get_vgg16(num_classes):
    model = Sequential()

    # Conv layer 1
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

    # Conv layer 2
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

    # Conv layer 3
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

    # Conv layer 4
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

    # Conv layer 5
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

    # Fully-connected layer
    model.add(Flatten()) # feed (flattened) features into fully-connected layer
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))

    # Output
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model



# Test driver code
'''
# Train and test VGG-16 model
vgg16 = get_vgg16(num_classes);
print(vgg16.summary())
history = vgg16.fit(x_train, y_train, epochs=10, validation_split=0.1, batch_size=64, verbose=1, shuffle=1)
score = vgg16.evaluate(x_test, y_test, verbose=0)

print('Test accuracy:', score[1])
'''
