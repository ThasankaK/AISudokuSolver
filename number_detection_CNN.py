import cv2 as cv
import numpy as np
import os
from sklearn.model_selection import train_test_split 
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam

#----------------------------------------------------------------------#
# 1. Data Preparation
#----------------------------------------------------------------------#
images = []
labelNum = [] # list with labels of images

print("Extracting Data from Label #'s:", end=" ")
for num in range(10):
    picList = os.listdir("NumberImages" + '/' + str(num)) 

    for image in picList:
        curImage = cv.imread("NumberImages" + '/' + str(num) + '/' + image) 

        if curImage is not None:
            curImage = cv.resize(curImage, (32, 32)) 
            images.append(curImage)
            labelNum.append(num)
        else:
            print(f"Failed to load image: {image}")

    print(num, end = " ")

print(" ")

images = np.array(images)
labelNum = np.array(labelNum)

#----------------------------------------------------------------------#
# 2. Splitting the Data into Training/Testing/Validating
#----------------------------------------------------------------------#
print(f"Original Set of Data: {images.shape} \n")


X_train, X_test, Y_train, Y_test = train_test_split(images, labelNum, test_size=0.2) # Train 80%, Test 20%, function ensures data is randomized
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.2) # Train 64%, Test 20%, Validation 16%

print(f"Data to Train: {X_train.shape}")
print(f"Data to Test: {X_test.shape}")
print(f"Data to Validate: {X_validation.shape}")


# np.where(Y_train==x) outputs an np array, first element is an array with all numbers with respective label
numOfImages = []
for x in range(10):
    numOfImages.append(len(np.where(Y_train==x)[0]))
    print(f"How many [{x}] labels?: {len(np.where(Y_train==x)[0])}") 

#----------------------------------------------------------------------#
# 3. Preprocessing Images
#----------------------------------------------------------------------#
def Preprocessing(image):
    image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    image = cv.equalizeHist(image) 
    image = image/255
    return image
    

# map each value into function -> into a list -> into an np array 
X_train = np.array(list(map(Preprocessing, X_train)))
X_test = np.array(list(map(Preprocessing, X_test)))
X_validation = np.array(list(map(Preprocessing, X_validation)))

# Add a depth of 1 for the CNN, tells that only 1 color channel per image. If RGB, put 3

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

AugmentedData = ImageDataGenerator(width_shift_range=0.1, # %
                             height_shift_range=0.1, # %
                             zoom_range=0.2, # %
                             shear_range=0.1, # %
                             rotation_range=10) # °
AugmentedData.fit(X_train) # Generating augmented images as batches that get sent back to X_train while training

#-----------------------------------------------------------------------#
# 4. CNN
# ----------------------------------------------------------------------#


Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)
Y_validation = to_categorical(Y_validation, 10)

def CNN_Model():
    numOfFilters = 60
    filterSize1 = (5,5)
    filterSize2 = (3,3)
    poolSize = (2,2)
    numOfNode = 500

    model = Sequential() # Keras Sequential model

    model.add((Conv2D(numOfFilters, filterSize1, input_shape=(32, 32, 1), activation='relu'))) # 28x28x60
    model.add((Conv2D(numOfFilters, filterSize1, activation='relu'))) # 24x24x60
    model.add(MaxPooling2D(pool_size=poolSize)) # 12x12x60
    model.add((Conv2D(numOfFilters//2, filterSize2, activation='relu'))) # 21x21x30
    model.add((Conv2D(numOfFilters//2, filterSize2, activation='relu'))) # 18x18x30
    model.add(MaxPooling2D(pool_size=poolSize)) # 9x9x30
    model.add(Dropout(0.5)) # half the neurons are inactive, only used in training
    model.add(Flatten()) # 9x9x30 = 2430
    model.add(Dense(numOfNode, activation='relu')) 
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax')) # 10 nodes for 10 classes
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = CNN_Model()
print(model.summary())

history = model.fit_generator(AugmentedData.flow(X_train, Y_train,
                                 batch_size = 50), # number of training samples used in one iteration
                                 steps_per_epoch = len(X_train)//50, # total number of training samples / batch size
                                 epochs = 50, # one full cycle through training dataset
                                 validation_data = (X_validation, Y_validation),
                                 shuffle = 1)


model.save("model_trained.h5")
