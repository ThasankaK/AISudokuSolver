import cv2 as cv
import numpy as np


# send in all contours found
def biggestContour(contours):
    biggest = np.array([])
    maxArea = 0

    for i in contours: # loop through all contours
        area = cv.contourArea(i) # find area of all contours 

        if area > 50: # contour can't be too small, or its just noise
            perim = cv.arcLength(i, True) # find contour perimeters
            approx = cv.approxPolyDP(i, 0.02 * perim, True) # approximates contour by removing vertices, but keeping shape
            # epsilon value allows control for level of simplificatoin. Within 2% of original contour
            # returns simplifed polygonal curve

            if area > maxArea and len(approx) == 4: # len(approx) == 4, lets us only find rectangles or squares
                biggest = approx
                maxArea = area

    return biggest, maxArea # biggest will contain corner points, maxArea contains area

def reorder(myPoints):
    myPoints = myPoints.reshape((4,2)) # reshapes array into 4 by 2
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32) # 4 x 1 array with 2 elements each row-column combination
    add = myPoints.sum(1)     # add points


    myPointsNew[0] = myPoints[np.argmin(add)] # lowest will be 0,0
    myPointsNew[3] = myPoints[np.argmax(add)] # add again, largest  value will be [width,height]
    diff = np.diff(myPoints, axis=1) # take 
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def splitBoxes(image):
    rows = np.vsplit(image, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)
    return boxes
# ARRANGES ALL IMAGES IN ONE WINDOW 
# work on this, could shorten it
def getPrediction(boxes, model):
    result = []
    for image in boxes:
        ## PREPARE IMAGE
        image = np.asarray(image)
        image = image[4:image.shape[0] - 4, 4:image.shape[1] - 4]
        image = cv.resize(image, (32, 32))
        image = image / 255
        image = image.reshape(1, 32, 32, 1)
        ## GET PREDICTION
        predictions = model.predict(image)
        classIndex = np.argmax(predictions, axis=-1)
        probabilityValue = np.amax(predictions)
        print(classIndex, probabilityValue)
        ## SAVE TO RESULT
        if probabilityValue > 0.2:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result

def displayNumbers(image, numbers, color = (255, 255, 255)):
    secW = int(image.shape[1]/9)
    secH = int(image.shape[0]/9)
    for x in range(0,9):
        for y in range(0,9):
            if numbers[(y*9)+x] != 0:
                cv.putText(image, str(numbers[(y*9)+x]),
                           (x*secW+int(secW/2)-10, int((y+0.8)*secH)), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           2, color, 2, cv.LINE_AA)
    return image

def drawGrid(img):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for i in range (0,9):
        pt1 = (0,secH*i)
        pt2 = (img.shape[1],secH*i)
        pt3 = (secW * i, 0)
        pt4 = (secW*i,img.shape[0])
        cv.line(img, pt1, pt2, (255, 255, 255),2)
        cv.line(img, pt3, pt4, (255, 255, 255),2)
    return img

def imageArrangement(imageArray, scale):
    rows = len(imageArray)
    cols = len(imageArray[0])

    height, width, channels = imageArray[0][0].shape # .shape returns its dimensions as a tuple
    # channels = 3, RGB

    resizedImages = []

    for r in range(rows):
        rowImages = []

        for c in range(cols):
            image = np.array(imageArray[r][c])  # convert image to np array
            if image is None or len(image) == 0:
                continue

            if image.ndim < channels:  # ndim -> number of dimensions. grayscale has 2, height and width. normal images 3, height, width, colors
            # this means its a grayscale (2 channel) -> convert to RGB (3 channels) BUT does not change the actual grayscale content?
            # all 3 color channels will have same values, maintaining grayscale, allows for all images in imageArray to have same dimension

                image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)  # Convert the image to RGB

            
            # add a if-statement for more than 3 channels?

        

            imageResized = cv.resize(image, (height*scale, width*scale))
            rowImages.append(imageResized)

        resizedImages.append(rowImages)

    rowArranged = [np.hstack(rowImages) for rowImages in resizedImages]
    arrangedImages = np.vstack(rowArranged)

    return arrangedImages
