import cv2 as cv
import numpy as np
import SudokuMath
from utils import * 
from tensorflow.keras.models import load_model
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

#---------------------------------------------------#
imagePath = "images/image1.png"
height = 450
width = 450
model = load_model("model_trained.h5") # My CNN model
#---------------------------------------------------#

#------------------------------------------------#
# 1. Image Processing
#------------------------------------------------#


imageBlank = np.zeros((height, width, 3), np.uint8)
image = cv.imread(imagePath)
image = cv.resize(image, (height, width))

imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
imageBlur = cv.GaussianBlur(imageGray, (5, 5), 1)
imageThreshold = cv.adaptiveThreshold(imageBlur, 255, 1, 1, 11, 2)


#------------------------------------------------#
# 2. Finding Contours/Finding Puzzle and Numbers
#------------------------------------------------#


imageContours = image.copy() 
imageBigContour = image.copy()

contours, hierarchy = cv.findContours(imageThreshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

cv.drawContours(imageContours, contours, -1, (0, 0, 255), 2)


biggest, maxArea = biggestContour(contours)
print(biggest)
if biggest.size != 0:
    biggest = reorder(biggest) 
    print(biggest)

    cv.drawContours(imageBigContour, biggest, -1, (0, 0, 255), 10)

    pts1 = np.float32(biggest) #preparing for warp perspective
    pts2 = np.float32([[0,0], [width, 0], [0, height], [width, height]]) # this will be the order of the points. (top left, top right, bottom left, bottom right)
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    imageWarpColored = cv.warpPerspective(image, matrix, (width, height))
    imageDetectedDigits = imageBlank.copy()
    imageWarpColored = cv.cvtColor(imageWarpColored, cv.COLOR_BGR2GRAY)


    #--------------------------------------------#
    # 3. Finding Each Digit
    #--------------------------------------------#


    imageSolvedDigits = imageBlank.copy()
    boxes = splitBoxes(imageWarpColored)



    #cv.imshow("Sample", boxes[0])
    numbers = getPrediction(boxes, model)
    print(numbers)
    imageDetectedDigits = displayNumbers(imageDetectedDigits, numbers, color=(255,255,255))
    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0, 0, 1)


    #---------------------------------------------#
    # 4. Solution of the Sudoku Puzzle
    #---------------------------------------------#


    board = np.array_split(numbers, 9)

    try:
        SudokuMath.solve(board)
    except:
        pass

    flatList = []
    for sublist in board:
        for item in sublist:
            flatList.append(item)
    solvedNumbers = flatList*posArray
    imageSolvedDigits = displayNumbers(imageSolvedDigits, solvedNumbers, color=(0, 255, 0))


    #---------------------------------------------#
    # 5. Displaying the Progess and Final Solution
    #---------------------------------------------#


    pts2 = np.float32(biggest) # prepare poitns for warp
    pts1 = np.float32([[0,0], [width, 0], [0, height], [width, height]])
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    imgInvWarpColored = image.copy()
    imgInvWarpColored = cv.warpPerspective(imageSolvedDigits, matrix, (width, height))
    inv_perspective = cv.addWeighted(imgInvWarpColored, 1, image, 0.5, 1)
    imageDetectedDigits = drawGrid(imageDetectedDigits)
    imageSolvedDigits = drawGrid(imageSolvedDigits)

    imageArray = [[image, imageThreshold, imageContours, imageBigContour,],  # R1
                [imageWarpColored, imageDetectedDigits, imageSolvedDigits, inv_perspective]]  # R2


    arrangedImages = imageArrangement(imageArray, 1)
    cv.imshow('Arranged Images', arrangedImages)

cv.waitKey(0)