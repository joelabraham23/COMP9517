import os
import cv2
import glob

def generateNegFile():
    with open('neg.txt', 'w') as f:
        for filename in os.listdir('negative'):
            f.write('negative/' + filename + '\n')

def rotateFiles():
    folder = input("Enter folder name:")
    folder_name = folder + '/*.jpg'
    images = [cv2.imread(file) for file in glob.glob(folder_name)]
    i = 0
    for image in images:
        img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        name = folder + '/'+ str(i) + '_rotate.jpg'
        i+=1
        cv2.imwrite(name, img)

def greyscaleFiles():
    folder = input("Enter folder name:")
    folder_name = folder + '/*.jpg'
    images = [cv2.imread(file) for file in glob.glob(folder_name)]
    i = 0
    for image in images:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        name = folder + '/'+ str(i) + '_b&w.jpg'
        i+=1
        cv2.imwrite(name, img)
    

#image annotation program
# C:/Users/npetr/Documents/Coding/Personal/opencv/build/x64/vc15/bin/opencv_annotation.exe --annotations=pos.txt --images=positive/

# create vector files from pos.txt output to pos.vec
# C:/Users/npetr/Documents/Coding/Personal/opencv/build/x64/vc15/bin/opencv_createsamples.exe -info pos.txt -w 24 -h 24 -num 300 -vec pos.vec

# training from vector file
# C:/Users/npetr/Documents/Coding/Personal/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data cascade/ -vec pos.vec -bg neg.txt -w 24 -h 24 -numPos 240 -numNeg 120 -numStages 10

# C:/Users/npetr/Documents/Coding/Personal/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data cascade/ -vec pos.vec -bg neg.txt -w 24 -h 24 -ptrcalcValBufSize 6000 -precalcIdxBufSize 6000 -numPos 950 -numNeg 1900 -numStages 12 -maxFalseAlarmRate 0.25 -minHitRate 0.999