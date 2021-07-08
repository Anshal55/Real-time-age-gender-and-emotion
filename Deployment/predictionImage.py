# GETTING HELPER FUNCTION AND LIBRARIES
from prediction_helper import *
import cv2 as cv


def predictImage(path):

    img = cv.imread(path)

    if img.shape[0] > 1080 and img.shape[1] > 1920:
        img  = cv.resize(img,(img.shape[1]//3,img.shape[0]//3))

    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    # face tracker
    face_tracker = cv.CascadeClassifier("xmls/haarcascade_frontalface_default.xml")

    # faces
    faces = face_tracker.detectMultiScale(gray,1.2,5)


    # getting and predicting on faces
    for (x,y,w,h) in faces:
        cropped_image = gray[y:y+h,x:x+w]
        cropped_image = cv.resize(cropped_image, (48, 48))

        # predictions
        age = makePredictionAge(cropped_image)
        gender = makePredictionGender(cropped_image)
        emotion = makePredictionEmotion(cropped_image)
        #ethnicity = makePredictionEthnicity(cropped_image)

    
    return age,gender,emotion
        





