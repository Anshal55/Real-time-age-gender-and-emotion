# GETTING HELPER FUNCTION AND LIBRARIES
from prediction_helper import *
import cv2 as cv

# reading image
path = input("Enter the path of the image : ")
path  = f"{path}"
img = cv.imread(path)

if img.shape[0] > 1080 and img.shape[1] > 1920:
    img  = cv.resize(img,(img.shape[1]//3,img.shape[0]//3))

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# face tracker
face_tracker = cv.CascadeClassifier("xmls/haarcascade_frontalface_default.xml")

# faces
faces = face_tracker.detectMultiScale(gray,1.2,5)

print(f"faces found {faces}")

predictions = []

# getting and predicting on faces
for (x,y,w,h) in faces:
    cropped_image = gray[y:y+h,x:x+w]
    cropped_image = cv.resize(cropped_image, (48, 48))

    # predictions
    age = makePredictionAge(cropped_image)
    gender = makePredictionGender(cropped_image)
    emotion = makePredictionEmotion(cropped_image)
    #ethnicity = makePredictionEthnicity(cropped_image)

    predictions.extend([age,gender,emotion])

    cv.rectangle(img, (x,y), (x + w, y + h), (255, 255, 255))
    cv.putText(img,f"Approximate age is {age}",(x,y-20),cv.FONT_HERSHEY_COMPLEX,0.7,(255,0,0))
    cv.putText(img, f"Gender is {gender}", (x, y+5), cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0))
    cv.putText(img, f"Emotion is {emotion}", (x, y+30), cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0))
    #cv.putText(img, f"Ethnicity hoepfully is {ethnicity}", (10, y+55), cv.FONT_HERSHEY_COMPLEX, 0.7, (255,0, 0))

cv.imshow("Press q to quit",img)
if cv.waitKey(0) & 0xFF == ord("q"):
    cv.destroyAllWindows()

print(f"Predictions made are : {predictions}")



