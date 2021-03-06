from prediction_helper import *
import cv2 as cv

# face tracker
face_tracker = cv.CascadeClassifier("xmls/haarcascade_frontalface_default.xml")



class VideoCamera(object):
    def __init__(self):
        self.video = cv.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        _,frame = self.video.read()

        
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        
        
        
        # faces
        faces = face_tracker.detectMultiScale(gray, 1.2, 5)

        # faces
        faces = face_tracker.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            cropped_image = gray[y:y + h, x:x + w]
            cropped_image = cv.resize(cropped_image, (48, 48))

            # predictions
            age = makePredictionAge(cropped_image)
            gender = makePredictionGender(cropped_image)
            emotion = makePredictionEmotion(cropped_image)
            ethnicity = makePredictionEthnicity(cropped_image)



            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv.putText(frame, f"Approximate age is {age}", (10, 20), cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0))
            cv.putText(frame, f"Gender is {gender}", (10, 45), cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0))
            cv.putText(frame, f"Emotion is {emotion}", (10, 70), cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0))
            cv.putText(frame, f"Ethnicity hoepfully is {ethnicity}", (10,95), cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0))



        _,jpeg = cv.imencode(".jpg",frame)
        return jpeg.tobytes()