# dependencies
import numpy as np
from keras.models import load_model

def makePredictionEmotion(image):
    classes = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
    model_emotion = load_model("Deployment/models/emotion.h5")
    img = np.expand_dims(image,axis=0)
    img = img/255.0
    pred = model_emotion.predict(img)
    emotion = classes[np.argmax(pred)]

    return emotion

def makePredictionAge(image):
    model_age = load_model("Deployment/models/age.h5")
    img = np.expand_dims(image, axis=0)
    img = img/255.0
    pred = model_age.predict(img)
    age = np.round(*pred[0])

    return age

def makePredictionGender(image):
    classes = ["Male","Female"]
    model_gender = load_model("Deployment/models/gender.h5")
    img = np.expand_dims(image, axis=0)
    img = img / 255.0
    pred = model_gender.predict(img)
    gender = classes[int(np.round(pred[0][0]))]

    return gender

def makePredictionEthnicity(image):
    classes = ["White","Black","Asian","Indian","Hispanic"]
    model_ethnicity = load_model("Deployment/models/ethnicity.h5")
    img = np.expand_dims(image, axis=0)
    img = img / 255.0
    pred = model_ethnicity.predict(img)
    ethnicity = classes[np.argmax(pred)]

    return ethnicity
