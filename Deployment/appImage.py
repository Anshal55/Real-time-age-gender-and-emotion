# dependencies
from predictionImage import predictImage

import os
import numpy as np 

from flask import Flask,request,render_template

app = Flask(__name__)

IMAGE_FOLDER = os.getcwd() + "/static"

@app.route("/")
def home():
    return render_template("home.html",data="Hey!!")


@app.route("/predict",methods=["GET","POST"])
def predict():
    if request.method == "POST":
        img = request.files["img"]

        if img:
            img_loc = os.path.join(
                IMAGE_FOLDER,
                img.filename
            )
            img.save(img_loc)
    age,gender,emotion = predictImage(img_loc)

    return render_template("home.html",age = age,gender = gender,emotion = emotion,image_loc = img.filename)


if __name__ == "__main__":
    app.run(debug=True,port=20000)