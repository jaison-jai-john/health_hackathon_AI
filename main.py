# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
import io
import os

import numpy as np
import tensorflow.keras as K
from flask import Flask, request
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array

from predict import model, types

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)


# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route("/", methods=["POST", "GET"])
# ‘/’ URL is bound with hello_world() function.
def predict():
    # get image from request
    # resize image to 299x299
    img = request.files["image"]
    img = image.load_img(io.BytesIO(img.read()), target_size=(299, 299))
    x = img_to_array(img)
    x = K.applications.xception.preprocess_input(x)

    prediction = model.predict(np.array([x]))[0][0:6]

    return {
        "prediction": [
            {"class": types[i], "probability": float(prediction[i]) * 100.0}
            for i in range(len(prediction))
        ]
    }


def t():
    app.run()


if __name__ == "__main__":
    port = int(
        os.environ.get("PORT", 8000)
    )  # Use Render's assigned port or default to 8000
    app.run(host="0.0.0.0", port=port)
# test
