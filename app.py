import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from model_init import init, preprocess_img, my_label
from keras.models import model_from_json
import base64

# load json and create model
global model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
model = loaded_model
model._make_predict_function()
print("Model Loaded")


app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

def new_method():
    return render_template("index.html")

@app.route("/", methods=["GET"])
def index():
   return new_method()

@app.route("/", methods=["POST"])
def inference():
    if not request.files["img"]:
        return "No image loaded"

    for item in request.form:
        print(item)

    file = request.files["img"]
    content_type = file.content_type
    bytes = file.read()

    img = preprocess_img(file)
    result = model.predict(img, batch_size=None, steps=1)
    test = np.argmax(result, axis=1)
    test =np.array(test)
    my_img = my_label(test)
    my_img = my_img.capitalize()
    b64_string = base64.b64encode(bytes)
    b64_data = "data:" + content_type + ";base64," + str(b64_string)[2:-1]

    return render_template("result.html", result=my_img, image=b64_data)

if __name__ == '__main__':
   app.run()