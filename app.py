from cgi import test
from sklearn.preprocessing import scale
from url_to_data import *
import pandas as pd
from flask import Flask, render_template, request, url_for,redirect
import pickle
from scaler import *
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# model_dir = "./mnist_model"

scalerobj = scaler()

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/response", methods=('GET', 'POST'))
def solution():

    if request.method == 'POST':
        url = request.form['URL']
        if url == "":
            return render_template("index.html")

        obj = UrlFeaturizer(url)
        data_test = pd.DataFrame(obj.run(),index=[0])
        cols = data_test.columns.tolist()
        cols.sort()
        data_test = data_test[cols]
        print(data_test)
        data_test = scalerobj.scale(data_test)
        # localhost_save_option = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")

        test_data = pd.DataFrame(data_test)
        print(test_data)
        # pickled_model = pickle.load(open('model.pkl', 'rb'))
        # model2 = tf.keras.models.load_model(model_dir, options=localhost_save_option)
        model2 = load_model("model.h5")
        y = model2.predict(test_data)
        print(y)
    return render_template("response.html")