from url_to_data import *
import pandas as pd
from flask import Flask, render_template, request, url_for,redirect
import pickle
app = Flask(__name__)

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
        data_test
        # print(data_test)
    return render_template("response.html")