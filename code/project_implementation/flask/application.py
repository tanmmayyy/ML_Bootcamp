import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

#import ridge regressor and standard scaler pickle
ridge_model = pickle.load(open('models/ridge.pkl', "rb"))
standard_scaler = pickle.load(open('models/scaler.pkl', "rb"))




@app.route("/")
def index():
    return render_template("index.html")






@app.route("/predict_data", methods = ["POST","GET"])
def predict_data_point():
    if request.method == "POST":
        Temprature = float (request.form.get("Temprature"))
        RH = float (request.form.get("RH"))
        WS = float (request.form.get("WS"))
        Rain = float (request.form.get("Rain"))
        FFMC = float (request.form.get("FFMC"))
        DMC = float (request.form.get("DMC"))
        ISI = float (request.form.get("ISI"))
        Classes = float (request.form.get("Classes"))
        Region = float (request.form.get("Region"))

        new_data_scaled = standard_scaler.transform ([[Temprature, RH, WS, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_data_scaled)

        return render_template("home.html", results = result[0])





    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0")