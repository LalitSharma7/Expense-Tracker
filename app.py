import numpy as np
from flask import Flask, request, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
mod = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = mod.predict(features)
    output = round(prediction[0], 2)
    return render_template("index.html", prediction_text =  "The car price is {}".format(output))

if __name__ == "__main__":
    flask_app.run(debug=True)