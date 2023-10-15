import joblib
import numpy as np
import pickle
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
model = pickle.load(open('templates/Model.pkl', 'rb'))


@app.route("/")
def HelloWorld():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    int_data = [int(x) for x in request.form.values()]
    data = [np.array(int_data)]
    pred = model.predict(data)
    if pred[0] == 1:
        return render_template(
            "index.html",
            prediction_text="The result is {} - DIABETES POSITIVE".format(pred))
    elif pred[0] == 0:
        return render_template(
            "index.html",
            prediction_text="The result is {} - DIABETES NEGATIVE".format(pred))


if __name__ == "__main__":
    app.run(debug=True)
    # print(predict())