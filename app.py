from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)
@app.route("/")
def HelloWorld():
    return render_template('home.html')

@app.route("/predict", methods=['POST'])
def predict():
    categorical_mapping = {
        "Yes": 1,
        "No": 0,
        "Male": 1,
        "Female": 0,
    }

    data = request.json
    model = joblib.load('Random_Forest_Model.pkl')

    # Convert categorical data to numerical using the mapping
    numerical_data = {key: categorical_mapping.get(value, value) for key, value in data.items()}

    # Make a prediction
    prediction = model.predict([list(numerical_data.values())])

    return jsonify({"prediction": prediction[0]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8080)
