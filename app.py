import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
##Load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    
    try:
        data = request.json['data']
        print(data)  # Log the received data
        input_array = np.array(list(data.values())).reshape(1, -1)
        new_data = scaler.transform(input_array)
        output = regmodel.predict(new_data)
        print("Model Prediction:", output[0])
        return jsonify({"prediction": output[0]}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred while processing the request."}), 400

if __name__ == "__main__":
    app.run(debug=True)


