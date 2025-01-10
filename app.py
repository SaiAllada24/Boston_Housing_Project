import pickle
import json
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
##Load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))


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
    

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x)for x in request.form.values()]
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template('home.html',prediction_text="The final house price prediction is:{}".format(output))



if __name__ == "__main__":
    app.run(debug=True)


