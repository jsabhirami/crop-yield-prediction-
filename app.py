from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the pre-trained model using pickle
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

Item_mapping = {
    'Cassava': 0, 'Maize': 1, 'Plantains and others': 2, 'Potatoes': 3,
    'Rice, paddy': 4, 'Sorghum': 5, 'Soybeans': 6, 'Sweet potatoes': 7,
    'Wheat': 8, 'Yams': 9
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result',methods=['POST'])
def result():
    return render_template('result.html')

@app.route('/submit', methods=['POST'])
def submit():
       Item = request.form['Item']
       Year= int(request.form['Year'])
       average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
       pesticides_tonnes = float(request.form['pesticides_tonnes'])
       avg_temp=float(request.form['avg_temp'])
       item_encoded = Item_mapping.get(Item)

       prediction_input = ([[item_encoded,Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp]])

       prediction_result = model.predict(prediction_input)

       return render_template('result.html',prediction_text = prediction_result)


if __name__ == "__main__":
    app.run(port=8000)