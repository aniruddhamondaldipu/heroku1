import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd



df=pd.read_csv('Final_test_csv_with_predictions.csv')
dff=pd.read_csv('Accuracy.csv')

app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    output=dff['accuracy'].iat[0]
    if output < 100:
        output=100-output
    else:
        output=output-100
    output="{:.2f}".format(output)
    return render_template('index.html', prediction_text='{} %'.format(output))

@app.route('/table')
def table():

    data = df
    return render_template('table.html', tables=[data.to_html()], titles=[''])

if __name__ == "__main__":
    app.run(debug=True)










