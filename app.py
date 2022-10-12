# flask, scikit-learn, pandas, pickle-mixin
import numpy as np
import pandas as pd
from flask import Flask, render_template, request          #Flask is python Framework used for rendering the ml projects, it works in the backend
import pickle

data= pd.read_csv("Cleaned_data.csv")

app = Flask(__name__)

models = pickle.load(open("Ridge_Model.pkl","rb"))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html',locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')
    #print(location, bhk, bath, sqft)
    #d={'location':[location],'total_sqft':[sqft], 'bath':[bath], 'bhk':[bhk]}
    inputs = pd.DataFrame([[location,sqft,bath,bhk]],columns=['location', 'total_sqft', 'bath', 'bhk'])
    #inputs = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    #inputs = pd.DataFrame(d)
    predictions = models.predict(inputs)[0]*100000

    return str(np.round_(predictions,2))
if __name__=="__main__":
    app.run(debug=True, port=5006)
