
import pandas as pd
from flask import Flask, render_template,request
import pickle

app = Flask(__name__)
data = pd.read_csv('Clean_Data.csv')
pipe = pickle.load(open('Banaglore_house_price_model.pkl','rb'))

@app.route('/')
def index():
    loactions =sorted(data['location'].unique())
    return render_template('index.html',locations = loactions)

@app.route('/predict',methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    print(location,bhk,bath,sqft)
    input = pd.DataFrame([[location,sqft,bath,bhk]],columns=['location','total_sqft','bath','BHK'])
    prediction = pipe.predict(input)[0]
    prediction = prediction * 100000
    return str(prediction)

if __name__ == "__main__":
    app.run(debug=True , port=5001)