from flask import Flask,render_template,url_for,request
import pandas as pd 

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import pickle

# load the model from disk
filename = 'nb_model.pkl'
clf = pickle.load(open(filename, 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():


	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		#vect = cv.transform(data).toarray()
		out_prediction = clf.predict(data)
	return render_template('result.html',prediction = out_prediction)



if __name__ == '__main__':
	app.run()
    
