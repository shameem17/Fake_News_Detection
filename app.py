
# import packages
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd 
from sklearn.model_selection import train_test_split 
import re
import string 
import os 
from sklearn.feature_extraction.text import TfidfVectorizer


app=Flask(__name__)

#  import pictures for webview
picFolder = os.path.join('static','pic')

app.config['UPLOAD_FOLDER'] = picFolder


# Load both the saved model and the fitted vectorizer
with open('model_and_vectorizer.pkl', 'rb') as f:
    vectorization, model = pickle.load(f)



def word_process(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text= re.sub("\\W"," ",text)
    text=re.sub('http?://\S+|www\.\S+','',text)
    text=re.sub('<.*?>+','',text)
    text=re.sub('[%s]' %re.escape(string.punctuation),'',text)
    text=re.sub('\n','',text)
    text=re.sub('\w*\d\w*','',text)
    return text

a=1.0
def output_label(n):
    if n==False:
        return "Fake"
    else:
        return "True"
# you can add the saved model but you need to do all the data processing for the input 
def manual_testing(news):
    testing_news={"text":[news]}
    new_def_test=pd.DataFrame(testing_news)
    new_def_test["text"]=new_def_test["text"].apply(word_process)
    new_x_test=new_def_test["text"]
    new_xv_test= vectorization.transform(new_x_test)
    predict_rfc=model.predict(new_xv_test)
    prob_rfc=model.predict_proba(new_xv_test)
    a=prob_rfc[0][1]
    result=output_label(predict_rfc)
    return result

twt = os.path.join(app.config['UPLOAD_FOLDER'],'twitter.png')
fb = os.path.join(app.config['UPLOAD_FOLDER'],'facebook.png')
ins = os.path.join(app.config['UPLOAD_FOLDER'],'instagram.png')

# define the route
@app.route('/')
def home():
    return render_template('home.html',img1=twt,fbp=fb,inst=ins) #pass the parameters

@app.route('/about')
def about():
    return render_template('about.html',img1=twt,fbp=fb,inst=ins)
@app.route('/predt')
def predt():
     return render_template('index.html',img1=twt,fbp=fb,inst=ins)

@app.route('/cont')
def cont():
     return render_template('contact.html',img1=twt,fbp=fb,inst=ins)

@app.route('/home2')
def home2():
     return render_template('home.html',img1=twt,fbp=fb,inst=ins)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = manual_testing(message)

        return render_template('index.html', prediction=pred,img1=twt,fbp=fb,inst=ins)
    else:
        return render_template('index.html', prediction="Something went wrong",img1=twt,fbp=fb,inst=ins)

if __name__ == '__main__':
    app.run(debug=True)
