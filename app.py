from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd 
from sklearn.model_selection import train_test_split 
import re
import string 


app=Flask(__name__)
fake_news=pd.read_csv("Fake.csv")
true_news=pd.read_csv("True.csv")
fake_news["label"]=False
true_news["label"]=True
fake_manual_testing=fake_news.tail(1000)
for i in range(23480,22480,-1):
    fake_news.drop([i],axis=0,inplace=True)
    
true_manual_testing=true_news.tail(1000)
for i in range(21416,20416,-1):
    true_news.drop([i],axis=0,inplace=True)

manual_testing=pd.concat([true_manual_testing, fake_manual_testing], axis=0)

marged_data=pd.concat([fake_news, true_news], axis=0)
temp=marged_data.drop(["title","subject","date"], axis=1)
temp=temp.sample(frac=1)

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


temp["text"]=temp["text"].apply(word_process)
x=temp["text"]
y=temp["label"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
###vectorize
###from sklearn.feature_extraction.text import TfidfVectorizer
vectorization =TfidfVectorizer()
xv_train=vectorization.fit_transform(x_train)
xv_test=vectorization.transform(x_test)

###logistic regression
from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(xv_train,y_train)
LogisticRegression()
predict_LR=LR.predict(xv_test)
prob_LR=LR.predict_proba(xv_test)

a=1.0;
def output_label(n):
    if n==False:
        return "Fake"
    else:
        return "True"

def manual_testing(news):
    testing_news={"text":[news]}
    new_def_test=pd.DataFrame(testing_news)
    new_def_test["text"]=new_def_test["text"].apply(word_process)
    new_x_test=new_def_test["text"]
    new_xv_test= vectorization.transform(new_x_test)
    predict_LR= LR.predict(new_xv_test)
    prob_LR=LR.predict_proba(new_xv_test)
    a=prob_LR[0][1]
    print(a)
    result=output_label(predict_LR)
    return result


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = manual_testing(message)
        print(pred)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)
