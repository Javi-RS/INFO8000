from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from io import BytesIO
import base64
sns.set()
app = Flask("iris_online_example",template_folder='templates')
clf = None
df = None

#userful function that gets the current figure as a base 64 image for embedding into websites
def getCurrFigAsBase64HTML():
    im_buf_arr = BytesIO()
    plt.gcf().savefig(im_buf_arr,format='png')
    im_buf_arr.seek(0)
    b64data = base64.b64encode(im_buf_arr.read()).decode('utf8');
    return render_template('img.html',img_data=b64data) 
    
def train():
    global df, clf
    X = df.drop(columns=['variety'])
    y = df['variety']
    clf = GaussianNB()
    clf.fit(X,y)
    pickle.dump(clf,open("model","wb"))
    pickle.dump(df,open("data","wb"))
    return clf.score(X,y)
    
def init():
    global df
    np.random.seed(8000)
    df = pd.read_csv('iris.csv')
    train()
    

try:
    clf = pickle.load(open("model","rb"))
    df = pickle.load(open("data","rb"))
except:
    init()

#this method resets/initializes everything (database, model) (should probably password protect this)
@app.route("/reset")
def reset():
    init()
    return "reset model"
    

# show an interface to add/test data, which will hit test
@app.route("/")
def main():
    return render_template("main.html")

# this function adds a row to the dataset and retrains
@app.route("/run_obs'ervation",methods=["POST"])
def add_data():
    global df
    global clf
    try:
        sepal_width = float(request.values.get('sepal_width',0.0))
        petal_width = float(request.values.get('petal_width',0.0))
        sepal_length = float(request.values.get('sepal_length',0.0))
        petal_length = float(request.values.get('petal_length',0.0))
        variety = request.values.get('variety','Virginica')
        is_add = request.values.get("add","no")
        is_test = request.values.get("test","no")
    except: 
        return "Error parsing entries"
    
    if is_add != "no":
        obs = pd.DataFrame([[sepal_length,sepal_width,petal_length,petal_width,variety]],
                           columns=["sepal.length","sepal.width","petal.length","petal.width","variety"])
        df = pd.concat([df,obs],ignore_index=True)
        s = train()
        sns.pairplot(data=df,hue="variety")
        img_html = getCurrFigAsBase64HTML();
        return "Added new sample " + "<pre>"+ df.to_string() \
            + "</pre><br> ... <br> and retrained. <br>  Score is now: " + str(s) + "<br>" + img_html
    
    if is_test != "no":
        obs = pd.DataFrame([[sepal_length,sepal_width,petal_length,petal_width]],
                           columns=["sepal.length","sepal.width","petal.length","petal.width"])
        return clf.predict(obs)[0]
        
    return "not implemented"
if __name__ == "__main__":
    app.run()