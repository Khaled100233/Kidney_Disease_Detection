from flask import Flask, request, render_template
from joblib import load
import pickle



app = Flask(__name__)
model = load('ml.pkl') # or load('ml.pkl') if error

@app.route('/', methods=["POST", 'GET'])
def home():
    if request.method == "POST":
        dia = [[float(x) for x in request.form.values()]]
        predict = model.predict(dia)
        if predict == 0:
            predict = "You have Kidney Disease"
        else:
            predict = "You don't have Kidney Disease"
        return render_template('answer.html', predict=predict)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run()
