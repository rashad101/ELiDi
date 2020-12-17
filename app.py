import flask
import os
from flask import Flask, render_template,request,jsonify


template_dir = os.path.abspath("ui")
app = flask.Flask(__name__,template_folder=template_dir)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['GET','POST'])
def ask():
    questions = []

    if request.method == 'POST':
        text = request.form['question']
        if text:
            processed_text = str(text)
            throughput = processed_text.upper()
            questions.append(throughput)

    return render_template('index.html', questions=questions)


app.run(debug=True)