import math
from flask import Flask, render_template, url_for, request
import pickle
import os
from Week5 import breast_cancer
import numpy as np

model = pickle.load(open("breast_cancer.pkl","rb"))

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = False
    if request.method == 'POST':
        form = request.form
        result = calculate(form)
    return render_template('index.html', result=result)


def calculate(form):

    mean_radius = float(request.form['mean radius'])
    mean_texture = float(request.form['mean texture'])
    mean_perimeter = float(request.form['mean perimeter'])
    mean_area = float(request.form['mean area'])
    mean_smoothness  = float(request.form['mean smoothness'])
    mean_compactness = float(request.form['mean compactness'])
    #mean_concavity = float(request.form['mean concavity'])
    #mean_concave_points  = float(request.form['mean concave points'])
    #mean_symmetry = float(request.form['mean symmetry'])
    #mean_fractal_dimension = float(request.form['mean fractal dimension'])

    #QTc_result = entry2.all()
    result = model.predict(mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness) #, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension)
    return result


if __name__ == "__main__":
    app.run(debug=True)