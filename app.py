import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

houses = pd.read_csv('Walmart_Store_sales.csv')



houses.columns
from flask import Flask, render_template, url_for
import requests
import json

app = Flask(__name__)

@app.route('/')
def index(): 
    return render_template('index.html')


if __name__ == "__main__": 
    app.run(debug=True)



