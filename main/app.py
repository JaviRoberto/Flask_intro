import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from flask import Flask, render_template, request
from sklearn.metrics import accuracy_score



# Flask read template folder
app = Flask(__name__, template_folder='template')





#function to that both reads housing dating from CVS and trains AI, and takes in parameters to estimate house price.
def test_info(bedrooms, bathrooms,sqft_living, sqft_lot, floors, condition, grade):
    houses = pd.read_csv("kc_house_data.csv")

    # Drop unnecessary columns
    houses_mod = houses.drop(['id', 'date', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'waterfront','view','sqft_above','sqft_basement', 'sqft_lot15', 'sqft_living15'], axis=1)
    houses_mod.to_csv("house_mod.csv")
    # Save pairplot to file
    main_fig = sns.pairplot(houses_mod)
    main_fig.savefig("mainplot.png")
    plt.show()

    # Pairplot for 'price' and 'sqft_living'
    price_sqfig = sns.pairplot(houses_mod, x_vars=['price'], y_vars=['sqft_living'])
    price_sqfig.savefig("price_sq_livinq.png")
    plt.show()

    # Pairplot for 'price' and 'sqft_living'
    price_bathroooms = sns.pairplot(houses_mod, x_vars=['price'], y_vars=['bathrooms'])
    price_bathroooms.savefig("price_bath.png")
    plt.show()

    price_grade = sns.pairplot(houses_mod, x_vars=['price'], y_vars=['grade'])
    price_grade.savefig("price_grade.png")

    plt.show()  #Display any plots if needed

    #parses and prepares data to be trained.
    x = houses_mod.drop(['price'], axis=1)
    y = houses_mod['price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)

    #Calles model type and fits training data in.
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)

    #predicts left over 40% of data.
    predict = lin_reg.predict(x_test)

    #compares test data to prediction model
    plt.scatter(y_test, predict)
    plt.savefig('prediction_graph.png')  # Save the plot before displaying it
    plt.show()

    # assigns parameters to dict
    new_data = [['bedrooms', bedrooms], ['bathrooms', bathrooms],['sqft_living', sqft_living],['sqft_lot', sqft_lot], ['floors', floors], ['condition', condition], ['grade',grade]]

    # Convert the list of lists to a pandas DataFrame
    df = pd.DataFrame(new_data, columns=['feature', 'value'])

    # Map feature names to corresponding values
    features = df['feature'].values
    values = df['value'].values

    # Create a dictionary with feature names as keys and values as values
    input_dict = dict(zip(features, values))

    # Convert the dictionary to a DataFrame with a single row
    input_df = pd.DataFrame([input_dict])

    # Predict using the model
    new_pred = lin_reg.predict(input_df)




    #returnb prdiction into dollar format.
    return ("{:,}".format(int(new_pred[0])))
    #return int(new_pred[0] * 2.2)


#initiate flask requirments
app = Flask(__name__)

#assign index page
@app.route('/')
def index():
    return render_template('index.html')

#assign submission form response page
@app.route('/prediction', methods=['POST'])
def submit():
    #gathers input data from form
    if request.method == 'POST':
        bedrooms =request.form['bedrooms']
        bathrooms = request.form['bathrooms']
        sqft_living = request.form['sqft_living']
        sqft_lot = request.form['sqft_lot']
        floors = request.form['floors']
        condition = request.form['condition']
        grade = request.form['grade']
        #Runs Machine learning model by calling test_info()
        prediction = test_info(bedrooms, bathrooms,sqft_living, sqft_lot, floors, condition, grade)

        #returns and formats 2015 prediction and new 2024 prediction
        temp_prediction = int(float(prediction.replace(',', '')) * 2.2)
        new_prediction = f"{temp_prediction:,.2f}"

        #returns parameters to flask page
        return render_template("prediction.html",bedrooms=bedrooms,bathrooms= bathrooms,sqft_living=sqft_living,sqft_lot=sqft_lot,floors=floors,condition=condition, grade= grade, prediction=  prediction, new_prediction= new_prediction)



if __name__ == "__main__":
    app.run(debug=True)
