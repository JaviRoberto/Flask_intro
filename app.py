import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Read data
houses = pd.read_csv("kc_house_data.csv")

# Display the first few rows and information about the dataframe
print(houses.head())
print(houses.info())

# Drop unnecessary columns
# houses_mod = houses.drop(['id', 'date', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long'], axis=1)

houses_mod = houses.drop(['id', 'date', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'waterfront','view','sqft_above','sqft_basement', 'sqft_lot15', 'sqft_living15'], axis=1)
houses_mod.to_csv("house_mod.csv")

# Display the first few rows and information about the modified dataframe
print(houses_mod.head())
print(houses_mod.info())

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


#price_sqlot = sns.pairplot(houses_mod, x_vars=['price'], y_vars=['sqft_lot15'])
#price_sqlot.savefig("price_sqare_lot.png")


plt.show()  #Display any plots if needed

x = houses_mod.drop(['price'], axis=1)
y = houses_mod['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)

lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

predict = lin_reg.predict(x_test)

print(predict)
print(y_test)


plt.scatter(y_test, predict)
plt.savefig('prediction_graph.png')  # Save the plot before displaying it
plt.show()






#  #   Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   PRICE         4801 non-null   int64
#  1   BEDS          4801 non-null   int64
#  2   BATH          4801 non-null   float64
#  3   PROPERTYSQFT  4801 non-null   float64




from flask import Flask, render_template, url_for
import requests
import json

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
