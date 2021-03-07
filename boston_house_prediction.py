import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# Load the boston housing data set from sklearn.datasets and print it
from sklearn.datasets import load_boston
boston=load_boston()

# Transform the dataset into a data frame
# data = the data we want and the independent variables also known as the x values
# feature_names = the columns name of the data
# target = the target variables or the price fo the houses or dependent variables also known as the y value
df_x = pd.DataFrame(boston.data,columns=boston.feature_names)
df_y = pd.DataFrame(boston.target)
# df_x.to_csv('boston_house_prediction.csv',index=False)

# Get some statistics from the data set, count, mean
df_x.describe()

df_y.describe()

# Initialize the linear regression model
reg=linear_model.LinearRegression()

# splite the data into 67% training and 33% testing data
x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.33,random_state=42)

# Train the model with our training data
reg.fit(x_train,y_train)

# print the coefficients/weights for each feature/column of our model
print(reg.coef_) # f(x)=mx+b=y where m=coefficient

# print the prediction on our test data
y_pred=reg.predict(x_test)

# print the actual values
print(y_test)

# check the model performance/accuracy using Mean Squared Error (MSE)
print(np.mean((y_pred-y_test)**2))

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,y_pred))