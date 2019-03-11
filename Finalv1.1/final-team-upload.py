# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:47:01 2019

@author: nt18254
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Read the train data. this data is used to train our model.
training_data = pd.read_csv('C:\\Users\\natha\\OneDrive - University of Essex\\Team Project\\home-data-for-ml-course\\train.csv')#change file directory to test

# we are creating the model
train_y = training_data.SalePrice
predictor_vals = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']#these are the variables we have selected to use.
#these variables will be the ones the model focuses on when we predict the test data.
# Create training predictors data
train_X = training_data[predictor_vals]

The_model = RandomForestRegressor()#the type of regression used
The_model.fit(train_X, train_y)
# Read the test data
test_data_read = pd.read_csv('C:\\Users\\natha\\OneDrive - University of Essex\\Team Project\\home-data-for-ml-course\\test.csv')
# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test_data_read[predictor_vals]
# Use the model to make predictions
predicted_prices = The_model.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)
my_submission = pd.DataFrame({'Id': test_data_read.Id, 'SalePrice': predicted_prices})
#We have elected to keep with file name submission (any name can be chosen).
my_submission.to_csv('submission.csv', index=False)