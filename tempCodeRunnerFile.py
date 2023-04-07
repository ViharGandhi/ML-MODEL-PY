import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

label_encoder = LabelEncoder()

# Load the dataset
dataset = pd.read_csv('IPL_Data.csv')

dataset['National Side'] = label_encoder.fit_transform(dataset['National Side'])

# Replace NaN values with mean of the column
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(dataset[['National Side', 'MatchPlayed','BattingAVG','BattingS/R']])

y = dataset['RunsScored']

# Replace NaN values in y with mean of the column
y = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()

model = LinearRegression()

model.fit(X, y)

name_input = input("Enter country")
MatchPlayed_input = int(input("Match played are:"))
BattingAVG_input = float(input("Average is"))
BattingSr_input = float(input("Strike rate is"))
name_input_encode = label_encoder.transform([name_input])[0]

user_input = [[name_input_encode, MatchPlayed_input, BattingAVG_input, BattingSr_input]]
# Replace NaN values in user_input with mean of the column
user_input = imputer.transform(user_input)

prediction = model.predict(user_input)
print(prediction)
