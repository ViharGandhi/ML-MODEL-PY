
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset
dataset = pd.read_csv('train.csv')

# Split the data into training and testing sets
X = dataset['size'].values.reshape(-1, 1)
y = dataset['Price'].values.reshape(-1, 1)

# Create the linear regression model
model = LinearRegression()

# Train the model on the training set
model.fit(X, y)

# Make predictions on the testing set
y_pred = model.predict(X)

# Visualize the data and the linear regression line

size_input = int(input("Enter the toss winner team name: "))
user_input = [[size_input]]
prediction = model.predict(user_input)
print(prediction)