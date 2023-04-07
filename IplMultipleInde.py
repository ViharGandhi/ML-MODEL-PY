import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# Load the dataset
dataset = pd.read_csv('IplData.csv')

dataset['toss_decision'] = label_encoder.fit_transform(dataset['toss_decision'])
dataset['toss_winner'] = label_encoder.fit_transform(dataset['toss_winner'])
dataset['winner'] = label_encoder.fit_transform(dataset['winner'])

# Split the data into independent and dependent variables
X = dataset[['toss_winner', 'toss_decision']].values.reshape(-1, 2)
y = dataset['winner'].values.reshape(-1, 1)

# Create the linear regression model
model = LinearRegression()

# Train the model on the training set
model.fit(X, y)

# Make predictions on the testing set
y_pred = model.predict(X)

# Visualize the data and the linear regression line
plt.scatter(X[:, 0], y, color='blue')
plt.scatter(X[:, 1], y, color='green')
plt.plot(X[:, 0], y_pred, color='red')
plt.xlabel('toss_winner, toss_decision')
plt.ylabel('winner')
plt.show()
