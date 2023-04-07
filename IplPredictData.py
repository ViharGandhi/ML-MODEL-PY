import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# Load the dataset
dataset = pd.read_csv('IplData.csv')

dataset['toss_decision'] = label_encoder.fit_transform(
    dataset['toss_decision'])
dataset['toss_winner'] = label_encoder.fit_transform(dataset['toss_winner'])
dataset['winner'] = label_encoder.fit_transform(dataset['winner'])

# Split the data into independent and dependent variables
X = dataset[['toss_decision', 'toss_winner']].values.reshape(-1, 2)
y = dataset['winner'].values.reshape(-1, 1)

# # Create the linear regression model
model = LinearRegression()

# # Train the model on the training set
model.fit(X, y)

# # Get user input
toss_winner_input = input("Enter the toss winner team name: ")
toss_decision_input = input("Enter the toss decision (bat/field): ")

if toss_decision_input=='bat':
    toss_decision_input = 0
else:
    toss_decision_input = 1

#  Encode the user input using the label encoder
toss_winner_encoded = label_encoder.transform([toss_winner_input])[0]
# print(toss_winner_encoded)
# toss_decision_encoded = label_encoder.transform([toss_decision_input])[0]
# print(toss_decision_encoded)
# # # Make a prediction based on the user input
user_input = [[toss_winner_encoded, toss_decision_input]]
prediction = model.predict(user_input)

# # # # Decode the predicted label using the label encoder
predicted_winner = label_encoder.inverse_transform(prediction)[0]

# # # # Print the predicted winner
print("The predicted winner is:", predicted_winner)
