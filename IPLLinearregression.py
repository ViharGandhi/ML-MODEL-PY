import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
# Load the dataset
dataset = pd.read_csv('IplData.csv')
for i in range(15):
    
    team = dataset['toss_winner'].values[i]
    teamnumber = label_encoder.fit_transform(dataset['toss_winner'])[i]
    print(team,teamnumber)



dataset['toss_winner'] = label_encoder.fit_transform(dataset['toss_winner'])
dataset['winner'] = label_encoder.fit_transform(dataset['winner'])

# Split the data into training and testing sets
X = dataset['toss_winner'].values.reshape(-1, 1)
y = dataset['winner'].values.reshape(-1, 1)

# Create the linear regression model
model = LinearRegression()

# Train the model on the training set
model.fit(X, y)

# Make predictions on the testing set
y_pred = model.predict(X)

# Visualize the data and the linear regression line
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.xlabel('toss_winner')
plt.ylabel('winner')
plt.show()
