# Lecture1-CipherSchools
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample data: distance (km), delivery_time (minutes), traffic (0: low, 1: medium, 2: high), on_time (0: no, 1: yes)
data = {
    'distance': [5, 10, 2, 8, 7, 3, 1, 4, 6, 12],
    'delivery_time': [30, 60, 15, 45, 50, 20, 10, 25, 35, 70],
    'traffic': [1, 2, 0, 1, 2, 0, 0, 1, 2, 2],
    'on_time': [1, 0, 1, 1, 0, 1, 1, 1, 0, 0]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and target variable
X = df[['distance', 'delivery_time', 'traffic']]
y = df['on_time']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')

# Predict the on-time delivery for a new order
new_order = [[7, 40, 1]]  # Example: 7 km, 40 minutes, medium traffic
prediction = model.predict(new_order)

print('On-time delivery prediction for new order:', 'Yes' if prediction[0] == 1 else 'No')
