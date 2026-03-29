import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load data
data = pd.read_csv('../data/students.csv')

X = data[['hours_studied', 'attendance']]
y = data['passed']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open('../model/model.pkl', 'wb'))

print("Model trained and saved!")