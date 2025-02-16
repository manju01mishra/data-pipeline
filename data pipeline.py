import pandas as pd

# Example: Reading data from a CSV file
data = pd.read_csv('data.csv')
# Handling missing values
data.fila(method='fill', inplace=True)

# Dropping duplicate rows
data.drop_duplicates(inplace=True)
# Example: Creating a new feature
data['new_feature'] = data['existing_feature'] ** 2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Splitting the data into training and testing sets
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicting and evaluating the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')
import joblib

# Saving the model
joblib.dump(model, 'model.joblib')

# Loading the model (for future use)
loaded_model = joblib.load('model.joblib')
