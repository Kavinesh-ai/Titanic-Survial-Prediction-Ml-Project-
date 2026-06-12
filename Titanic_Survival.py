import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Dataset
data = pd.read_csv("Titanic.csv")

# Features
X = data[["Pclass", "Age"]]

# Target
y = data["Survived"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LogisticRegression()

model.fit(X_train, y_train)

# Make Predictions
predictions = model.predict(X_test)

# Calculate Accuracy
score = accuracy_score(y_test, predictions)

# Output
print("Titanic Survival Prediction")
print("-" * 35)

print("Accuracy:", round(score, 2))

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

print("\nSample Predictions")
print(results.head())