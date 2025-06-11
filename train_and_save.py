from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Train a Random Forest model to get feature importances
rf = RandomForestClassifier(random_state=0)
rf.fit(X, y)
importances = rf.feature_importances_
# Get indices of top 7 features
top_indices = np.argsort(importances)[-7:]
X_selected = X[:, top_indices]

# Split data using the selected features
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=0)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest on the selected features
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)

# Save model and scaler
joblib.dump(rf, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved!")