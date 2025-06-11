from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
print(data.feature_names)
print(data.data[0])  # Example values for the first sample
