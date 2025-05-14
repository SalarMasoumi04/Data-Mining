import pandas as pd
import matplotlib.pyplot as plt

# Data Sets Upload
data = pd.read_csv("diabetes_dataset.csv")

# Boxplot Drawing
features = ['Glucose', 'BMI', 'Insulin', 'SkinThickness']
data[features].boxplot(figsize=(10, 6))
plt.title("Boxplot of Numerical Features")
plt.ylabel("Values")
plt.show()