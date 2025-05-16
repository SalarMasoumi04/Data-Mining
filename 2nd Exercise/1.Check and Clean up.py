import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Dataset Upload
data = pd.read_csv("diabetes_dataset.csv")

#BoxPlot
features = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
plt.figure(figsize=(12, 6))
sns.boxplot(data=data[features])
plt.title("Boxplot of Features to Detect Outliers")
plt.show()

#IQR
Q1 = data[features].quantile(0.25)
Q3 = data[features].quantile(0.75)
IQR = Q3 - Q1

#Deleting extra datas
data_cleaned = data[~((data[features] < (Q1 - 1.5 * IQR)) | (data[features] > (Q3 + 1.5 * IQR))).any(axis=1)]

#Histogram
data_cleaned.hist(figsize=(12, 8))
plt.show()

#Dataset Upload
data = pd.read_csv("diabetes_dataset.csv")

#Convert Number to Binary
features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

#Convert Number to Binary
binary_data = data.copy()
for feature in features:
    threshold = data[feature].median()  #Miane
    binary_data[feature] = (data[feature] >= threshold).astype(int)  # Convert to 0 & 1

#New Data Set
print(binary_data.head())