import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  

# Step 1: Load the dataset  
df = pd.read_csv("diabetes.csv")  # Ensure this file is in the same folder  

# Step 2: Data Preprocessing  
print(df.isnull().sum())  # Check for missing values  
print(df.describe())  # Summary statistics  

# Step 3: Visualize correlations  
plt.figure(figsize=(10,6))  
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)  
plt.show()  

# Step 4: Feature selection  
X = df.drop(columns=["Outcome"])  # Features  
y = df["Outcome"]  # Target variable  

# Normalize the dataset  
scaler = StandardScaler()  
X_scaled = scaler.fit_transform(X)  

# Split Data for Training & Testing  
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)  

# Step 5: Train ML Model  
model = RandomForestClassifier(n_estimators=100, random_state=42)  
model.fit(X_train, y_train)  

# Step 6: Model Evaluation  
y_pred = model.predict(X_test)  
accuracy = accuracy_score(y_test, y_pred)  
print("Accuracy:", accuracy)  
print("Classification Report:\n", classification_report(y_test, y_pred))  

# Confusion Matrix  
cm = confusion_matrix(y_test, y_pred)  
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=0.5)  
plt.xlabel("Predicted")  
plt.ylabel("Actual")  
plt.title("Confusion Matrix")  
plt.show()  

# Step 7: Make Predictions on New Data  
new_data = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])  
new_data_scaled = scaler.transform(new_data)  
prediction = model.predict(new_data_scaled)  
print("Prediction:", "Diabetic" if prediction[0] == 1 else "Non-Diabetic")  
# Taking user input for prediction  
print("Enter patient details:")  
features = [float(input(f"Enter {col}: ")) for col in df.columns[:-1]]  

# Scale input & predict  
new_data_scaled = scaler.transform([features])  
prediction = model.predict(new_data_scaled)  

print("Prediction:", "Diabetic" if prediction[0] == 1 else "Non-Diabetic")  
