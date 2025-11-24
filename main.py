import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datetime import datetime

# Step 1: Hardcoded dataset (expanded to ~30 rows)
data = {
    'Age': [25,30,45,50,35,40,60,55,48,33,29,52,41,38,47,36,58,62,44,39,31,49,53,46,34,37,42,51,59,28],
    'BMI': [22.5,28.0,30.5,35.2,26.8,29.1,32.0,31.5,27.3,24.0,23.5,33.1,28.7,27.9,30.2,25.6,34.5,36.0,29.8,28.2,26.0,32.5,31.8,30.0,27.0,28.5,29.9,33.0,35.5,24.5],
    'Glucose': [85,90,120,150,100,110,160,140,130,95,88,145,118,105,135,99,155,165,125,112,92,138,142,128,97,108,115,148,158,89],
    'BloodPressure': [70,80,85,90,75,88,95,92,86,78,72,91,84,83,87,76,93,96,82,81,74,89,90,85,77,79,86,92,94,73],
    'Diabetes': [0,0,1,1,0,0,1,1,1,0,0,1,0,0,1,0,1,1,1,0,0,1,1,1,0,0,0,1,1,0]
}

df = pd.DataFrame(data)

# Step 2: Features & target
X = df[['Age', 'BMI', 'Glucose', 'BloodPressure']]
y = df['Diabetes']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Timestamp for filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Step 6: Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes','Diabetes'],
            yticklabels=['No Diabetes','Diabetes'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()   # show first
plt.savefig(f"confusion_matrix_{timestamp}.png")  # then save
plt.close()

# Step 7: Feature Distribution Visualization
plt.figure(figsize=(8,6))
sns.scatterplot(x='Glucose', y='BMI', hue='Diabetes', data=df, palette='Set1')
plt.title("Glucose vs BMI (Colored by Diabetes)")
plt.show()   # show first
plt.savefig(f"glucose_vs_bmi_{timestamp}.png")    # then save
plt.close()

# Step 8: Predict new patient
new_patient = pd.DataFrame([[45, 29.0, 135, 85]], 
                           columns=['Age','BMI','Glucose','BloodPressure'])
prediction = model.predict(new_patient)

print("Prediction (1 = Diabetes, 0 = No Diabetes):", prediction[0])

# Save prediction result to a text file
with open(f"prediction_output_{timestamp}.txt", "w") as f:
    f.write("New Patient Data:\n")
    f.write(str(new_patient.to_dict(orient="records")[0]) + "\n")
    f.write(f"Prediction (1 = Diabetes, 0 = No Diabetes): {prediction[0]}\n")

print(f"Prediction saved to prediction_output_{timestamp}.txt")
print(f"Plots saved as confusion_matrix_{timestamp}.png and glucose_vs_bmi_{timestamp}.png")