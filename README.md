Disease Prediction Project

A simple machine learning project that predicts diabetes using a **hardcoded dataset** with Logistic Regression. Includes evaluation metrics and visualizations, with outputs saved automatically.

Features
- Logistic Regression model  
- Accuracy & classification report  
- Confusion matrix (heatmap)  
- Scatter plot (Glucose vs BMI)  
- Automatic saving of results (PNG, TXT with timestamps)  

Usage
```bash
python main.py
```

Outputs
- Console: Accuracy, classification report, prediction  
- Files:  
  - `prediction_output_<timestamp>.txt`  
  - `confusion_matrix_<timestamp>.png`  
  - `glucose_vs_bmi_<timestamp>.png`  

Requirements
- Python 3.8+  
- pandas, scikit-learn, matplotlib, seaborn  
