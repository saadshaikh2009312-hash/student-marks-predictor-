import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# -------------------------------
# 1. Generate mock student data
# -------------------------------
def generate_data(n=500):
    np.random.seed(42)
    hours_study = np.random.randint(0, 20, n)
    attendance = np.random.randint(50, 101, n)
    assignment = np.random.randint(50, 101, n)
    midterm = np.random.randint(50, 101, n)

    # Final grade formula + randomness
    final_grade = (
        0.3 * hours_study + 0.2 * attendance + 0.2 * assignment + 0.3 * midterm
        + np.random.normal(0, 5, n)
    )
    final_grade = np.clip(final_grade, 0, 100)

    data = pd.DataFrame({
        "Hours_Study": hours_study,
        "Attendance": attendance,
        "Assignment_Score": assignment,
        "Midterm": midterm,
        "Final_Grade": final_grade
    })
    return data

data = generate_data()
print("Sample data:")
print(data.head())

# -------------------------------
# 2. Prepare features and target
# -------------------------------
X = data[["Hours_Study", "Attendance", "Assignment_Score", "Midterm"]]
y = data["Final_Grade"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 3. Train Linear Regression Model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# 4. Evaluate model
# -------------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation:\nMSE: {mse:.2f}\nR2 Score: {r2:.2f}")

# -------------------------------
# 5. Save model
# -------------------------------
joblib.dump(model, "student_model.pkl")
print("Model saved as 'student_model.pkl'")