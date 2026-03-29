import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Regression Dashboard", layout="wide")
st.title("📊 Regression Dashboard")

# ==============================
# LOAD DATA (STATIC CSV)
# ==============================
data_path = r"C:\Users\Male-26\Desktop\Saad Shaikhr\Regression model file.csv"
df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()

# ==============================
# PREPROCESSING
# ==============================
target = "target_column"  # CHANGE this to your actual target column
df = df.fillna(df.mean(numeric_only=True))

X = df.drop(target, axis=1)
y = df[target]

original_X = X.copy()
X = pd.get_dummies(X, drop_first=True)

# TRAIN MODEL
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# ==============================
# KPI CARDS
# ==============================
st.subheader("📌 Key Metrics")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric("Rows", df.shape[0])
kpi2.metric("Columns", df.shape[1])
kpi3.metric("MAE", round(mae, 3))
kpi4.metric("R² Score", round(r2, 3))

# ==============================
# FIRST ROW CHARTS
# ==============================
st.subheader("📊 Data Visualizations")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Histogram")
    fig, ax = plt.subplots()
    sns.histplot(df[numeric_cols[0]], kde=True, ax=ax, color="skyblue")
    st.pyplot(fig)

with col2:
    st.markdown("### Scatter Plot")
    if len(numeric_cols) > 1:
        fig2, ax2 = plt.subplots()
        sns.scatterplot(x=df[numeric_cols[0]], y=df[numeric_cols[1]], ax=ax2, color="orange")
        st.pyplot(fig2)

# ==============================
# SECOND ROW CHARTS
# ==============================
st.subheader("📉 Model Performance & Correlation")
col3, col4 = st.columns(2)

with col3:
    st.markdown("### Actual vs Predicted")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax3, color="green")
    ax3.set_xlabel("Actual")
    ax3.set_ylabel("Predicted")
    st.pyplot(fig3)

with col4:
    st.markdown("### Correlation Heatmap")
    fig4, ax4 = plt.subplots(figsize=(6,5))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax4)
    st.pyplot(fig4)

# ==============================
# PREDICTION PANEL
# ==============================
st.subheader("🔢 Example Prediction")
input_data = {}
for col in original_X.columns:
    if original_X[col].dtype == "object":
        input_data[col] = original_X[col].mode()[0]
    else:
        input_data[col] = float(original_X[col].mean())

input_df = pd.DataFrame([input_data])
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=X.columns, fill_value=0)

predicted_value = model.predict(input_df)[0]
st.success(f"Predicted Value (using mean/mode inputs): {predicted_value:.2f}")

# ==============================
# MODEL DETAILS
# ==============================
st.subheader("📌 Model Details")
st.write("Intercept:", model.intercept_)
st.write("Coefficients:", model.coef_)