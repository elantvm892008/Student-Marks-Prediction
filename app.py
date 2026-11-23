import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Student Marks Predictor", page_icon="📚")
st.title("📚 Student Marks Prediction")
st.write("Estimate a student's marks based on study hours using Linear Regression.")

@st.cache_data
def load_data():
    return pd.read_csv("students.csv")

data = load_data()
X = data[["study_hours"]]
y = data["marks"]

if len(data) >= 5:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
else:
    X_train, X_test, y_train, y_test = X, X, y, y

model = LinearRegression()
model.fit(X_train, y_train)

st.subheader("Try a prediction")
hours = st.number_input("Study hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5)

if st.button("Predict marks"):
    prediction = float(model.predict(pd.DataFrame({"study_hours": [hours]}))[0])
    prediction = max(0, min(100, prediction))
    st.success(f"Estimated marks: {prediction:.1f} / 100")

if len(data) >= 5:
    predictions = model.predict(X_test)
    st.caption(
        f"Test-set MAE: {mean_absolute_error(y_test, predictions):.2f} marks | "
        f"R²: {r2_score(y_test, predictions):.2f}"
    )

st.subheader("Sample dataset")
st.dataframe(data, use_container_width=True)
st.info("This is a learning demo using a small sample dataset. Study hours alone cannot reliably predict real exam performance.")
