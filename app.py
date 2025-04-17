import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import os

st.set_page_config(page_title="Delivery Time Predictor", page_icon="🚚", layout="wide")
st.title("Delivery Time Estimator")
st.markdown("Predict the estimated delivery time based on order details.")

# ✅ Load the Model
model_path = r"C:\Users\Dell\OneDrive\Desktop\Delivery-Time-Estimator-main\Delivery-Time-Estimator-main\delivery_time_model.pkl"
if not os.path.exists(model_path):
    st.error("🚨 Model file 'delivery_time_model.pkl' not found! Make sure it's in the correct directory.")
    st.stop()

try:
    model = joblib.load(r"C:\Users\Dell\OneDrive\Desktop\Delivery-Time-Estimator-main\Delivery-Time-Estimator-main\delivery_time_model.pkl")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ✅ Streamlit Sidebar Inputs
st.sidebar.title(" Order Details")
product_category = st.sidebar.selectbox(" Select Product Category", ["Electronics", "Clothing", "Furniture", "Books", "Others"])
customer_location = st.sidebar.selectbox(" Select Customer Location", ["Urban", "Suburban", "Rural"])
shipping_method = st.sidebar.selectbox(" Select Shipping Method", ["Standard", "Express", "Same-Day"])

order_quantity = st.sidebar.number_input("📦 Enter Order Quantity", min_value=1, step=1)

# ✅ Use Day Names Instead of Numbers
days_dict = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6
}
order_day = st.sidebar.selectbox(" Purchased Day", list(days_dict.keys()))  
order_hour = st.sidebar.slider(" Purchased Hour (0-23)", min_value=0, max_value=23)

# ✅ Use Numeric Input for Distance Instead of Dropdown
distance = st.sidebar.number_input(" Shipping Distance (in km)", min_value=1, step=1)

# Convert Inputs to Model-Compatible Format
feature_dict = {
    "Electronics": 0, "Clothing": 1, "Furniture": 2, "Books": 3, "Others": 4,
    "Urban": 0, "Suburban": 1, "Rural": 2,
    "Standard": 0, "Express": 1, "Same-Day": 2,
}

# ✅ Ensure Feature Count Matches Model
input_features = np.array([
    feature_dict[product_category], 
    feature_dict[customer_location], 
    feature_dict[shipping_method],
    order_quantity, 
    days_dict[order_day],  
    order_hour,
    distance  # Using actual distance in km
]).reshape(1, -1)  

# 🔍 Predict Delivery Time
if st.sidebar.button(" Predict Delivery Time"):
    try:
        predicted_time = model.predict(input_features)[0]
        st.subheader(f" Estimated Delivery Time: **{predicted_time:.2f} days**")

        # ✅ 📊 Improved Graph Visualization 
        chart_data = pd.DataFrame({"Delivery Estimate": ["Predicted Delivery Time"], "Days": [predicted_time]})
        chart = alt.Chart(chart_data).mark_bar(size=50).encode(
            x=alt.X("Days:Q", title="Estimated Days", scale=alt.Scale(domain=(0, predicted_time + 2))),
            y=alt.Y("Delivery Estimate:N", title=""),
            color=alt.value("#FF4B4B"),

            tooltip=["Days"]

        ).properties(
            title=" Estimated Delivery Time",
            width=700,
            height=250
        )

        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f" Prediction failed: {e}")

st.markdown("---")
st.markdown(" Get your estimated delivery time instantly with this tool!") 