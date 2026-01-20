import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Page Config
st.set_page_config(page_title="Traffic & Parking Management System", layout="centered")

st.title("🚦 AI-Driven Traffic & Parking Management System")
st.markdown("Using **Data Analytics + Machine Learning** for smart urban mobility")

# Load & Prepare Dataset
np.random.seed(42)

data = {
    "time": np.random.randint(6, 23, 100),
    "vehicle_count": np.random.randint(50, 500, 100),
    "parking_slots_used": np.random.randint(50, 200, 100),
}

df = pd.DataFrame(data)

df["congestion_level"] = pd.cut(
    df["vehicle_count"],
    bins=[0, 150, 300, 500],
    labels=["Low", "Medium", "High"]
)

df["parking_status"] = pd.cut(
    df["parking_slots_used"],
    bins=[0, 120, 180, 200],
    labels=["Available", "Limited", "Full"]
)

# Encode Targets
le_cong = LabelEncoder()
df["congestion_encoded"] = le_cong.fit_transform(df["congestion_level"].astype(str))

le_park = LabelEncoder()
df["parking_encoded"] = le_park.fit_transform(df["parking_status"].astype(str))

X = df[["time", "vehicle_count", "parking_slots_used"]]

# Train Models
congestion_model = DecisionTreeClassifier(random_state=42)
congestion_model.fit(X, df["congestion_encoded"])

parking_model = DecisionTreeClassifier(random_state=42)
parking_model.fit(X, df["parking_encoded"])

# User Input Section
st.header("🔢 Enter Traffic Details")

time = st.slider("Time (Hour of Day)", 6, 22, 6 )
vehicle_count = st.slider("Vehicle Count", 50, 500, 200)
parking_used = st.slider("Parking Slots Used", 50, 200, 120)

input_data = pd.DataFrame([[time, vehicle_count, parking_used]],
                          columns=["time", "vehicle_count", "parking_slots_used"])

# Prediction
if st.button("🔮 Predict"):
    congestion_pred = congestion_model.predict(input_data)
    parking_pred = parking_model.predict(input_data)

    congestion_result = le_cong.inverse_transform(congestion_pred)[0]
    parking_result = le_park.inverse_transform(parking_pred)[0]

    st.subheader("📊 Prediction Results")
    st.success(f"🚦 Traffic Congestion Level: **{congestion_result}**")
    st.info(f"🅿️ Parking Availability: **{parking_result}**")

# Footer
st.markdown("---")
st.caption("Smart Governance | AI + Data Analytics Project")
