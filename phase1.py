import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)

data = {
    "date": pd.date_range(start="2025-01-01", periods=100),
    "time": np.random.randint(6, 23, 100),
    "area": np.random.choice(["Area A", "Area B", "Area C", "Area D"], 100),
    "vehicle_count": np.random.randint(50, 500, 100),
    "parking_slots_total": 200,
}

df = pd.DataFrame(data)

df["parking_slots_used"] = np.random.randint(50, 200, 100)

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

print(df.head())




print(df.describe())
print(df["congestion_level"].value_counts())
print(df["parking_status"].value_counts())






import matplotlib.pyplot as plt

plt.figure()
plt.plot(df["time"], df["vehicle_count"])
plt.xlabel("Time (Hour of Day)")
plt.ylabel("Vehicle Count")
plt.title("Traffic Volume vs Time")
plt.show()


df["congestion_level"].value_counts().plot(kind="bar")
plt.xlabel("Congestion Level")
plt.ylabel("Number of Observations")
plt.title("Traffic Congestion Distribution")
plt.show()


plt.figure()
plt.hist(df["parking_slots_used"])
plt.xlabel("Parking Slots Used")
plt.ylabel("Frequency")
plt.title("Parking Usage Distribution")
plt.show()

area_traffic = df.groupby("area")["vehicle_count"].mean()

area_traffic.plot(kind="bar")
plt.xlabel("Area")
plt.ylabel("Average Vehicle Count")
plt.title("Average Traffic by Area")
plt.show()




# Encode target labels
le = LabelEncoder()
df["congestion_encoded"] = le.fit_transform(df["congestion_level"])

X = df[["time", "vehicle_count", "parking_slots_used"]]
y = df["congestion_encoded"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("\n--- Parking Availability Prediction Model ---")

# Encode parking status
df["parking_status"] = df["parking_status"].astype(str)
le_parking = LabelEncoder()
df["parking_encoded"] = le_parking.fit_transform(df["parking_status"])

X_p = df[["time", "vehicle_count", "parking_slots_used"]]
y_p = df["parking_encoded"]

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_p, y_p, test_size=0.2, random_state=42
)

parking_model = DecisionTreeClassifier(random_state=42)
parking_model.fit(X_train_p, y_train_p)

y_pred_p = parking_model.predict(X_test_p)

print("Parking Model Accuracy:", accuracy_score(y_test_p, y_pred_p))
print(classification_report(y_test_p, y_pred_p))


#happy ending guys with my chotasa project with ml model