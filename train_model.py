import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load the new logical ship data
df = pd.read_csv("vessel_signatures.csv")
X = df.drop('Ship_Type', axis=1)
y = df['Ship_Type']

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scaling (Makes sure the AI treats 'Length' and 'Speed' fairly)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 4. Train KNN (Look at 5 most similar ships)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# 5. Save the brain
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Ship AI Trained with 1,000 logical examples!")