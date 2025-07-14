import zipfile
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 🔓 1. Extract ZIP file
zip_path = r"C:\Users\RISHITA\Downloads\archive.zip"
  # Rename this to match your ZIP file
extract_dir = './data'

# Create folder if it doesn’t exist
if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
    print("✅ ZIP file extracted!")

# 📂 2. Load the CSV file from extracted folder
# Auto-detect the first .csv file in the extracted folder
csv_file = None
for file in os.listdir(extract_dir):
    if file.endswith('.csv'):
        csv_file = os.path.join(extract_dir, file)
        break

if csv_file is None:
    raise FileNotFoundError("⚠️ No CSV file found inside ZIP!")

df = pd.read_csv(csv_file)
print("📊 Dataset Preview:")
print(df.head())

# 🧼 3. Prepare the data
required_cols = ['square_feet', 'bedrooms', 'bathrooms', 'price']
if not all(col in df.columns for col in required_cols):
    raise ValueError("⚠️ Your dataset must contain the columns: square_feet, bedrooms, bathrooms, price")

X = df[['square_feet', 'bedrooms', 'bathrooms']]
y = df['price']

# 🔀 4. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 5. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 🔮 6. Predict
y_pred = model.predict(X_test)

# ✅ 7. Evaluate
print("\n📈 Evaluation Results:")
print("📊 Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("📉 R² Score:", r2_score(y_test, y_pred))
print("💰 Intercept:", model.intercept_)
print("📌 Coefficients:", model.coef_)

# 📊 8. Visualize Results
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.grid(True)
plt.tight_layout()
plt.show()
