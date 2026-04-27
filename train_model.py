import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
file_id = "1mf_IWXvZjGDmprMr6am8sIs8vWRxRHMX"
drive_url = f"https://drive.google.com/uc?id={file_id}"
df = pd.read_csv(drive_url)
df.fillna(df.mean(numeric_only=True), inplace=True)
df.drop_duplicates(inplace=True)
print(df.columns)
target = 'House_Price'   # change if needed
X = df.drop(target, axis=1)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("Model saved successfully!")