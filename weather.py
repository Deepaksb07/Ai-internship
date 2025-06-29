import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load your dataset
df = pd.read_csv('weather.csv')  # Replace with actual dataset path

# Check for null values
df = df.dropna()

# Optional: Remove duplicates
df = df.drop_duplicates()

# Let's assume we're predicting 'Temperature' based on other factors
X = df[['Humidity', 'WindSpeed']]  # independent variables
y = df['Temperature']              # dependent variable

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# Predict temperature with new values
new_data = pd.DataFrame({'Humidity': [60], 'WindSpeed': [15]})
future_temp = model.predict(new_data)
print("Predicted Temperature:", future_temp[0])
