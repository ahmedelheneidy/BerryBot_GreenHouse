import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('dataset.csv')

# Encode categorical variables
label_encoder = LabelEncoder()
data['Air Quality'] = label_encoder.fit_transform(data['Air Quality'])
data['Shade Control'] = label_encoder.fit_transform(data['Shade Control'])
data['LED Control'] = label_encoder.fit_transform(data['LED Control'])
data['Air Cooler Control'] = label_encoder.fit_transform(data['Air Cooler Control'])
data['Ventilation Control'] = label_encoder.fit_transform(data['Ventilation Control'])

# Split the dataset into input (X) and output (y) variables
X = data[['Indoor Temperature', 'Indoor Humidity', 'Outdoor Temperature', 'Outdoor Humidity',
          'Indoor Light Intensity', 'Outdoor Anemometer', 'Outdoor Light Intensity',
          'Air Quality', 'Soil Moisture']]
y = data[['Shade Duration', 'Air Cooler Duration', 'Ventilation Duration', 'LED Duration']]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train separate RandomForestRegressor models for each output control
models = {}
for column in y.columns:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train[column])
    models[column] = model

# Evaluate the models on the testing set
for column, model in models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test[column], y_pred)
    print(f"Mean Squared Error for {column}: {mse}")
