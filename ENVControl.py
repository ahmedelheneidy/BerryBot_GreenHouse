import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
data = pd.read_csv('dataset.csv')

# Split data into features (inputs) and target (outputs)
X = data[['Indoor Temperature', 'Indoor Humidity', 'Outdoor Temperature', 'Outdoor Humidity', 'Indoor Light Intensity',  'Outdoor Light Intensity', 'Air Quality', 'Soil Moisture']]
y_shade_control = data['Shade Control']
y_irrigation_amount = data['Irrigation Amount']
y_fertilization_amount = data['Fertilization Amount']
y_air_cooler_control = data['Air Cooler Control']
y_ventilation_control = data['Ventilation Control']
y_led_control = data['LED Control']
y_shade_duration = data['Shade Duration']
y_air_cooler_duration = data['Air Cooler Duration']
y_ventilation_duration = data['Ventilation Duration']
y_led_duration = data['LED Duration']

# Encode categorical variable 'Air Quality'
X_encoded = pd.get_dummies(X, columns=['Air Quality'])

# Split data into training and testing sets
X_train, X_test, y_train_shade_control, y_test_shade_control = train_test_split(X_encoded, y_shade_control, test_size=0.2, random_state=42)
X_train, X_test, y_train_irrigation_amount, y_test_irrigation_amount = train_test_split(X_encoded, y_irrigation_amount, test_size=0.2, random_state=42)
X_train, X_test, y_train_fertilization_amount, y_test_fertilization_amount = train_test_split(X_encoded, y_fertilization_amount, test_size=0.2, random_state=42)
X_train, X_test, y_train_air_cooler_control, y_test_air_cooler_control = train_test_split(X_encoded, y_air_cooler_control, test_size=0.2, random_state=42)
X_train, X_test, y_train_ventilation_control, y_test_ventilation_control = train_test_split(X_encoded, y_ventilation_control, test_size=0.2, random_state=42)
X_train, X_test, y_train_led_control, y_test_led_control = train_test_split(X_encoded, y_led_control, test_size=0.2, random_state=42)
X_train, X_test, y_train_shade_duration, y_test_shade_duration = train_test_split(X_encoded, y_shade_duration, test_size=0.2, random_state=42)
X_train, X_test, y_train_air_cooler_duration, y_test_air_cooler_duration = train_test_split(X_encoded, y_air_cooler_duration, test_size=0.2, random_state=42)
X_train, X_test, y_train_ventilation_duration, y_test_ventilation_duration = train_test_split(X_encoded, y_ventilation_duration, test_size=0.2, random_state=42)
X_train, X_test, y_train_led_duration, y_test_led_duration = train_test_split(X_encoded, y_led_duration, test_size=0.2, random_state=42)

# Train separate Random Forest models for each target variable
model_shade_control = RandomForestRegressor(n_estimators=100, random_state=42)
model_shade_control.fit(X_train, y_train_shade_control)

model_irrigation_amount = RandomForestRegressor(n_estimators=100, random_state=42)
model_irrigation_amount.fit(X_train, y_train_irrigation_amount)

model_fertilization_amount = RandomForestRegressor(n_estimators=100, random_state=42)
model_fertilization_amount.fit(X_train, y_train_fertilization_amount)

model_air_cooler_control = RandomForestRegressor(n_estimators=100, random_state=42)
model_air_cooler_control.fit(X_train, y_train_air_cooler_control)

model_ventilation_control = RandomForestRegressor(n_estimators=100, random_state=42)
model_ventilation_control.fit(X_train, y_train_ventilation_control)

model_led_control = RandomForestRegressor(n_estimators=100, random_state=42)
model_led_control.fit(X_train, y_train_led_control)

model_shade_duration = RandomForestRegressor(n_estimators=100, random_state=42)
model_shade_duration.fit(X_train, y_train_shade_duration)

model_air_cooler_duration = RandomForestRegressor(n_estimators=100, random_state=42)
model_air_cooler_duration.fit(X_train, y_train_air_cooler_duration)

model_ventilation_duration = RandomForestRegressor(n_estimators=100, random_state=42)
model_ventilation_duration.fit(X_train, y_train_ventilation_duration)

model_led_duration = RandomForestRegressor(n_estimators=100, random_state=42)
model_led_duration.fit(X_train, y_train_led_duration)

# Make predictions for each target variable
predictions_shade_control = model_shade_control.predict(X_test)
predictions_irrigation_amount = model_irrigation_amount.predict(X_test)
predictions_fertilization_amount = model_fertilization_amount.predict(X_test)
predictions_air_cooler_control = model_air_cooler_control.predict(X_test)
predictions_ventilation_control = model_ventilation_control.predict(X_test)
predictions_led_control = model_led_control.predict(X_test)
predictions_shade_duration = model_shade_duration.predict(X_test)
predictions_air_cooler_duration = model_air_cooler_duration.predict(X_test)
predictions_ventilation_duration = model_ventilation_duration.predict(X_test)
predictions_led_duration = model_led_duration.predict(X_test)

# Print or use predictions as needed
print("Predictions for Shade Control:", predictions_shade_control)
print("Predictions for Irrigation Amount:", predictions_irrigation_amount)
print("Predictions for Fertilization Amount:", predictions_fertilization_amount)
print("Predictions for Air Cooler Control:", predictions_air_cooler_control)
print("Predictions for Ventilation Control:", predictions_ventilation_control)
print("Predictions for LED Control:", predictions_led_control)
print("Predictions for Shade Duration:", predictions_shade_duration)
print("Predictions for Air Cooler Duration:", predictions_air_cooler_duration)
print("Predictions for Ventilation Duration:", predictions_ventilation_duration)
print("Predictions for LED Duration:", predictions_led_duration)

import matplotlib.pyplot as plt

# Function to plot feature importances
def plot_feature_importances(model, feature_names, target_name):
    feature_importances = model.feature_importances_
    sorted_idx = feature_importances.argsort()
    y_ticks = range(len(sorted_idx))
    fig, ax = plt.subplots()
    ax.barh(y_ticks, feature_importances[sorted_idx])
    ax.set_yticklabels(feature_names[sorted_idx])
    ax.set_yticks(y_ticks)
    ax.set_title(f'Feature Importances for {target_name}')
    plt.show()

# Plot feature importances for each target variable
plot_feature_importances(model_shade_control, X_train.columns, 'Shade Control')
plot_feature_importances(model_irrigation_amount, X_train.columns, 'Irrigation Amount')
plot_feature_importances(model_fertilization_amount, X_train.columns, 'Fertilization Amount')
plot_feature_importances(model_air_cooler_control, X_train.columns, 'Air Cooler Control')
plot_feature_importances(model_ventilation_control, X_train.columns, 'Ventilation Control')
plot_feature_importances(model_led_control, X_train.columns, 'LED Control')
plot_feature_importances(model_shade_duration, X_train.columns, 'Shade Duration')
plot_feature_importances(model_air_cooler_duration, X_train.columns, 'Air Cooler Duration')
plot_feature_importances(model_ventilation_duration, X_train.columns, 'Ventilation Duration')
plot_feature_importances(model_led_duration, X_train.columns, 'LED Duration')

