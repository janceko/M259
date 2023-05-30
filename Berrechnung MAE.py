import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Laden des Datensatzes
df = pd.read_csv('cleaned_data.csv')

# 3.1 Teilen Sie den Datensatz in einen Test- und einen Trainingsdatensatz auf
X = df.drop('quality', axis=1)
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3.2 Wählen Sie einen geeigneten Algorithmus aus und trainieren Sie das Modell
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 3.3 Generieren Sie Vorhersagen
predictions = model.predict(X_test)

# 4.2 Berechnung des MAE
mae = mean_absolute_error(y_test, predictions)
print('MAE:', mae)