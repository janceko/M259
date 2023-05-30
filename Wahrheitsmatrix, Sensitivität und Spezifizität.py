import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix

# Laden des Datensatzes
df = pd.read_csv('cleaned_data.csv')

# Aufteilen des Datensatzes in Trainings- und Testdaten
X = df.drop('quality', axis=1)
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisieren und Trainieren des Modells
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Generieren von Vorhersagen
predictions = model.predict(X_test)

# Umwandlung der Vorhersagen in diskrete Klassen (optional)
predictions = predictions.round().astype(int)

# Berechnung der Confusion Matrix
cm = confusion_matrix(y_test, predictions)
print('Confusion Matrix:')
print(cm)

# Berechnung von Sensitivität und Spezifität
true_positive = cm[1, 1]
false_positive = cm[0, 1]
true_negative = cm[0, 0]
false_negative = cm[1, 0]

sensitivity = true_positive / (true_positive + false_negative)
specificity = true_negative / (true_negative + false_positive)

print('Sensitivity:', sensitivity)
print('Specificity:', specificity)