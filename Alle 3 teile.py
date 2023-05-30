import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Laden des Datensatzes
df = pd.read_csv('winequality-red.csv')

# Sortieren des Datensatzes nach einer Spalte (z.B. 'quality')
df = df.sort_values(by='quality')

# Überprüfen auf fehlende Werte
print(df.isnull().sum())

# Entfernen von Zeilen mit fehlenden Werten
df = df.dropna()

# Überprüfen auf Duplikate
print(df.duplicated().sum())

# Entfernen von Duplikaten
df = df.drop_duplicates()

# Umbenennen von Spalten
df = df.rename(columns={'fixed acidity': 'Fixed Acidity', 'volatile acidity': 'Volatile Acidity',
                        'citric acid': 'Citric Acid', 'residual sugar': 'Residual Sugar',
                        'chlorides': 'Chlorides', 'free sulfur dioxide': 'Free Sulfur Dioxide',
                        'total sulfur dioxide': 'Total Sulfur Dioxide', 'density': 'Density',
                        'pH': 'pH', 'sulphates': 'Sulphates', 'alcohol': 'Alcohol', 'quality': 'Quality'})

# Konvertieren von Datentypen
df['Alcohol'] = df['Alcohol'].astype(float)

# Exportieren des bereinigten und sortierten Datensatzes als Excel-Datei
df.to_excel('cleaned_data.xlsx', index=False, float_format="%.2f", header=True)

# 2.1 Datenfeld für Vorhersagen
prediction_field = 'Quality'

# 2.2 Statistische Informationen für jedes Feld
statistics = df.describe()

# 2.3 Erstellung einer Grafik (Histogramm)
plt.hist(df['Alcohol'], bins=10, edgecolor='black')
plt.xlabel('Alcohol')
plt.ylabel('Count')
plt.title('Distribution of Alcohol')
plt.show()

# 2.4 Skalierung eines Datenfelds
# Beispiel: Skalierung des Datenfelds 'pH'
# Hier verwenden wir die Min-Max-Skalierung als Beispiel
min_value = df['pH'].min()
max_value = df['pH'].max()
df['scaled_pH'] = (df['pH'] - min_value) / (max_value - min_value)

# Speichern der bearbeiteten Daten
df.to_csv('processed_data.csv', index=False)

# 3.1 Teilen Sie den Datensatz in einen Test- und einen Trainingsdatensatz auf
X = df.drop(prediction_field, axis=1)
y = df[prediction_field]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3.2 Wählen Sie einen geeigneten Algorithmus aus und trainieren Sie das Modell
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 3.3 Generieren Sie Vorhersagen und überprüfen Sie die Ergebnisse
predictions = model.predict(X_test)
# Hier können Sie Ihre manuelle Überprüfung der Vorhersagen durchführen und die Ergebnisse zusammenfassen