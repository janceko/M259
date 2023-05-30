import pandas as pd
import matplotlib.pyplot as plt

# Laden des Datensatzes
df = pd.read_csv('winequality-red.csv')

# 2.1 Datenfeld für Vorhersagen
prediction_field = 'quality'

# 2.2 Statistische Informationen für jedes Feld
statistics = df.describe()

# 2.3 Erstellung einer Grafik (Histogramm)
plt.hist(df['alcohol'], bins=10, edgecolor='black')
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