import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Laden des bereinigten Datensatzes
df = pd.read_csv('cleaned_data.csv')

# Extrahieren der Merkmale und der Zielvariable
X = df.drop('quality', axis=1)
y = df['quality']

# Trainieren des Modells auf dem gesamten Datensatz
model = RandomForestRegressor()
model.fit(X, y)

# Extrahieren der Merkmalswichtigkeiten
feature_importances = model.feature_importances_

# Erstellen einer DataFrame mit den Merkmalen und ihren Wichtigkeiten
feature_importances_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sortieren der Merkmale nach Wichtigkeit in absteigender Reihenfolge
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

# Visualisierung der Merkmalswichtigkeiten
plt.barh(feature_importances_df['Feature'], feature_importances_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()

# Anzeigen der top k aussagekräftigsten Merkmale (z.B. top 5)
top_k_features = feature_importances_df.head(5)
print(top_k_features)