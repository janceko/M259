import pandas as pd

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