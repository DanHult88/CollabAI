# weather_predictor.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Läs in datan
data = pd.read_csv('weatherAUS.csv')

# 2. Välj ut de kolumner vi ska använda
data = data[['MinTemp', 'MaxTemp', 'Rainfall',
             'WindSpeed9am', 'WindSpeed3pm',
             'Humidity9am', 'Humidity3pm',
             'RainToday', 'RainTomorrow']]

# 3. Ta bort rader med saknade värden
data = data.dropna()

# 4. Konvertera 'Yes'/'No' till 1/0
data['RainToday'] = data['RainToday'].map({'Yes': 1, 'No': 0})
data['RainTomorrow'] = data['RainTomorrow'].map({'Yes': 1, 'No': 0})

# 5. Dela upp i features (X) och label (y)
X = data.drop('RainTomorrow', axis=1)
y = data['RainTomorrow']

# 6. Dela upp i träning/test (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 7. Skapa och träna en Random Forest-modell
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Gör förutsägelser
y_pred = model.predict(X_test)

# 9. Skriv ut resultat
print("Utvärdering:\n")
print(classification_report(y_test, y_pred))
