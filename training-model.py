import numpy as np
import random
import joblib  # Zum Speichern und Laden des Modells
import os

from sklearn.linear_model import LinearRegression

# Einfache Funktion zur Simulation beschädigter Daten
def generate_data(num_samples=100):
    data = []
    for i in range(num_samples):
        # Generiere zufällige "intakte" Daten
        data.append(i + random.gauss(0, 1))  # zufälliges Rauschen hinzufügen
    return np.array(data)

# Funktion zum Erstellen eines Regressionsmodells zur Vorhersage beschädigter Werte
def train_recovery_model(data):
    # Umwandlung von Daten in X (Merkmale) und Y (Zielwerte)
    X = np.array(range(len(data))).reshape(-1, 1)  # Indizes als Merkmale
    y = data  # Beschädigte Daten als Zielwerte
    
    model = LinearRegression()
    model.fit(X, y)  # Modell trainieren
    return model

# Funktion zur Wiederherstellung fehlender Werte
def recover_data(model, num_samples=100):
    recovered_data = []
    for i in range(num_samples):
        # Vorhersage von Werten an fehlenden Stellen
        recovered_data.append(model.predict([[i]])[0])
    return np.array(recovered_data)

# Funktion zum Auswählen eines Modells
def select_or_train_model():
    model_directory = "./models"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    
    model_files = [f for f in os.listdir(model_directory) if f.endswith('.pkl')]
    
    if model_files:
        print("Vorhandene Modelle:")
        for i, model_file in enumerate(model_files):
            print(f"{i + 1}. {model_file}")
        
        user_choice = input(f"Bitte wähle ein Modell (1-{len(model_files)}) oder gib 'neu' ein, um ein neues Modell zu erstellen: ").strip()
        
        if user_choice.lower() == 'neu':
            return None  # Zeigt an, dass ein neues Modell trainiert werden soll
        else:
            try:
                selected_model_file = model_files[int(user_choice) - 1]
                model_path = os.path.join(model_directory, selected_model_file)
                model = joblib.load(model_path)
                print(f"Modell '{selected_model_file}' erfolgreich geladen!")
                return model
            except (ValueError, IndexError):
                print("Ungültige Auswahl, ein neues Modell wird erstellt.")
                return None
    else:
        print("Kein vorhandenes Modell gefunden, ein neues Modell wird erstellt.")
        return None

# Simulation von beschädigten Daten
original_data = generate_data(100)
damaged_data = original_data.copy()

# Zerstöre einige Datenpunkte (Simulation)
for i in range(30, 50):
    damaged_data[i] = np.nan  # Setze einige Werte auf NaN, um beschädigte Daten zu simulieren

# Modell laden oder neues Modell erstellen
model = select_or_train_model()

if model is None:
    # Falls kein Modell gefunden wurde, trainiere ein neues Modell
    model = train_recovery_model(damaged_data)
    model_filename = f"./models/recovery_model_{len(os.listdir('./models')) + 1}.pkl"
    joblib.dump(model, model_filename)  # Speichern des neuen Modells
    print(f"Neues Modell gespeichert als {model_filename}.")

# Wiederherstellung der beschädigten Daten
recovered_data = recover_data(model)

# Ausgabe der Ergebnisse
print("Original Data (first 10 values):", original_data[:10])
print("Damaged Data (with NaN):", damaged_data[30:50])
print("Recovered Data:", recovered_data[30:50])

# Optional: Speichern der wiederhergestellten Daten in einer Datei
with open("recovered_data.txt", "w") as f:
    for value in recovered_data:
        f.write(f"{value}\n")

# Optional: Visualisierung der wiederhergestellten Daten
import matplotlib.pyplot as plt

plt.plot(range(len(original_data)), original_data, label="Original Data")
plt.plot(range(len(damaged_data)), damaged_data, label="Damaged Data", linestyle='dashed')
plt.plot(range(len(recovered_data)), recovered_data, label="Recovered Data", linestyle='dotted')
plt.legend()
plt.show()
