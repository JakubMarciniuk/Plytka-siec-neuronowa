import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss

# ==============================================================================
# KROK 1: WCZYTANIE I PRZYGOTOWANIE DANYCH (Wariant >4.0)
# ==============================================================================
print("1. Pobieranie i przetwarzanie danych...")

url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
try:
    data = pd.read_csv(url)
except:
    print("Błąd: Brak połączenia z internetem lub pliku csv.")
    exit()

# Wybór cech (X) i celu (y)
X = data[['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']]
y = data['Survived']

# Definicja transformacji
numeric_features = ['Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['Pclass', 'Sex']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Transformacja danych PRZED pętlą (wymagane dla partial_fit)
X_processed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Pobranie nazw cech do wykresów (czytelne etykiety)
feature_names = preprocessor.get_feature_names_out()
feature_names = [name.replace('num__', '').replace('cat__', '') for name in feature_names]

# ==============================================================================
# KROK 2: TRENING Z RĘCZNYM EARLY STOPPING (Wymogi 3.0 i 4.0)
# ==============================================================================
print("2. Inicjalizacja modelu i trening...")

# Konfiguracja na 4.0 (Adam, Mini-batch) i >4.0 (Sieć płytka/dwuwarstwowa)
clf = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=32,
    random_state=42,
    warm_start=True,  # Pozwala na douczanie w pętli
    max_iter=1  # Jedna iteracja na wywołanie partial_fit
)

# Listy na historię
loss_train_history = []
loss_test_history = []
class_err_train_history = []
class_err_test_history = []

epochs = 100
classes = np.unique(y)
patience = 5  # Ile epok czekać na poprawę
best_loss = np.inf
no_change_counter = 0
real_epochs = 0

print(f"-> Rozpoczynam pętlę uczenia (max {epochs} epok)...")

for i in range(epochs):
    real_epochs += 1
    # Krok uczenia (Mini-batch realizowany wewnętrznie przez partial_fit z batch_size=32)
    clf.partial_fit(X_train, y_train, classes=classes)

    # Obliczanie metryk
    loss_train = log_loss(y_train, clf.predict_proba(X_train))
    loss_test = log_loss(y_test, clf.predict_proba(X_test))

    acc_train = clf.score(X_train, y_train)
    acc_test = clf.score(X_test, y_test)

    # Zapis do historii
    loss_train_history.append(loss_train)
    loss_test_history.append(loss_test)
    class_err_train_history.append(1 - acc_train)
    class_err_test_history.append(1 - acc_test)

    # Ręczny Early Stopping (Wymóg 3.0: opcja szybszego kończenia)
    if loss_test < best_loss:
        best_loss = loss_test
        no_change_counter = 0
    else:
        no_change_counter += 1

    if no_change_counter >= patience:
        print(f"   [Early Stopping] Zatrzymano uczenie w epoce {real_epochs}. Brak poprawy na zbiorze testowym.")
        break

print(f"-> Trening zakończony.")

# ==============================================================================
# KROK 3: GENEROWANIE WYKRESÓW (Dowody zaliczenia)
# ==============================================================================
print("3. Generowanie wykresów...")

# Wykres 1: Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(loss_train_history, label='Zbiór Uczący', color='blue')
plt.plot(loss_test_history, label='Zbiór Testowy', color='orange', linestyle='--')
plt.title('Przebieg funkcji kosztu (Log-Loss)')
plt.xlabel('Epoki')
plt.ylabel('Wartość błędu')
plt.legend()
plt.grid(True)
plt.savefig('error_loss_plot.png')
print("   -> Zapisano: error_loss_plot.png")

# Wykres 2: Błąd Klasyfikacji
plt.figure(figsize=(10, 5))
plt.plot(class_err_train_history, label='Zbiór Uczący', color='blue')
plt.plot(class_err_test_history, label='Zbiór Testowy', color='orange', linestyle='--')
plt.title('Błąd Klasyfikacji (1 - Accuracy)')
plt.xlabel('Epoki')
plt.ylabel('Błąd (0.0 - 1.0)')
plt.legend()
plt.grid(True)
plt.savefig('error_class_plot.png')
print("   -> Zapisano: error_class_plot.png")

# Wykres 3: Wagi Warstwa 1 (Czytelna Heatmapa)
plt.figure(figsize=(12, 8))
# Transpozycja wag, aby cechy były na osi Y
sns.heatmap(clf.coefs_[0], cmap='RdBu', center=0, yticklabels=feature_names, xticklabels=False)
plt.title('Mapa Wpływu Cech na Warstwę Ukrytą\n(Czerwony=Stymulacja, Niebieski=Hamowanie)')
plt.ylabel('Cechy Pasażera')
plt.xlabel('Neurony w Warstwie Ukrytej')
plt.tight_layout()
plt.savefig('weights_layer1_readable.png')
print("   -> Zapisano: weights_layer1_readable.png")

# Wykres 4: Wagi Warstwa 2 (Słupki)
plt.figure(figsize=(12, 6))
plt.bar(range(len(clf.coefs_[1])), clf.coefs_[1].flatten(), color='purple', alpha=0.7)
plt.title('Wagi Decyzyjne (Warstwa Ukryta -> Wyjście)')
plt.xlabel('Numer Neuronu Ukrytego')
plt.ylabel('Siła wpływu na decyzję')
plt.grid(True, alpha=0.3)
plt.savefig('weights_layer2_readable.png')
print("   -> Zapisano: weights_layer2_readable.png")

# ==============================================================================
# KROK 4: WYNIKI KOŃCOWE
# ==============================================================================
print("\n--- RAPORT KOŃCOWY ---")
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Dokładność (Accuracy): {acc * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Nie przeżył', 'Przeżył']))
print(f"Liczba epok: {real_epochs}")