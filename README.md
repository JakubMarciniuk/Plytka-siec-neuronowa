# Podstawy Sieci Neuronowych — Klasyfikacja Pasażerów Titanica

Projekt realizowany w ramach przedmiotu **Informatyczne Systemy Automatyki**. Skupia się na implementacji płytkiej sieci neuronowej typu Feed-Forward do rozwiązywania problemu klasyfikacji binarnej na zbiorze danych Titanic (Kaggle).

## 🚀 Cel Projektu
Głównym zadaniem było stworzenie modelu zdolnego do przewidywania przeżywalności pasażerów na podstawie cech takich jak klasa biletu, płeć, wiek czy opłata za bilet. Projekt realizuje wymagania dla wariantu powyżej 4.0, uwzględniając zaawansowane techniki optymalizacji i przetwarzania danych.

## 🛠️ Architektura Modelu
Zastosowano sieć neuronową o strukturze dwuwarstwowej:
* **Warstwa wejściowa:** 9 neuronów (po transformacji danych kategorycznych).
* **Warstwa ukryta:** 100 neuronów z funkcją aktywacji **ReLU**.
* **Warstwa wyjściowa:** 1 neuron z funkcją **Sigmoid** (klasyfikacja binarna).
* **Funkcja straty:** Log-Loss (Binary Cross-Entropy).

## 📊 Metody Uczenia i Optymalizacja
Model wykorzystuje zaawansowany zestaw narzędzi zapewniających stabilność i szybkość uczenia:
* **Optymalizator Adam:** Zastosowanie momentum oraz adaptacyjnego współczynnika uczenia.
* **Trening Mini-batch:** Aktualizacja wag w paczkach po 32 próbki.
* **Early Stopping:** Autorski mechanizm przerywający proces uczenia po 5 epokach bez poprawy błędu walidacyjnego, co zapobiega overfittingowi.

## 🧹 Przygotowanie Danych (Preprocessing)
Kluczowym elementem projektu jest zaawansowany potok przetwarzania danych (Pipeline):
1. **Imputacja:** Braki w danych numerycznych uzupełniono medianą, a w kategorycznych modą.
2. **Kodowanie:** Zastosowano One-Hot Encoding dla zmiennych kategorycznych (Płeć, Klasa biletu).
3. **Standaryzacja:** Cechy numeryczne zostały przeskalowane przy użyciu StandardScaler (Z-score normalization).

## 📈 Wyniki i Wnioski
Model osiągnął dokładność (Accuracy) na poziomie **75-80%**. 

Analiza wag modelu potwierdziła logiczną poprawność procesu uczenia — sieć nadała największe znaczenie cechom `Sex_female` (płeć żeńska) oraz `Pclass_1` (pierwsza klasa), co koreluje z danymi historycznymi dotyczącymi priorytetów podczas ewakuacji.
