# Phase 2 : Baseline PMC régression sur California Housing
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# --- Chargement et préparation des données (repris de la Phase 1) ---
housing = fetch_california_housing()
X, y = housing.data, housing.target

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=42
)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_norm = scaler.transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)


def build_regression_model(input_dim):
    # Pourquoi PAS de sigmoid en sortie :
    # sigmoid écrase toute sortie dans l'intervalle [0, 1]. Or nos cibles
    # (prix médian en centaines de milliers de $) vont jusqu'à ~5.
    # Avec une sigmoid, le modèle ne pourrait JAMAIS prédire au-delà de 1,
    # quel que soit l'entraînement : la loss descendrait un peu puis
    # plafonnerait, avec une erreur structurelle incompressible sur les
    # valeurs hautes. Pour une régression à valeur continue non bornée,
    # la sortie doit être linéaire (pas d'activation).
    model = keras.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))  # pas d'activation : régression

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


model = build_regression_model(input_dim=8)
model.summary()

history = model.fit(
    X_train_norm, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val_norm, y_val),
    verbose=1
)

test_loss, test_mae = model.evaluate(X_test_norm, y_test, verbose=0)
print(f"MAE test : {test_mae:.4f} (en centaines de milliers de $)")
