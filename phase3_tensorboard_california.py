# Phase 3 : Diagnostic TensorBoard sur California Housing (deux runs : norm vs raw)
import datetime
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# --- Chargement et préparation des données ---
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


def build_regression_model(input_dim):
    model = keras.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_with_tensorboard(X_train, y_train, X_val, y_val, run_name, epochs=100):
    """Entraîne un modèle de régression avec un callback TensorBoard horodaté."""
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    log_dir = f"logs/fit/{run_name}_{timestamp}"

    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model = build_regression_model(input_dim=8)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[tb_callback],
        verbose=0
    )

    print(f"Run '{run_name}' terminé. Logs dans {log_dir}")
    return model, history


# --- Run 1 : données normalisées ---
model_norm, history_norm = train_with_tensorboard(
    X_train_norm, y_train, X_val_norm, y_val, run_name="california_norm"
)

# --- Run 2 : données brutes (non normalisées) ---
model_raw, history_raw = train_with_tensorboard(
    X_train, y_train, X_val, y_val, run_name="california_raw"
)

# --- Interprétation (à compléter après observation dans TensorBoard) ---
# TODO : indiquer ici dans quelle situation (a), (b) ou (c) vous tombez,
# et votre interprétation, une fois les courbes observées dans TensorBoard.

# --- Interprétation ---
# Situation observée : (a) pour california_norm — train_loss et val_loss
# descendent ensemble de façon saine, se stabilisent après quelques dizaines
# d'epochs, sans écart significatif entre les deux courbes.
# Pour california_raw : la loss reste très élevée et instable/plafonne,
# car les features à grande échelle (Latitude/Longitude ~37-38) dominent
# les gradients et empêchent le modèle d'apprendre correctement les
# features à petite échelle (AveRooms ~5-6). Même architecture, même
# optimizer, même durée : seule la normalisation change, et c'est elle
# qui explique tout l'écart entre les deux runs. Ceci justifie le
# preprocessing (StandardScaler) mis en place en Phase 1.
