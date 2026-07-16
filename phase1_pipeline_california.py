# Pipeline California Housing: load → split → scale → train → evaluate
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Choix : split AVANT scaler.fit(X_train).
# Justification : si on fit le scaler sur X entier, les statistiques
# (mean, std) sont calculées en incluant val et test. Le modèle "voit"
# indirectement des informations sur des données qu'il est censé
# découvrir uniquement lors de l'évaluation (data leakage).
# En fittant uniquement sur X_train, on simule la vraie situation :
# au moment du déploiement, le scaler ne connaît que les données
# d'entraînement, jamais les nouvelles données à venir.

# Charger le dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Premier split : train+val (80%) / test (20%)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Second split : train (80% de trainval) / val (20% de trainval)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=42
)

# StandardScaler fitté UNIQUEMENT sur X_train
scaler = StandardScaler()
scaler.fit(X_train)

# Transform sur les trois sous-ensembles
X_train_norm = scaler.transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)

# Shapes
print("Shapes :")
print(f"  X_train_norm : {X_train_norm.shape}")
print(f"  X_val_norm   : {X_val_norm.shape}")
print(f"  X_test_norm  : {X_test_norm.shape}")

# Stats descriptives (doivent être ~0 pour mean, ~1 pour std sur le train)
print("\nStats descriptives de X_train_norm :")
print(f"  mean par feature : {X_train_norm.mean(axis=0)}")
print(f"  std  par feature : {X_train_norm.std(axis=0)}")

# feature_names
print(f"\nFeature names ({len(housing.feature_names)}) :")
print(housing.feature_names)
assert len(housing.feature_names) == 8, "Il devrait y avoir 8 features"
print("\n✅ Pipeline California Housing terminé avec succès.")
