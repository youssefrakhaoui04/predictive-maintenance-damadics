# %% [1] Chargement des données et inspection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Chargement du fichier CSV fourni
df = pd.read_csv("damadics_actuator_sample.csv")
print("Dimensions du dataset :", df.shape)
print("\nAperçu des premières lignes :")
print(df.head())

# %% [2] Visualisation des signaux capteurs et dérive
plt.figure(figsize=(11, 4.5))
plt.plot(
    df["timestamp_h"],
    df["consigne"],
    label="Consigne (mm)",
    color="black",
    linestyle="--",
    alpha=0.7,
)
plt.plot(
    df["timestamp_h"],
    df["position"],
    label="Position mesurée tige (mm)",
    color="#1f77b4",
    alpha=0.85,
)
plt.axvline(
    x=302,
    color="orange",
    linestyle=":",
    linewidth=2,
    label="Fenêtre alerte anticipée (< 48h)",
)
plt.axvline(
    x=350,
    color="red",
    linestyle="-",
    linewidth=1.8,
    label="Défaillance critique (t = 350h)",
)
plt.title("Réponse dynamique de l'actionneur pneumatique (Benchmark DAMADICS)")
plt.xlabel("Temps (heures)")
plt.ylabel("Déplacement (mm)")
plt.legend(loc="lower left")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# %% [3] Ingénierie des caractéristiques physiques (Feature Engineering)
# Résidu algébrique direct : consigne vs mesure
df["err_position"] = df["consigne"] - df["position"]

# Dérivées temporelles (dynamique de vitesse et variation de pression)
df["d_position"] = df["position"].diff().fillna(0)
df["d_pression"] = df["pression"].diff().fillna(0)

# Statistiques glissantes pour filtrer le bruit haute fréquence
df["rolling_mean_err"] = (
    df["err_position"].rolling(window=10, min_periods=1).mean()
)
df["rolling_std_err"] = (
    df["err_position"].rolling(window=10, min_periods=1).std().fillna(0)
)

features = [
    "consigne",
    "position",
    "pression",
    "debit",
    "err_position",
    "d_position",
    "d_pression",
    "rolling_mean_err",
    "rolling_std_err",
]

X = df[features]
y = df["fault_label"]

# %% [4] Partitionnement, normalisation et entraînement du modèle
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Classifieur Random Forest
clf = RandomForestClassifier(
    n_estimators=120, max_depth=10, min_samples_split=5, random_state=42
)
clf.fit(X_train_scaled, y_train)

# Inférence et évaluation
y_pred = clf.predict(X_test_scaled)

print("\n--- Rapport de performance ---")
print(
    classification_report(y_test, y_pred, target_names=["Sain", "Alerte < 48h"])
)

# %% [5] Visualisation : Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Sain", "Alerte < 48h"],
    yticklabels=["Sain", "Alerte < 48h"],
)
plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.title("Matrice de confusion - Anticipation 48h")
plt.tight_layout()
plt.show()

# %% [6] Analyse d'explicabilité : Importance des capteurs
importances = pd.Series(clf.feature_importances_, index=features).sort_values(
    ascending=True
)

plt.figure(figsize=(7, 4))
importances.plot(kind="barh", color="#2ca02c")
plt.title("Contribution relative des indicateurs physiques")
plt.xlabel("Score d'importance Gini")
plt.tight_layout()
plt.show()