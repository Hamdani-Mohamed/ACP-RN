 # Importer les bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Charger les données
file_path = "C:/Users/Sidou/Downloads/iris.arff.csv"   # Remplacez par le chemin de votre fichier
data = pd.read_csv(file_path)

# Séparer les caractéristiques (X) et la cible (y)
X = data.drop(columns=['class'])  # Assurez-vous que la colonne cible s'appelle 'class'
y = data['class']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Étape 1 : Centrage et réduction des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Vérification des moyennes et variances
print("Moyennes après centrage :", X_train_scaled.mean(axis=0))
print("Variances après réduction :", X_train_scaled.var(axis=0))

# Étape 2 : Calcul de la matrice de covariance
cov_matrix = np.cov(X_train_scaled, rowvar=False)
print("\nMatrice de covariance :\n", cov_matrix)

# Étape 3 : Calcul des valeurs propres et des vecteurs propres
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\nValeurs propres :\n", eigenvalues)
print("\nVecteurs propres :\n", eigenvectors)

# Étape 4 : Tri des composantes principales par variance expliquée
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

print("\nValeurs propres triées :\n", eigenvalues)

# Calcul de la variance expliquée
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
cumulative_variance = np.cumsum(explained_variance_ratio)

print("\nVariance expliquée par chaque composante principale :\n", explained_variance_ratio)
print("\nVariance expliquée cumulée :\n", cumulative_variance)

# Étape 5 : Projection sur les composantes principales
n_components = 2  # Visualisation en 2D
X_train_pca = np.dot(X_train_scaled, eigenvectors[:, :n_components])
X_test_pca = np.dot(X_test_scaled, eigenvectors[:, :n_components])

# Étape 6 : Visualisation des composantes principales
plt.figure(figsize=(10, 7))

# Définir les couleurs pour chaque classe
class_labels = y_train.unique()
colors = ['green', 'red']  # Adapter les couleurs au nombre de classes

# Tracer les données projetées
for label, color in zip(class_labels, colors):
    plt.scatter(
        X_train_pca[y_train == label, 0],
        X_train_pca[y_train == label, 1],
        label=f"Classe {label} ({color})",
        c=color,
        alpha=0.6
    )

# Ajouter un titre, des étiquettes et une légende
plt.title("Projection sur les deux premières composantes principales")
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.legend(title="Signification des classes")
plt.grid(True)

# Ajouter une annotation expliquant les couleurs
plt.annotate(
    "Légende des couleurs :\n- Vert : Classe 0\n- Rouge : Classe 1",
    xy=(1, 0.5), xycoords='axes fraction',
    fontsize=10, ha='right', va='center',
    bbox=dict(boxstyle="round", fc="wheat", alpha=0.5)
)

plt.show()