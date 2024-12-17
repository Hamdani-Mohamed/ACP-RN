import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler # type: ignore
import matplotlib.pyplot as plt

# 1. Chargement des données depuis un fichier CSV
# Remplacez le chemin par le fichier de votre choix
data = pd.read_csv("C:\\Users\\Sidou\\Downloads\\ionosphere.arff.csv")

# Séparation des caractéristiques (features) et des étiquettes (labels)
X = data.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière pour les features
y = data.iloc[:, -1].values   # La dernière colonne pour les étiquettes

# 2. Encodage des étiquettes (si elles sont sous forme de texte)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y) 
y_categorical = to_categorical(y_encoded)  # Conversion en format catégorique (one-hot encoding)

# 3. Normalisation des caractéristiques
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)  # Normalisation des données

# 4. Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_categorical, test_size=0.2, random_state=42)

# la architecture 
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer='l2'),  
    Dropout(0.3),  
    Dense(32, activation='relu', kernel_regularizer='l2'),  
    Dropout(0.3),
    Dense(16, activation='relu', kernel_regularizer='l2'), 
    Dense(y_train.shape[1], activation='softmax') 
])

#  Compilation du modèle avec un optimiseur adapté
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#  Callbacks 
def scheduler(epoch, lr):
    if epoch < 1500:
        return lr  
    else:
        return lr * 0.90 

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr_scheduler = LearningRateScheduler(scheduler, verbose=1)

# Entraînement
print("Training the model...")
history = model.fit(X_train, y_train, epochs=1500, batch_size=8, verbose=1, validation_split=0.1, callbacks=[early_stopping, lr_scheduler])

#  Évaluation du modèle sur l'ensemble de test
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
