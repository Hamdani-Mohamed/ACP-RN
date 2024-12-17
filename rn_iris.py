import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler # type: ignore
import matplotlib.pyplot as plt

# 1. Chargement des données
data = pd.read_csv("C:/Users/Sidou/Downloads/iris.arff.csv")

#Séparation des caractéristiques (features) et des étiquettes (labels)
X = data.iloc[:, :-1].values 
y = data.iloc[:, -1].values   

#Encodage des étiquettes
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  
y_categorical = to_categorical(y_encoded)  

#Normalisation des caractéristiques
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)  

#Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_categorical, test_size=0.2, random_state=42)

#  la architecture 
model = Sequential([
    Dense(8, activation='relu', input_shape=(X_train.shape[1],)),  
    Dense(4, activation='relu'), 
    Dense(y_train.shape[1], activation='softmax') 
])

# 6. Compilation du modèle avec un taux d'apprentissage modéré
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#  Callbacks
def scheduler(epoch, lr):
    if epoch < 100:
        return lr 
    else:
        return lr * 0.95  

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(scheduler)

# Entraînement 
print("Training the model...")
history = model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, validation_split=0.1, callbacks=[early_stopping, lr_scheduler])

# Évaluation du modèle sur l'ensemble de test
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

