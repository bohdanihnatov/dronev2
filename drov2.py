import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Подготовка данных
X = np.random.random((1000, 10))
y = np.random.randint(2, size=(1000, 1))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Построение модели нейронной сети
model = Sequential([
    Dense(32, activation='relu', input_shape=(10,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Компиляция и обучение модели
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_scaled, y, epochs=10, batch_size=32)
