"""
TAREA 11 - SISTEMAS INTELIGENTES I
Modelo CNN-LSTM para clasificación de CIFAR-10
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

# Configuración para CPU AMD Ryzen 7 5700U
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(2)

# Cargar y preparar datos CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalizar
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

print(f"Datos de entrenamiento: {x_train.shape}")
print(f"Datos de prueba: {x_test.shape}")

# Crear modelo CNN-LSTM
model = models.Sequential([
    # Bloque CNN
    layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    
    # Preparar para LSTM
    layers.RepeatVector(8),
    
    # Bloque LSTM
    layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
    layers.LSTM(32, dropout=0.3, recurrent_dropout=0.3),
    layers.Dropout(0.5),
    
    # Clasificador
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compilar
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
]

# Entrenar
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=80,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Evaluar
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Exactitud en prueba: {test_accuracy:.4f}")
print(f"Pérdida en prueba: {test_loss:.4f}")

# Predicciones
predictions = model.predict(x_test, verbose=0)
pred_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# 1. Curvas de entrenamiento
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Exactitud del Modelo')
plt.xlabel('Época')
plt.ylabel('Exactitud')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida del Modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 2. Matriz de confusión
cm = confusion_matrix(true_classes, pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 3. Reporte de clasificación
print("\nReporte de Clasificación:")
report = classification_report(true_classes, pred_classes, target_names=class_names, output_dict=True)
print(classification_report(true_classes, pred_classes, target_names=class_names))

# 4. Exactitud por clase
class_accuracies = []
for i in range(10):
    mask = (true_classes == i)
    acc = np.mean(pred_classes[mask] == true_classes[mask])
    class_accuracies.append(acc)
    print(f"{class_names[i]}: {acc:.3f}")

plt.figure(figsize=(10, 6))
bars = plt.bar(class_names, class_accuracies, color='skyblue', edgecolor='navy')
plt.title('Exactitud por Clase')
plt.ylabel('Exactitud')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 5. Distribución de confianza
confidence_scores = np.max(predictions, axis=1)
correct_predictions = (pred_classes == true_classes)

plt.figure(figsize=(10, 6))
plt.hist(confidence_scores[correct_predictions], bins=30, alpha=0.7, 
         label='Correctas', color='green', density=True)
plt.hist(confidence_scores[~correct_predictions], bins=30, alpha=0.7, 
         label='Incorrectas', color='red', density=True)
plt.title('Distribución de Confianza')
plt.xlabel('Confianza')
plt.ylabel('Densidad')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 6. Ejemplos de predicciones
plt.figure(figsize=(12, 8))
for i in range(16):
    plt.subplot(4, 4, i+1)
    idx = np.random.randint(0, len(x_test))
    plt.imshow(x_test[idx])
    
    pred = class_names[pred_classes[idx]]
    real = class_names[true_classes[idx]]
    conf = np.max(predictions[idx])
    
    color = 'green' if pred == real else 'red'
    plt.title(f'P:{pred}\nR:{real}\n{conf:.2f}', color=color, fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.show()

# Guardar pesos sinápticos y polarizaciones
model.save_weights('pesos_modelo_cnn_lstm.weights.h5')

# Extraer estadísticas de pesos
weights_data = {}
for i, layer in enumerate(model.layers):
    weights = layer.get_weights()
    if weights:
        layer_name = f"{layer.name}"
        weights_data[layer_name] = {
            'pesos_forma': weights[0].shape if weights else None,
            'bias_forma': weights[1].shape if len(weights) > 1 else None,
            'pesos_media': float(weights[0].mean()) if weights else None,
            'pesos_std': float(weights[0].std()) if weights else None,
            'bias_media': float(weights[1].mean()) if len(weights) > 1 else None,
            'bias_std': float(weights[1].std()) if len(weights) > 1 else None,
            'tipo_capa': layer.__class__.__name__
        }

# Guardar estadísticas
with open('estadisticas_pesos.pkl', 'wb') as f:
    pickle.dump(weights_data, f)

print("\nEstadísticas de Pesos:")
for name, data in weights_data.items():
    if data['pesos_media'] is not None:
        print(f"{name}: media={data['pesos_media']:.4f}, std={data['pesos_std']:.4f}")

print(f"\nParámetros totales: {model.count_params():,}")
print("\nArchivos guardados:")
print("- pesos_modelo_cnn_lstm.weights.h5")
print("- estadisticas_pesos.pkl")