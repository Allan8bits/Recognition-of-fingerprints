###########################################################
# Nome do projeto: Reconhecimeto de sinais datilológicos
# Autor: Allan Rodrigo
# Data: 23/05/2023
###########################################################

# Importação das bibliotecas necessárias
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Step 1: Pré-processamento das imagens
def preprocess_image(image):
    # Redimensiona a imagem para 64x64 pixels
    image = cv2.resize(image, (64, 64))
    # Normaliza os valores dos pixels para o intervalo [0, 1]
    image = image / 255.0
    return image

# Step 2: Carregamento do conjunto de dados
def load_dataset(directory):
    data = []  # Lista para armazenar as imagens
    labels = []  # Lista para armazenar os rótulos

    label_encoder = LabelEncoder()  # Objeto para codificar os rótulos em formato numérico

    # Percorre o diretório e subdiretórios para obter as imagens e rótulos
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg'):
                label = os.path.basename(root)  # Extrai o rótulo do nome do diretório
                image_path = os.path.join(root, file)  # Caminho completo da imagem
                image = cv2.imread(image_path)  # Carrega a imagem usando OpenCV
                image = preprocess_image(image)  # Pré-processa a imagem
                data.append(image)  # Adiciona a imagem à lista de dados
                labels.append(label)  # Adiciona o rótulo à lista de rótulos

    # Converte as listas em arrays numpy
    data = np.array(data)
    labels = np.array(labels)

    # Codifica os rótulos em formato numérico
    labels = label_encoder.fit_transform(labels)

    return data, labels

# Step 3: Divisão do conjunto de dados em treinamento e teste
def split_dataset(data, labels, test_size=0.2, random_state=42):
    # Divide o conjunto de dados em treinamento e teste
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_size, random_state=random_state)
    # Redimensiona os arrays de rótulos para uma dimensão
    train_labels = train_labels.flatten()
    test_labels = test_labels.flatten()
    return train_data, test_data, train_labels, test_labels

# Step 4: Definição da arquitetura da CNN
def create_model():
    # Criação do modelo sequencial do TensorFlow
    model = tf.keras.Sequential([
        # Camada de convolução com 16 filtros, função de ativação ReLU e tamanho de entrada 64x64x3
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        # Camada de pooling máximo com tamanho 2x2
        tf.keras.layers.MaxPooling2D((2, 2)),
        # Camada de flatten para converter os mapas de características em um vetor unidimensional
        tf.keras.layers.Flatten(),
        # Camada densa com 64 neurônios e função de ativação ReLU
        tf.keras.layers.Dense(64, activation='relu'),
        # Camada densa de saída com 26 neurônios (número de classes) e função de ativação softmax
        tf.keras.layers.Dense(26, activation='softmax')
    ])
    # Compilação do modelo com otimizador Adam, função de perda sparse_categorical_crossentropy e métrica de acurácia
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 5: Treinamento da rede neural
def train_model(model, train_data, train_labels, test_data, test_labels, epochs, batch_size):
    print("Treinando o modelo...")
    # Treina o modelo usando os dados de treinamento
    model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(test_data, test_labels))
    print("Treinamento do modelo concluído.")

# Step 6: Realização da busca em grade para ajuste de hiperparâmetros
def perform_grid_search(data, labels, param_grid, cv=3):
    label_encoder = LabelEncoder()
    # Codifica os rótulos em formato numérico
    labels_encoded = label_encoder.fit_transform(labels)

    # Cria um classificador KerasWrapper usando a função create_model
    model = KerasClassifier(build_fn=create_model, verbose=0)
    # Realiza a busca em grade usando GridSearchCV do scikit-learn
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv)
    grid_search.fit(data, labels_encoded)
    return grid_search.best_params_

# Step 7: Avaliação do modelo com os melhores hiperparâmetros
def evaluate_model(model, test_data, test_labels):
    print("Avaliando o modelo...")
    # Avalia o modelo usando os dados de teste
    loss, accuracy = model.evaluate(test_data, test_labels)
    print(f"Loss: {loss}")
    print(f"Acurácia: {accuracy}")

# Step 8: Exibição das imagens de teste com as letras correspondentes
def display_test_images(data, labels, model, class_to_letter):
    print("Exibindo imagens de teste...")
    # Realiza a predição das classes das imagens de teste
    predictions = model.predict(data)
    predicted_labels = np.argmax(predictions, axis=1)

    # Cria uma figura com subplots para exibir as imagens
    fig, axs = plt.subplots(6, 5, figsize=(10, 12))
    fig.tight_layout()

    # Percorre as imagens de teste e exibe-as juntamente com as letras verdadeiras e preditas
    for i, ax in enumerate(axs.flat):
        image = data[i]
        true_label = labels[i]
        predicted_label = predicted_labels[i]

        ax.imshow(image)
        ax.axis('off')

        true_letter = class_to_letter[true_label]
        predicted_letter = class_to_letter[predicted_label]

        ax.set_title(f'Verdadeiro: {true_letter}\nPrevisto: {predicted_letter}', fontsize=8)

    plt.show()

#--------------------------------------ENTRADAS DO PROJETO--------------------------------------
data_directory = r'C:\Users\allan\OneDrive\Documentos\HandTracking\DATA'
param_grid = {
    'batch_size': [32, 64]
}

# Step 2: Carregamento do conjunto de dados
data, labels = load_dataset(data_directory)

# Step 3: Divisão do conjunto de dados
train_data, test_data, train_labels, test_labels = split_dataset(data, labels)

# Step 6: Busca em grade para ajuste de hiperparâmetros
best_params = perform_grid_search(train_data, train_labels, param_grid)

# Step 5: Treinamento da rede neural
best_model = create_model()
train_model(best_model, train_data, train_labels, test_data, test_labels, epochs=1000, batch_size=best_params['batch_size'])

# Step 7: Avaliação do modelo com os melhores hiperparâmetros
evaluate_model(best_model, test_data, test_labels)

# Mapeamento das classes para as letras correspondentes
class_to_letter = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y'
}

# Step 8: Exibição das imagens de teste com as letras correspondentes
display_test_images(test_data, test_labels, best_model, class_to_letter)
