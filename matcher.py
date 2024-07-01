import cv2
import numpy as np
import os

# Lista de nomes das pastas dos bancos de dados
database_folders = [
    "DB1_B",
    "DB2_B",
    "DB3_B Parte 1",
    "DB3_B Parte 2",
    "DB4_B"
]

# Caminhos das pastas
base_path = os.path.join("..", "Data")
test_descriptors_path = os.path.join(base_path, "Fingerprint", "file7.txt")
database_base_path = os.path.join(base_path, "Data-base")


# Função para carregar descritores de um arquivo de texto
def load_descriptors(filename):
    return np.loadtxt(filename, dtype=float)


# Carrega descritores da imagem de teste
test_descriptors = load_descriptors(test_descriptors_path)

# Inicializando variáveis
found = False

# Cria o objeto SIFT
sift = cv2.SIFT_create()


# Iteração por cada banco de dados
for folder in database_folders:
    db_path = os.path.join(database_base_path, folder)
    print(f"Searching in folder: {db_path}")
    for file in os.listdir(db_path):
        fingerprint_database_image = cv2.imread(os.path.join(db_path, file))
        file_path = os.path.join(db_path, file)
        print(f"Cheking file: {file_path}")
        if fingerprint_database_image is None:
            print(f"Error loading database image: {file_path}")
            continue

        # Detecta descritores para a imagem do banco de dados
        _, descriptors_2 = sift.detectAndCompute(fingerprint_database_image, None)

        if descriptors_2 is None:
            continue

        test_descriptors = test_descriptors.astype(np.float32)
        descriptors_2 = descriptors_2.astype(np.float32)

        # Comparação direta dos descritores
        if np.array_equal(test_descriptors, descriptors_2):
            found = True
            print("Fingerprint match found!")
            print(f"From: {folder}")
            print("Fingerprint ID: " + str(file))
            break

    if found:
        break

if not found:
    print("No match found in any database.")
