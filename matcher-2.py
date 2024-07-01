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

# Leitura da imagem de teste
test_original = cv2.imread("finger-print.png")
cv2.imshow("Test Original", test_original)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Inicializando variáveis
found = False
match_points = []
keypoints_1 = []
keypoints_2 = []

# Iteração por cada banco de dados
for folder in database_folders:
    db_path = os.path.join("./database", folder)
    for file in os.listdir(db_path):
        fingerprint_database_image = cv2.imread(os.path.join(db_path, file))
        if fingerprint_database_image is None:
            continue

        sift = cv2.xfeatures2d.SIFT_create()

        keypoints_1, descriptors_1 = sift.detectAndCompute(test_original, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_database_image, None)

        if descriptors_1 is None or descriptors_2 is None:
            continue

        matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), dict()).knnMatch(descriptors_1, descriptors_2, k=2)

        for p, q in matches:
            if p.distance < 0.1 * q.distance:
                match_points.append(p)

        keypoints = min(len(keypoints_1), len(keypoints_2))
        if keypoints > 0 and (len(match_points) / keypoints) > 0.3:
            found = True
            print("% match: ", len(match_points) / keypoints * 100)
            print("Fingerprint ID: " + str(file))
            result = cv2.drawMatches(test_original, keypoints_1, fingerprint_database_image, keypoints_2, match_points, None)
            result = cv2.resize(result, None, fx=2.5, fy=2.5)
            cv2.imshow("Result", result)
            cv2.waitKey(0)
            break

if not found:
    print("No match found in any database.")
