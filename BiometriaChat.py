# Carrega a imagem de teste
test_original = cv2.imread("finger-print.png")
cv2.imshow('Original Image', test_original)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Lista para armazenar os pontos de correspondência
match_points = []

# Itera por todos os arquivos no diretório do banco de dados
for file in [file for file in os.listdir("./database")]:
    print(file)
    fingerprint_database_image = cv2.imread("./database/"+file)
    
    # Cria o detector SIFT
    sift = cv2.SIFT_create()
    
    # Detecta pontos-chave e computa descritores para a imagem de teste
    keypoints_1, descriptors_1 = sift.detectAndCompute(test_original, None)
    print(keypoints_1)
    
    # Detecta pontos-chave e computa descritores para a imagem do banco de dados
    keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_database_image, None)
    print(keypoints_2)
    
    # Usa o algoritmo FLANN para encontrar correspondências
    matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), dict()).knnMatch(descriptors_1, descriptors_2, k=2)
    
    # Filtra correspondências válidas usando a razão de distância de Lowe
    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_points.append(p)

# Determina o número de pontos-chave
keypoints = min(len(keypoints_1), len(keypoints_2))
print(keypoints)  
print(len(match_points))

# Verifica se a taxa de correspondência é maior que 30%
if (len(match_points) / keypoints) > 0.3:
    print("% match: ", len(match_points) / keypoints * 100)
    print("Fingerprint ID: " + str(file)) 
    
    # Desenha as correspondências entre as duas imagens
    result = cv2.drawMatches(test_original, keypoints_1, fingerprint_database_image, keypoints_2, match_points, None)
    result = cv2.resize(result, None, fx=2.5, fy=2.5)
    
    # Mostra a imagem resultante com as correspondências
    cv2.imshow('Matched Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
