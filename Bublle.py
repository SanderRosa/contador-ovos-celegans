import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Caminho da imagem ---
caminho = "Contar.jpg"
img = cv2.imread(caminho)

if img is None:
    print("❌ Erro: não consegui abrir a imagem.")
    exit()

# --- Converter para cinza ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- CLAHE para contraste ---
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)

# --- Binarização adaptativa ---
thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 51, 7
)

# --- Máscara circular (para remover bordas) ---
mask = np.zeros_like(thresh)
h, w = thresh.shape
cv2.circle(mask, (w//2, h//2), min(h, w)//2 - 5, 255, -1)
thresh = cv2.bitwise_and(thresh, mask)

# --- Remover ruído pequeno ---
kernel = np.ones((2, 2), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# --- Contar componentes conectados ---
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

# Remover o "fundo" da contagem
pontinhos = num_labels - 1

print(f"✅ Quantidade de pontinhos encontrados: {pontinhos}")

# --- Desenhar resultado ---
saida = img.copy()
for i in range(1, num_labels):  # começa de 1 (pula fundo)
    x, y, w, h, area = stats[i]
    cx, cy = centroids[i]
    cv2.circle(saida, (int(cx), int(cy)), 3, (0, 0, 255), -1)  # centro
    cv2.rectangle(saida, (x, y), (x + w, y + h), (0, 255, 0), 1)

# --- Mostrar ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("Threshold")
plt.imshow(thresh, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Pontinhos Detectados")
plt.imshow(cv2.cvtColor(saida, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()
