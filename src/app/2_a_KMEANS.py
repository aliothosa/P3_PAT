from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

NUMERO_PUNTOS = 1_000_000
PORCENTAJE_ENTRENAMIENTO = 0.8
MAX_PUNTOS_GRAFICA = 20000
NUMEROS_CENTROIDES = [8, 16, 64, 256]
RANDOM_STATE = 42


arregloTuplas = np.loadtxt("src/resources/tuplas.csv", delimiter=",", dtype=float)


indice_corte = int(NUMERO_PUNTOS * PORCENTAJE_ENTRENAMIENTO)
puntos_entrenamiento = arregloTuplas[:indice_corte]
puntos_prueba = arregloTuplas[indice_corte:]

cantidad_grafica = min(MAX_PUNTOS_GRAFICA, len(puntos_prueba))
rng = np.random.default_rng(RANDOM_STATE)
indices_muestra = rng.choice(len(puntos_prueba), size=cantidad_grafica, replace=False)
puntos_muestra = puntos_prueba[indices_muestra]

# ==========================================
# 3. Configuración de figuras
# ==========================================
fig_kmeans, axes_kmeans = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
fig_kmeans.suptitle("Agrupamiento con KMeans", fontsize=16)
axes_kmeans = axes_kmeans.flatten()

distorsiones_kmeans = {}

# ==========================================
# 4. Función para graficar en subplots
# ==========================================
def graficar_en_eje(ax, puntos, etiquetas, centroides, titulo):
    scatter = ax.scatter(
        puntos[:, 0],
        puntos[:, 1],
        c=etiquetas,
        cmap="nipy_spectral",
        s=6,
        alpha=0.5
    )

    ax.scatter(
        centroides[:, 0],
        centroides[:, 1],
        s=120,
        marker="X",
        edgecolors="black",
        linewidths=1.2
    )

    ax.set_title(titulo)
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.grid(True, alpha=0.3)

# ==========================================
# 5. Entrenamiento y gráficas para cada k
# ==========================================
for idx, numero_centroides in enumerate(NUMEROS_CENTROIDES):
    print(f"\nProcesando {numero_centroides} regiones...")

    # =========================
    # KMeans
    # =========================
    kmeans = KMeans(
        n_clusters=numero_centroides,
        random_state=RANDOM_STATE,
        n_init=10
    )
    kmeans.fit(puntos_entrenamiento)

    centroides_kmeans = kmeans.cluster_centers_
    etiquetas_kmeans = kmeans.predict(puntos_muestra)

    graficar_en_eje(
        axes_kmeans[idx],
        puntos_muestra,
        etiquetas_kmeans,
        centroides_kmeans,
        f"KMeans - {numero_centroides} clusters"
    )

    # Distorsión KMeans
    distorsion_kmeans = np.sum(
        (puntos_muestra - centroides_kmeans[etiquetas_kmeans]) ** 2
    )
    distorsiones_kmeans[numero_centroides] = distorsion_kmeans

# ==========================================
# 6. Mostrar ventanas
# ==========================================
plt.show()

print("\n=== Distorsiones KMeans ===")
for k, v in distorsiones_kmeans.items():
    print(f"{k} clusters: {v:.6f}")