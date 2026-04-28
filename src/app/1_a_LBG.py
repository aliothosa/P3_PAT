import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from src.cuantizadores.BuzoGrey import CuantizadorVectorial

NUMERO_PUNTOS = 1_000_000
PORCENTAJE_ENTRENAMIENTO = 0.8
MAX_PUNTOS_GRAFICA = 20000
NUMEROS_CENTROIDES = [1,2,4,8, 16, 64, 256]
RANDOM_STATE = 42

arregloTuplas = np.loadtxt("src/resources/tuplas.csv", delimiter=",", dtype=float)


indice_corte = int(NUMERO_PUNTOS * PORCENTAJE_ENTRENAMIENTO)
puntos_entrenamiento = arregloTuplas[:indice_corte]
puntos_prueba = arregloTuplas[indice_corte:]

cantidad_grafica = min(MAX_PUNTOS_GRAFICA, len(puntos_prueba))
rng = np.random.default_rng(RANDOM_STATE)
indices_muestra = rng.choice(len(puntos_prueba), size=cantidad_grafica, replace=False)
puntos_muestra = puntos_prueba[indices_muestra]


fig_lbg, axes_lbg = plt.subplots(4, 2, figsize=(16, 14), constrained_layout=True)
fig_lbg.suptitle("Cuantización vectorial con Linde-Buzo-Gray (LBG)", fontsize=16)
axes_lbg = axes_lbg.flatten()

distorsiones_lbg = {}

def graficar_en_eje(ax, puntos, etiquetas, centroides, titulo, seed=42):
    etiquetas_unicas = np.unique(etiquetas)
    k = len(etiquetas_unicas)

    mapa = {lab: i for i, lab in enumerate(etiquetas_unicas)}
    etiquetas_reindexadas = np.array([mapa[e] for e in etiquetas])

    # Generar más colores de los necesarios
    colores = plt.cm.gist_ncar(np.linspace(0, 1, max(k * 4, 256)))

    # Eliminar colores blancos o casi blancos
    colores = np.array([
        color for color in colores
        if not (color[0] > 0.9 and color[1] > 0.9 and color[2] > 0.9)
    ])

    rng = np.random.default_rng(seed)
    rng.shuffle(colores)

    # Tomar solo k colores
    colores = colores[:k]

    cmap = ListedColormap(colores)
    norm = BoundaryNorm(np.arange(-0.5, k + 0.5, 1), cmap.N)

    ax.scatter(
        puntos[:, 0],
        puntos[:, 1],
        c=etiquetas_reindexadas,
        cmap=cmap,
        norm=norm,
        s=6,
        alpha=0.55
    )

    ax.scatter(
        centroides[:, 0],
        centroides[:, 1],
        c=np.arange(k),
        cmap=cmap,
        norm=norm,
        s=180,
        marker="X",
        edgecolors="black",
        linewidths=1.8,
        zorder=5
    ) 

    ax.set_title(titulo)
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.grid(True, alpha=0.3)


os.makedirs("src/resources/objects", exist_ok=True)

for idx, numero_centroides in enumerate(NUMEROS_CENTROIDES):
        print(f"\nProcesando {numero_centroides} regiones...")
        direccionObjeto = f"src/resources/objects/buzo_grey_{numero_centroides}_centroides.obj"
        
        if os.path.isfile(direccionObjeto):
            cuantizador = CuantizadorVectorial.load(direccionObjeto)
        else:
            cuantizador = CuantizadorVectorial(
                numeroDeCentroides=numero_centroides,
                perturbaciones=np.array(
                    [
                        [1.0, 1.001], 
                        [1.0, 0.999]
                    ],
                    dtype=np.float64
                )
            )
        
            cuantizador.entrenar(puntos_entrenamiento)

        centroides_lbg_dict = cuantizador.obtenerCentroides()
        centroides_lbg = np.array(
            [centroides_lbg_dict[i] for i in sorted(centroides_lbg_dict.keys())],
            dtype=np.float64
        )

        etiquetas_lbg = np.fromiter(
            (cuantizador.cuantizar(punto) for punto in puntos_muestra),
            dtype=np.int64,
            count=len(puntos_muestra)
        )

        graficar_en_eje(
            axes_lbg[idx],
            puntos_muestra,
            etiquetas_lbg,
            centroides_lbg,
            f"LBG - {numero_centroides} regiones"
        )

        # Distorsión LBG
        distorsion_lbg = np.sum(
            (puntos_muestra - centroides_lbg[etiquetas_lbg]) ** 2
        )
        distorsiones_lbg[numero_centroides] = distorsion_lbg
        CuantizadorVectorial.write(cuantizador, direccionObjeto)


plt.show()

print("\n=== Distorsiones LBG ===")
for k, v in distorsiones_lbg.items():
    print(f"{k} regiones: {v:.6f}")

