if __name__ == "__main__":
    from src.utils.generadores import generadorDeTuplas
    from src.Cuantizadores.BuzoGrey import CuantizadorVectorial
    from sklearn.cluster import KMeans
    import numpy as np
    import matplotlib.pyplot as plt

    NUMERO_PUNTOS = 1_000_000
    PORCENTAJE_ENTRENAMIENTO = 0.8
    MAX_PUNTOS_GRAFICA = 20000
    NUMERO_CENTROIDES = 8
    RANDOM_STATE = 42

    # =========================
    # 1. Generación de datos
    # =========================
    arregloTuplas = np.array(
        [generadorDeTuplas() for _ in range(NUMERO_PUNTOS)],
        dtype=np.float64
    )

    indice_corte = int(NUMERO_PUNTOS * PORCENTAJE_ENTRENAMIENTO)
    puntos_entrenamiento = arregloTuplas[:indice_corte]
    puntos_prueba = arregloTuplas[indice_corte:]

    # =========================
    # 2. Entrenamiento LBG
    # =========================
    cuantizador = CuantizadorVectorial(
        numeroDeCentroides=NUMERO_CENTROIDES,
        constantesDePerturbacion=np.array(
            [
                [1.001, 1.001],
                [0.999, 0.999]
            ],
            dtype=np.float64
        )
    )

    cuantizador.entrenar(puntos_entrenamiento)

    centroides_lbg_dict = cuantizador.obtenerCentroides()
    centroides_lbg = np.array(
        [centroides_lbg_dict[i] for i in sorted(centroides_lbg_dict.keys())]
    )

    # =========================
    # 3. Entrenamiento KMeans
    # =========================
    kmeans = KMeans(
        n_clusters=NUMERO_CENTROIDES,
        random_state=RANDOM_STATE,
        n_init=10
    )
    kmeans.fit(puntos_entrenamiento)

    centroides_kmeans = kmeans.cluster_centers_

    # =========================
    # 4. Muestra para graficar
    # =========================
    cantidad_grafica = min(MAX_PUNTOS_GRAFICA, len(puntos_prueba))
    rng = np.random.default_rng(RANDOM_STATE)
    indices_muestra = rng.choice(len(puntos_prueba), size=cantidad_grafica, replace=False)
    puntos_muestra = puntos_prueba[indices_muestra]

    # Etiquetas LBG
    etiquetas_lbg = np.array([cuantizador.cuantizar(punto) for punto in puntos_muestra])

    # Etiquetas KMeans
    etiquetas_kmeans = kmeans.predict(puntos_muestra)

    # =========================
    # 5. Función para graficar
    # =========================
    def graficar_clusters(puntos, etiquetas, centroides, titulo):
        plt.figure(figsize=(10, 8))

        for i in range(len(centroides)):
            puntos_cluster = puntos[etiquetas == i]
            if len(puntos_cluster) > 0:
                plt.scatter(
                    puntos_cluster[:, 0],
                    puntos_cluster[:, 1],
                    s=8,
                    alpha=0.4,
                    label=f"Región {i}"
                )

        plt.scatter(
            centroides[:, 0],
            centroides[:, 1],
            s=250,
            marker="X",
            edgecolors="black",
            linewidths=1.5,
            label="Centroides"
        )

        plt.title(titulo)
        plt.xlabel("Componente 1")
        plt.ylabel("Componente 2")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    # =========================
    # 6. Gráfica LBG
    # =========================
    graficar_clusters(
        puntos_muestra,
        etiquetas_lbg,
        centroides_lbg,
        "Cuantización vectorial con algoritmo Linde-Buzo-Gray"
    )

    # =========================
    # 7. Gráfica KMeans
    # =========================
    graficar_clusters(
        puntos_muestra,
        etiquetas_kmeans,
        centroides_kmeans,
        "Agrupamiento con KMeans"
    )

    # =========================
    # 8. Métrica opcional
    # =========================
    distorsion_lbg = np.sum([
        np.sum((puntos_muestra[i] - centroides_lbg[etiquetas_lbg[i]]) ** 2)
        for i in range(len(puntos_muestra))
    ])

    distorsion_kmeans = np.sum([
        np.sum((puntos_muestra[i] - centroides_kmeans[etiquetas_kmeans[i]]) ** 2)
        for i in range(len(puntos_muestra))
    ])

    print(f"Distorsión LBG en muestra: {distorsion_lbg:.6f}")
    print(f"Distorsión KMeans en muestra: {distorsion_kmeans:.6f}")