    
    
class CuantizadorDePotencias:
    def __init__(self, numeroDeCentroides: int, constantesDePerturbacion: np.ndarray, num_lpc: int):
        self.numeroDeCentroides = numeroDeCentroides
        self.constantesDePerturbacion = constantesDePerturbacion
        self.num_lpc = num_lpc
        self.centroides = {}

    def _crear_centroide_desde_señal(self, id_c: int, señal_promedio: List[float]) -> Segmento:
        """
        Usa tus funciones de Wiener para generar un nuevo centroide (Segmento)
        a partir de una señal promediada.
        """
        valores_iniciales = [0.0] * len(señal_promedio) # Asunción por defecto
        
        # Llamada a tu función de Wiener
        lpc_calculados, matriz_R, r_vector = filtroWiener(señal_promedio, valores_iniciales, self.num_lpc)

        return Segmento(
            senalSRC=f"CENTROIDE_{id_c}",
            numeroSegmento=id_c,
            vectorDePotencia=np.zeros(1),
            matrizDeCorrelacion=matriz_R,
            vectorDeCorrelacion=r_vector,
            coeficientesLPC=lpc_calculados
        )

    def _asignar_puntos(self, segmentos: List[Segmento], centroides: List[Segmento]):
        grupos = {i: [] for i in range(len(centroides))}
        distorsion_global = 0.0
        for seg in segmentos:
            # Distancia de Itakura-Saito (definida previamente)
            dists = [itakuraSaito(seg, c) for c in centroides]
            idx_min = np.argmin(dists)
            grupos[idx_min].append(seg)
            distorsion_global += dists[idx_min]
        return grupos, distorsion_global

    def _recalcular_centroides(self, grupos: dict, centroides_anteriores: List[Segmento]):
        nuevos_centroides = []
        for i in range(len(centroides_anteriores)):
            if not grupos[i]:
                nuevos_centroides.append(centroides_anteriores[i])
            else:
                # 1. Obtenemos las autocorrelaciones promedio del grupo
                # (En voz, el centroide es el modelo del promedio de las correlaciones)
                r_promedio = np.mean([s.vectorDeCorrelacion for s in grupos[i]], axis=0)
                
                # 2. Re-calculamos LPC usando la lógica de Wiener sobre el r_promedio
                # Para simplificar la integración con tus funciones que piden señalBase:
                # Usamos el r_promedio directamente para resolver el sistema.
                lpc_nuevo = self._obtener_lpc_desde_r(r_promedio)
                
                nuevos_centroides.append(Segmento(
                    senalSRC="CENTROIDE",
                    numeroSegmento=i,
                    vectorDePotencia=np.zeros(1),
                    matrizDeCorrelacion=np.zeros((1,1)),
                    vectorDeCorrelacion=r_promedio,
                    coeficientesLPC=lpc_nuevo
                ))
        return nuevos_centroides

    def _obtener_lpc_desde_r(self, r: np.ndarray) -> np.ndarray:
        """ Función interna para resolver Wiener sin re-procesar la señal entera """
        M = self.num_lpc
        matriz = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                matriz[i, j] = r[abs(i - j)]
        vector = r[1:M+1]
        matriz += np.eye(M) * 1e-6
        return np.linalg.solve(matriz, vector)

    def entrenar(self, segmentos: List[Segmento]) -> bool:
        # Lógica de Buzo-Gray (Splitting)
        e1, e2 = self.constantesDePerturbacion
        
        # 1. Centroide inicial
        r_global = np.mean([s.vectorDeCorrelacion for s in segmentos], axis=0)
        lpc_global = self._obtener_lpc_desde_r(r_global)
        centroides_actuales = [Segmento("C0", 0, np.zeros(1), np.zeros((1,1)), r_global, lpc_global)]

        while len(centroides_actuales) < self.numeroDeCentroides:
            # DIVIDIR (Splitting)
            temp = []
            for i, c in enumerate(centroides_actuales):
                temp.append(self._crear_variante(c, e1, i*2))
                temp.append(self._crear_variante(c, e2, i*2+1))
            centroides_actuales = temp
            
            # REFINAR (Asignar y Recalcular)
            dist_ant = np.inf
            for _ in range(50):
                grupos, dist_act = self._asignar_puntos(segmentos, centroides_actuales)
                centroides_actuales = self._recalcular_centroides(grupos, centroides_actuales)
                if abs(dist_ant - dist_act) < 1e-4: break
                dist_ant = dist_act
                
        self.centroides = {i: c for i, c in enumerate(centroides_actuales)}
        return True

    def _crear_variante(self, c: Segmento, epsilon: float, nuevo_id: int):
        return Segmento(c.senalSRC, nuevo_id, c.vectorDePotencia, 
                       c.matrizDeCorrelacion, c.vectorDeCorrelacion, c.coeficientesLPC * epsilon)

    def cuantizar(self, segmento: Segmento) -> int:
        dists = {i: itakuraSaito(segmento, c) for i, c in self.centroides.items()}
        return min(dists, key=dists.get)

    def obtenerCentroides(self) -> dict:
        return self.centroides