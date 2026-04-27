from src.utils.generadores import generadorDeTuplas
import os
DST_DIR = "src/resources"

os.makedirs(DST_DIR, exist_ok=True)

with open(os.path.join(DST_DIR, "tuplas.csv"), "w") as f:
    for _ in range(1_000_000):
        tupla = generadorDeTuplas()
        f.write(f"{tupla[0]},{tupla[1]}\n")

