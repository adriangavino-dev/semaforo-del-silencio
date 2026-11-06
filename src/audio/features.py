"""YAMNet como extractor de características de audio.

YAMNet está preentrenado sobre AudioSet y no se reentrena (restricción 1). Se le
dan dos usos:

  1. Etapa 1 (demo): sus 521 clases de AudioSet se muestran en pantalla para
     evidenciar que el sistema "escucha" y reconoce tipos de sonido sin haber
     entrenado nada todavía.
  2. Etapa 3: su vector de 1024 dimensiones es la entrada del RandomForest que
     clasifica las cuatro clases propias del proyecto (sección 5.3).

YAMNet procesa ventanas de 0.96 s. Si un fragmento produce varias ventanas, los
vectores se promedian, tal como indica la especificación.

Ejecutable de forma independiente (restricción 6):
    python -m src.audio.features --wav data/audio/raw/sesion01.wav
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config as cfg  # noqa: E402

DIM_EMBEDDING = 1024
SAMPLE_RATE_YAMNET = 16000


class Yamnet:
    """Envoltura mínima sobre el modelo YAMNet de TensorFlow Hub."""

    def __init__(self, configuracion: dict | None = None) -> None:
        conf = (configuracion or cfg.cargar())["audio"]
        self.handle: str = conf["yamnet"]["handle"]
        self.top_n: int = int(conf["yamnet"]["top_n"])
        self._modelo = None
        self._nombres_clases: list[str] = []

    # ------------------------------------------------------------------
    def cargar(self) -> "Yamnet":
        """Descarga el modelo la primera vez y lo deja en caché dentro del proyecto.

        TF Hub guarda por defecto en la carpeta temporal de Windows, que se
        limpia sola y deja el modelo a medias ("contains neither saved_model.pb").
        Por eso la caché va a rutas.cache_tfhub, salvo que TFHUB_CACHE_DIR ya
        esté definida. TensorFlow no acepta acentos en la ruta en Windows, así
        que si la carpeta del proyecto los tiene ("Semáforo") se usa una carpeta
        fija del usuario, que tampoco se limpia sola.
        """
        import os

        if "TFHUB_CACHE_DIR" not in os.environ:
            cache = cfg.ruta_de("cache_tfhub")
            if not str(cache).isascii():
                base = os.environ.get("LOCALAPPDATA") or str(Path.home())
                cache = Path(base) / "SemaforoDelSilencio" / "tfhub"
            os.environ["TFHUB_CACHE_DIR"] = str(cache)
        Path(os.environ["TFHUB_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
        import tensorflow_hub as hub

        self._modelo = hub.load(self.handle)
        ruta_mapa = self._modelo.class_map_path().numpy().decode("utf-8")
        with open(ruta_mapa, "r", encoding="utf-8") as fh:
            self._nombres_clases = [fila["display_name"] for fila in csv.DictReader(fh)]
        return self

    @property
    def cargado(self) -> bool:
        return self._modelo is not None

    @property
    def nombres_clases(self) -> list[str]:
        return self._nombres_clases

    # ------------------------------------------------------------------
    def _preparar(self, bloque: np.ndarray) -> np.ndarray:
        """Normaliza el bloque al formato que espera YAMNet: float32 mono 1-D."""
        onda = np.asarray(bloque, dtype=np.float32)
        if onda.ndim > 1:
            onda = onda.mean(axis=1)
        return np.ascontiguousarray(onda)

    def analizar(self, bloque: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (puntajes promediados de 521 clases, embedding de 1024 dims)."""
        if self._modelo is None:
            raise RuntimeError("El modelo no está cargado. Llama a cargar() primero.")

        puntajes, embeddings, _ = self._modelo(self._preparar(bloque))
        puntajes = puntajes.numpy()
        embeddings = embeddings.numpy()
        if puntajes.size == 0:
            return np.zeros(len(self._nombres_clases)), np.zeros(DIM_EMBEDDING, dtype=np.float32)
        # Promedio sobre las ventanas de 0.96 s (sección 5.3).
        return puntajes.mean(axis=0), embeddings.mean(axis=0)

    def clases_principales(self, bloque: np.ndarray, n: int | None = None) -> list[tuple[str, float]]:
        """Las n clases de AudioSet más probables para el bloque."""
        n = n or self.top_n
        puntajes, _ = self.analizar(bloque)
        indices = np.argsort(puntajes)[::-1][:n]
        return [(self._nombres_clases[i], float(puntajes[i])) for i in indices]

    def embedding(self, bloque: np.ndarray) -> np.ndarray:
        """Vector de 1024 dimensiones que resume el contenido sonoro del bloque."""
        _, vector = self.analizar(bloque)
        return vector


def extraer_embedding(ruta_wav: str, modelo: Yamnet | None = None) -> np.ndarray:
    """Devuelve un vector de 1024 dimensiones que representa el contenido sonoro
    del fragmento. Remuestrea a 16 kHz mono antes de pasar por YAMNet.

    Firma definida en la sección 5.3 de la especificación. La usará el script de
    entrenamiento de la etapa 3.
    """
    import librosa

    onda, _ = librosa.load(ruta_wav, sr=SAMPLE_RATE_YAMNET, mono=True)
    if modelo is None:
        modelo = Yamnet().cargar()
    return modelo.embedding(onda)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Prueba de YAMNet sobre un archivo WAV")
    ap.add_argument("--wav", required=True, help="ruta del archivo .wav a analizar")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    import librosa

    print("Cargando YAMNet (la primera vez descarga ~17 MB)...")
    yam = Yamnet().cargar()
    onda, _ = librosa.load(args.wav, sr=SAMPLE_RATE_YAMNET, mono=True)
    print(f"Duración: {len(onda) / SAMPLE_RATE_YAMNET:.2f} s")
    for nombre, puntaje in yam.clases_principales(onda, args.top):
        print(f"  {puntaje:.3f}  {nombre}")
    print(f"Embedding: {yam.embedding(onda).shape}")
