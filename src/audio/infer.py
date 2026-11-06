"""Inferencia del clasificador de audio — sección 5.5.

Mantiene una ventana deslizante con las últimas predicciones y reporta la moda de
esa ventana. Así un fragmento mal clasificado no alcanza a mover el estado del
sistema, que es justo lo que se quiere evitar: una falsa alarma cuesta más que
una detección omitida, porque si el semáforo miente el docente deja de mirarlo.

Ejecutable de forma independiente:
    python -m src.audio.infer --micro
    python -m src.audio.infer --wav data/audio/clips/s01/s01__murmullo_00000.wav
"""
from __future__ import annotations

import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config as cfg  # noqa: E402
from src.audio.capture import nivel_db  # noqa: E402
from src.audio.features import Yamnet  # noqa: E402


class ClasificadorAudio:
    """Clasifica bloques de 1 s en las cuatro clases del proyecto."""

    def __init__(self, configuracion: dict | None = None) -> None:
        self.conf = configuracion or cfg.cargar()
        ca = self.conf["audio"]
        self.referencia_db = float(ca["db"]["referencia_rms"])
        self.yamnet = Yamnet(self.conf)
        self._modelo = None
        self.clases: list[str] = []
        self.entrenado_el: str | None = None
        self._historial: deque[str] = deque(maxlen=int(ca["ventana_moda"]))
        # Nivel suavizado para el respaldo sin modelo: promedio de los últimos
        # bloques de 1 s, porque el nivel crudo salta ±10 dB de un segundo a otro.
        suav = int(ca["db"].get("sin_modelo", {}).get("suavizado_s", 3))
        self._niveles: deque[float] = deque(maxlen=max(1, suav))

    # ------------------------------------------------------------------
    @property
    def disponible(self) -> bool:
        """True si hay un modelo entrenado cargado."""
        return self._modelo is not None

    def cargar(self, obligatorio: bool = True) -> "ClasificadorAudio":
        """Carga YAMNet y el clasificador entrenado.

        Con obligatorio=False, si el modelo aún no existe se carga solo YAMNet y
        `disponible` queda en False. Eso permite correr la demo antes de haber
        entrenado.
        """
        import joblib

        self.yamnet.cargar()
        ruta = cfg.ruta_de("modelo_audio")
        if not ruta.exists():
            if obligatorio:
                raise FileNotFoundError(
                    f"No existe {ruta}. Entrena primero con: python -m src.audio.train"
                )
            print(f"[audio] aún no hay modelo entrenado en {ruta}", file=sys.stderr)
            return self

        paquete = joblib.load(ruta)
        self._modelo = paquete["modelo"]
        self.clases = list(paquete["clases"])
        self.entrenado_el = paquete.get("entrenado")
        return self

    # ------------------------------------------------------------------
    def clasificar_audio(self, bloque_audio: np.ndarray) -> dict:
        """Entrada: 1 segundo de audio a 16 kHz.

        Salida: {"clase", "confianza", "nivel_db"} — firma de la sección 5.5.
        La clase devuelta es la moda de la ventana deslizante, no la predicción
        cruda del último bloque.
        """
        db = nivel_db(bloque_audio, self.referencia_db)
        self._niveles.append(db)
        suave = sum(self._niveles) / len(self._niveles)
        if self._modelo is None:
            return {"clase": None, "confianza": 0.0, "nivel_db": db, "nivel_db_suave": suave, "clase_bloque": None}

        vector = self.yamnet.embedding(bloque_audio).reshape(1, -1)
        probabilidades = self._modelo.predict_proba(vector)[0]
        i = int(np.argmax(probabilidades))
        clase_bloque = str(self._modelo.classes_[i])
        self._historial.append(clase_bloque)

        clase = Counter(self._historial).most_common(1)[0][0]
        # La confianza se reporta sobre la clase que finalmente se informa.
        j = list(self._modelo.classes_).index(clase)
        return {
            "clase": clase,
            "confianza": float(probabilidades[j]),
            "nivel_db": db,
            "nivel_db_suave": suave,
            "clase_bloque": clase_bloque,
            # Probabilidad de cada clase en el último bloque, para verla en pantalla.
            "probabilidades": {str(c): float(p) for c, p in zip(self._modelo.classes_, probabilidades)},
        }

    def reiniciar(self) -> None:
        self._historial.clear()
        self._niveles.clear()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Prueba del clasificador de audio")
    ap.add_argument("--wav", help="clasificar un archivo")
    ap.add_argument("--micro", action="store_true", help="clasificar el micrófono en vivo")
    ap.add_argument("--segundos", type=float, default=20.0)
    args = ap.parse_args()

    clf = ClasificadorAudio().cargar()
    print(f"Modelo entrenado el {clf.entrenado_el} · clases: {clf.clases}")

    if args.wav:
        import librosa

        sr = int(cfg.cargar()["audio"]["sample_rate"])
        onda, _ = librosa.load(args.wav, sr=sr, mono=True)
        print(clf.clasificar_audio(onda))
    elif args.micro:
        import time

        from src.audio.capture import CapturaAudio

        captura = CapturaAudio()
        print(f"Escuchando {args.segundos:.0f} s...")
        with captura:
            fin = time.time() + args.segundos
            while time.time() < fin:
                bloque = captura.leer(timeout=0.5)
                if bloque is None:
                    continue
                r = clf.clasificar_audio(bloque)
                print(
                    f"  {r['nivel_db']:5.1f} dB   {r['clase']:<14} "
                    f"conf {r['confianza']:.2f}   (bloque: {r['clase_bloque']})"
                )
    else:
        ap.error("indica --wav ARCHIVO o --micro")
