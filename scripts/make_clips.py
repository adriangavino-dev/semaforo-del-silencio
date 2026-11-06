"""Corta las grabaciones en fragmentos de 1 segundo — Etapa 2.

Los fragmentos se toman con solapamiento de 0.5 s, lo que duplica la cantidad de
muestras aprovechables sin volver a grabar. Cada fragmento queda registrado en
data/audio/labels.csv junto con su sesión de origen.

Si el nombre del archivo trae la clase codificada (por ejemplo
`s01__murmullo.wav`, que es lo que produce record_session.py con --clase), los
fragmentos salen ya etiquetados. Si no, la clase queda vacía y la completa
label_tool.py.

Uso:
    python scripts/make_clips.py --todas
    python scripts/make_clips.py --archivo s01__murmullo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg
from src.audio.dataset import escribir_etiquetas, leer_etiquetas
from scripts.record_session import partir_nombre


def cortar(ruta_wav: Path, conf: dict, filas: dict[str, dict[str, str]]) -> int:
    """Corta un WAV en fragmentos y los registra. Devuelve cuántos generó."""
    import librosa
    import soundfile as sf

    ca = conf["audio"]
    sr = int(ca["sample_rate"])
    dur = float(ca["duracion_bloque_s"])
    salto = dur - float(ca["solapamiento_s"])
    if salto <= 0:
        raise ValueError("solapamiento_s debe ser menor que duracion_bloque_s")

    stem = ruta_wav.stem
    sesion, clase = partir_nombre(stem)
    onda, _ = librosa.load(str(ruta_wav), sr=sr, mono=True)

    muestras = int(sr * dur)
    paso = int(sr * salto)
    if len(onda) < muestras:
        print(f"  {stem}: dura menos de {dur:.0f} s, se omite")
        return 0

    destino = cfg.ruta_de("data_audio_clips") / sesion
    destino.mkdir(parents=True, exist_ok=True)

    n = 0
    for inicio in range(0, len(onda) - muestras + 1, paso):
        fragmento = onda[inicio : inicio + muestras]
        clip_id = f"{stem}_{n:05d}"
        sf.write(str(destino / f"{clip_id}.wav"), fragmento, sr)
        if clip_id not in filas:
            # No se pisa una etiqueta puesta a mano si el script se corre dos veces.
            filas[clip_id] = {"clip_id": clip_id, "clase": clase, "sesion": sesion}
        n += 1

    etiqueta = clase or "sin etiquetar"
    print(f"  {stem}: {n} fragmentos  (sesión {sesion}, {etiqueta})")
    return n


def main() -> int:
    conf = cfg.cargar()
    ap = argparse.ArgumentParser(description="Corta las grabaciones en fragmentos de 1 s")
    ap.add_argument("--archivo", help="nombre de una grabación concreta, sin la extensión")
    ap.add_argument("--todas", action="store_true", help="procesar todas las grabaciones")
    args = ap.parse_args()

    carpeta = cfg.ruta_de("data_audio_raw")
    if args.todas:
        archivos = sorted(carpeta.glob("*.wav"))
    elif args.archivo:
        archivos = [carpeta / f"{args.archivo}.wav"]
    else:
        ap.error("indica --archivo NOMBRE o --todas")

    faltan = [a for a in archivos if not a.exists()]
    if faltan:
        print("No se encontró: " + ", ".join(str(a) for a in faltan), file=sys.stderr)
        return 1
    if not archivos:
        print(f"No hay grabaciones en {carpeta}", file=sys.stderr)
        return 1

    ruta_labels = cfg.ruta_de("data_audio_labels")
    filas = leer_etiquetas(ruta_labels)

    print("Cortando fragmentos:")
    total = sum(cortar(a, conf, filas) for a in archivos)
    escribir_etiquetas(filas, ruta_labels)

    print(f"\nTotal: {total} fragmentos. Registro en {ruta_labels}")
    print("Revisa el estado con:\n    python scripts/label_tool.py --resumen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
