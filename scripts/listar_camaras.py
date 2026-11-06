"""Lista las cámaras que ve Windows y guarda una foto de cada una.

Sirve para saber qué número poner en vision.camara_index (config.yaml), por
ejemplo después de instalar iVCam, DroidCam o conectar una webcam USB.

    python scripts/listar_camaras.py

Las fotos quedan en reports/camara_<n>.jpg para ver cuál es cuál.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg  # noqa: E402


def main() -> int:
    carpeta = cfg.ruta_de("reports")
    carpeta.mkdir(parents=True, exist_ok=True)
    actual = cfg.cargar()["vision"]["camara_index"]
    encontradas = 0
    for i in range(6):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        ok, frame = (cap.read() if cap.isOpened() else (False, None))
        if ok and frame is not None:
            # La primera imagen de algunas cámaras virtuales sale negra: se lee otra.
            ok2, frame2 = cap.read()
            frame = frame2 if ok2 else frame
            ruta = carpeta / f"camara_{i}.jpg"
            # cv2.imwrite no acepta rutas con acentos ("Semáforo") en Windows.
            ruta.write_bytes(cv2.imencode(".jpg", frame)[1].tobytes())
            alto, ancho = frame.shape[:2]
            marca = "   <- la configurada ahora" if i == actual else ""
            print(f"  cámara {i}: {ancho}x{alto}  foto: {ruta}{marca}")
            encontradas += 1
        cap.release()
    if not encontradas:
        print("No se encontró ninguna cámara. Si usas iVCam, abre la app en el celular y el programa en la laptop.")
        return 1
    print(f"\nPon el número que quieras en config.yaml -> vision.camara_index (ahora: {actual}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
