"""Ensayo del enlace Wi-Fi con la placa real: órdenes, audio y clasificación.

Hace lo que una sesión hace por la red, sin cámara: repite la orden de color cada
3 s, recibe el audio del INMP441 y lo clasifica. Con serve_dashboard.py corriendo.

    python scripts/ensayo_enlace.py --minutos 10

Deja el resumen en reports/enlace_<fecha>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg  # noqa: E402
from src.audio.captura_esp32 import CapturaESP32  # noqa: E402
from src.audio.infer import ClasificadorAudio  # noqa: E402
from src.comm.esp32 import crear_cliente  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Ensayo del enlace Wi-Fi")
    ap.add_argument("--minutos", type=float, default=10.0)
    args = ap.parse_args()

    conf = cfg.cargar()
    cliente = crear_cliente(conf)
    clf = ClasificadorAudio(conf)
    clf.cargar(obligatorio=False)
    captura = CapturaESP32(cliente, conf)
    inicio = datetime.now()
    captura.iniciar(intentos=12)
    print("Audio en marcha.", flush=True)

    clases: Counter = Counter()
    sin_audio = 0
    t0 = time.time()
    fin = t0 + args.minutos * 60
    ultimo_progreso = t0
    while time.time() < fin:
        cliente.enviar({"estado": "VERDE", "origen": "ensayo"})   # se repite cada 3 s
        bloque = captura.leer(timeout=3.0)
        if bloque is None:
            sin_audio += 1
            continue
        clases[clf.clasificar_audio(bloque)["clase"]] += 1
        if time.time() - ultimo_progreso >= 60:
            ultimo_progreso = time.time()
            e = cliente.estadisticas()
            print(f"{(time.time() - t0) / 60:.0f} min: {e['audio_bloques']} bloques, {e['audio_bloques_perdidos']} perdidos, "
                  f"{e['enviados']} órdenes ok, {e['fallidos']} fallidas", flush=True)
    captura.detener()
    time.sleep(2.5)   # que terminen las órdenes que siguen en camino
    e = cliente.estadisticas()
    cliente.cerrar()

    bloques = e["audio_bloques"] + e["audio_bloques_perdidos"]
    ordenes = e["enviados"] + e["fallidos"]
    resumen = {
        "inicio": inicio.isoformat(timespec="seconds"),
        "minutos": round((time.time() - t0) / 60, 1),
        "ordenes_entregadas": e["enviados"],
        "ordenes_totales": ordenes,
        "ordenes_pct": round(100 * e["enviados"] / ordenes, 1) if ordenes else None,
        "audio_bloques": e["audio_bloques"],
        "audio_perdidos": e["audio_bloques_perdidos"],
        "audio_perdida_pct": round(100 * e["audio_bloques_perdidos"] / bloques, 2) if bloques else None,
        "segundos_sin_audio": sin_audio * 3,
        "clases": dict(clases),
    }
    ruta = cfg.ruta_de("reports") / f"enlace_{inicio:%Y%m%d_%H%M}.json"
    ruta.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumen, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
