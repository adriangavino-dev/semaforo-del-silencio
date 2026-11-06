"""Genera un reporte de sesión simulado.

Sirve para probar el tablero y para las demostraciones en clase sin tener que
grabar una sesión completa de verdad. Los reportes que produce quedan marcados
con `"simulada": true` y el tablero los muestra con una etiqueta, para que nunca
se confundan con mediciones reales.

Uso:
    python scripts/simular_sesion.py --minutos 45
    python scripts/simular_sesion.py --salon "Aula 302" --semilla 7
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg
from src.fusion.reporter import Reporte
from src.fusion.state import MaquinaEstado

# Guion de una clase típica: (minutos, clase de audio, dB base, personas de pie)
GUION = [
    (6, "murmullo", 57.0, 3),      # los alumnos entran y conversan
    (14, "voz_docente", 64.0, 1),  # el docente explica
    (7, "murmullo", 61.0, 2),      # trabajo en grupos
    (5, "silencio", 38.0, 0),      # trabajo individual
    (8, "murmullo", 67.0, 5),      # se desordena
    (5, "voz_docente", 63.0, 1),   # el docente retoma el control
]


def main() -> int:
    conf = cfg.cargar()
    ap = argparse.ArgumentParser(description="Genera un reporte de sesión simulado")
    ap.add_argument("--minutos", type=float, default=45.0)
    ap.add_argument("--salon", default=None)
    ap.add_argument("--semilla", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.semilla)
    maquina = MaquinaEstado(conf)
    reporte = Reporte(conf, salon=args.salon)

    # La sesión simulada se fecha hacia atrás para que parezca ya terminada.
    reporte.inicio = datetime.now() - timedelta(minutes=args.minutos)

    total_guion = sum(t for t, *_ in GUION)
    escala = args.minutos / total_guion
    segundo = 0.0

    for minutos, clase, db_base, de_pie_base in GUION:
        for _ in range(int(minutos * escala * 60)):
            segundo += 1.0
            db = float(db_base + rng.normal(0, 3.2))
            clase_bloque = clase

            # De vez en cuando entra un ruido brusco aislado.
            if rng.random() < 0.004:
                clase_bloque = "ruido_brusco"
                db += rng.uniform(10, 18)
                reporte.eventos.append({
                    "timestamp": (reporte.inicio + timedelta(seconds=segundo)).isoformat(timespec="seconds"),
                    "segundo": round(segundo, 1),
                    "tipo": "ruido_brusco",
                    "zona": "",
                    "posicion": 0,
                    "nivel_db": round(db, 1),
                })

            n_personas = int(np.clip(rng.normal(26, 1.6), 20, 32))
            n_de_pie = int(np.clip(rng.poisson(max(de_pie_base, 0.2)), 0, 8))

            # Movimiento anómalo cuando se junta desorden y gente parada.
            if de_pie_base >= 4 and rng.random() < 0.0025:
                zona = "alumnos"
                reporte.eventos.append({
                    "timestamp": (reporte.inicio + timedelta(seconds=segundo)).isoformat(timespec="seconds"),
                    "segundo": round(segundo, 1),
                    "tipo": "movimiento_anomalo",
                    "zona": zona,
                    "posicion": 0,
                })

            lectura = maquina.actualizar(clase_bloque, db, None, n_de_pie)

            reporte.filas.append({
                "timestamp": (reporte.inicio + timedelta(seconds=segundo)).isoformat(timespec="seconds"),
                "segundo": round(segundo, 1),
                "estado": lectura["estado"],
                "clase_audio": clase_bloque,
                "confianza_audio": round(float(rng.uniform(0.68, 0.96)), 3),
                "nivel_db": round(db, 1),
                "n_personas": n_personas,
                "n_de_pie": n_de_pie,
                "evento": "",
                "zona_evento": "",
                "posicion": 0,
            })

    reporte.cambios_estado = maquina.cambios
    ruta_csv, ruta_json = reporte.guardar(
        maquina.estado,
        extra={
            "simulada": True,
            "modelo_audio_entrenado": True,
            "linea_base_calibrada": True,
        },
    )

    resumen = reporte.resumen(maquina.estado)
    print(f"Sesión simulada de {args.minutos:.0f} min en '{resumen['salon']}'")
    print(f"  estado final:      {resumen['estado_final']}")
    print(f"  tiempo por estado: {resumen['porcentaje_estados']}")
    print(f"  clases de audio:   {resumen['audio']['distribucion_clases']}")
    print(f"  eventos:           {resumen['eventos']['total']}")
    print(f"  cambios de estado: {len(resumen['cambios_estado'])}")
    print(f"\n  {ruta_csv}\n  {ruta_json}")
    print("\nVer el tablero:\n    python scripts/serve_dashboard.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
