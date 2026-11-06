"""Simulador del ESP32 para probar la comunicación sin el circuito.

Se comporta como el firmware real:

- Es un servidor HTTP (como el ESP32) que acepta `POST /estado` con el color
  del semáforo y `POST /servo` con la posición a la que girar la cámara.
- Mide "decibeles" (inventados) y decide su propio color por umbrales con
  histéresis, como hará el INMP441. Si la laptop le manda un estado, lo respeta
  durante unos segundos; si deja de recibir, vuelve a su propio control.
- Cada segundo envía `POST /api/esp32/lectura` al servidor de la laptop con
  su nivel, su color y la posición del servo.

Sirve para verificar los dos sentidos del contrato y para la exposición si el
circuito no está listo. Cuando esté el ESP32 real, basta apuntar
`esp32.ip` a él y apagar este simulador.

Uso (en una terminal aparte, con serve_dashboard.py ya corriendo):
    python scripts/simular_esp32.py
    python scripts/simular_esp32.py --puerto 8081 --laptop http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg

ESTADOS = ("VERDE", "AMARILLO", "ROJO")


class EstadoESP32:
    """Lo que el firmware tendría en memoria."""

    def __init__(self, conf: dict) -> None:
        umb = conf["audio"]["db"]
        self.umbral_amarillo = float(umb["umbral_amarillo"])
        self.umbral_rojo = float(umb["umbral_rojo"])
        self.lock = threading.Lock()
        self.nivel_db = 45.0
        self.led = "VERDE"
        self.posicion_servo = 0
        self.girando_hasta = 0.0
        self.estado_laptop: str | None = None
        self.estado_laptop_hasta = 0.0
        self.recibidos = 0
        self.inicio = time.time()
        self.rng = np.random.default_rng(1)

    def medir(self) -> None:
        """Nivel sonoro simulado: deriva lenta más ruido, con rachas altas."""
        with self.lock:
            objetivo = 62.0 if (time.time() // 40) % 2 else 48.0
            self.nivel_db += 0.15 * (objetivo - self.nivel_db) + self.rng.normal(0, 1.5)
            self.nivel_db = float(np.clip(self.nivel_db, 30, 90))

            # Prioridad: si la laptop mandó un estado hace poco, se respeta.
            if self.estado_laptop and time.time() < self.estado_laptop_hasta:
                self.led = self.estado_laptop
                return
            # Si no, control propio por dB con histéresis de 2 dB.
            if self.nivel_db > self.umbral_rojo + 2:
                self.led = "ROJO"
            elif self.nivel_db > self.umbral_amarillo + 2:
                self.led = "AMARILLO"
            elif self.nivel_db < self.umbral_amarillo - 2:
                self.led = "VERDE"

    def lectura(self) -> dict:
        with self.lock:
            return {
                "nivel_db": round(self.nivel_db, 1),
                "estado_led": self.led,
                "posicion_servo": self.posicion_servo,
                "girando": time.time() < self.girando_hasta,
                "uptime_s": int(time.time() - self.inicio),
                "origen": "esp32-simulado",
            }


def hacer_manejador(estado: EstadoESP32, tiempo_giro_ms: int):
    class Manejador(BaseHTTPRequestHandler):
        def _json(self, datos: dict, codigo: int = 200) -> None:
            cuerpo = json.dumps(datos).encode("utf-8")
            self.send_response(codigo)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(cuerpo)))
            self.end_headers()
            self.wfile.write(cuerpo)

        def _leer(self) -> dict | None:
            largo = int(self.headers.get("Content-Length", "0") or 0)
            try:
                return json.loads(self.rfile.read(largo).decode("utf-8")) if largo else {}
            except ValueError:
                return None

        def do_POST(self) -> None:  # noqa: N802
            datos = self._leer()
            if datos is None:
                self._json({"ok": False, "error": "JSON inválido"}, 400)
                return

            if self.path == "/estado":
                nuevo = str(datos.get("estado", "")).upper()
                if nuevo not in ESTADOS:
                    self._json({"ok": False, "error": "estado desconocido"}, 400)
                    return
                with estado.lock:
                    estado.estado_laptop = nuevo
                    estado.estado_laptop_hasta = time.time() + 10.0
                    estado.recibidos += 1
                print(f"  [esp32] estado recibido: {nuevo}  (audio={datos.get('clase_audio')}, "
                      f"de_pie={datos.get('n_de_pie')}, evento={datos.get('evento')})")
                self._json({"ok": True})

            elif self.path == "/servo":
                try:
                    pos = int(datos.get("posicion"))
                except (TypeError, ValueError):
                    self._json({"ok": False, "error": "posicion inválida"}, 400)
                    return
                with estado.lock:
                    estado.posicion_servo = pos
                    estado.girando_hasta = time.time() + tiempo_giro_ms / 1000.0
                print(f"  [esp32] servo -> posición {pos}")
                self._json({"ok": True, "posicion": pos, "tiempo_giro_ms": tiempo_giro_ms})

            else:
                self._json({"ok": False, "error": "ruta desconocida"}, 404)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/lectura":
                self._json(estado.lectura())
            else:
                self._json({"ok": True, "esp32": "simulado", "rutas": ["/estado", "/servo", "/lectura"]})

        def log_message(self, *a) -> None:  # silencio: ya se imprime lo relevante
            pass

    return Manejador


def bucle_envio(estado: EstadoESP32, url_laptop: str, cada_s: float, detener: threading.Event) -> None:
    """Como el firmware: mide, decide y manda su lectura a la laptop."""
    import requests

    fallos = 0
    while not detener.is_set():
        estado.medir()
        lectura = estado.lectura()
        try:
            r = requests.post(url_laptop, json=lectura, timeout=2.0)
            r.raise_for_status()
            fallos = 0
        except Exception as exc:  # noqa: BLE001
            fallos += 1
            if fallos in (1, 10, 60):
                print(f"  [esp32] no llego a la laptop ({url_laptop}): {exc}", file=sys.stderr)
        detener.wait(cada_s)


def main() -> int:
    conf = cfg.cargar()
    ap = argparse.ArgumentParser(description="Simula el ESP32 del Semáforo del Silencio")
    ap.add_argument("--puerto", type=int, default=int(conf["esp32"]["puerto"]) if int(conf["esp32"]["puerto"]) != 80 else 8081,
                    help="puerto del servidor simulado (el real usa 80; aquí 8081 para no pedir permisos)")
    ap.add_argument("--laptop", default=f"http://localhost:{conf['dashboard']['puerto']}",
                    help="dirección del servidor de la laptop")
    ap.add_argument("--cada", type=float, default=1.0, help="segundos entre lecturas enviadas")
    ap.add_argument("--tiempo-giro-ms", type=int, default=800)
    args = ap.parse_args()

    estado = EstadoESP32(conf)
    servidor = HTTPServer(("0.0.0.0", args.puerto), hacer_manejador(estado, args.tiempo_giro_ms))
    detener = threading.Event()
    hilo = threading.Thread(
        target=bucle_envio, args=(estado, f"{args.laptop}/api/esp32/lectura", args.cada, detener), daemon=True
    )
    hilo.start()

    print(f"ESP32 simulado escuchando en http://localhost:{args.puerto}  (POST /estado, POST /servo)")
    print(f"Enviando lecturas cada {args.cada:.0f} s a {args.laptop}/api/esp32/lectura")
    print(f"Para que la laptop le hable, en config.yaml:  esp32.habilitado: true, esp32.ip: 127.0.0.1, esp32.puerto: {args.puerto}")
    print("Ctrl+C para detener.\n")
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nSimulador detenido.")
    finally:
        detener.set()
        servidor.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
