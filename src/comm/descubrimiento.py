"""Descubrimiento automático del ESP32 en la red local.

El ESP32 anuncia por UDP (difusión) "aquí estoy, mi IP es X" cada pocos
segundos. La laptop escucha ese puerto, se guarda la IP y le responde con la
dirección a la que debe enviar sus lecturas. Ninguno de los dos necesita conocer
la IP del otro de antemano, así que el sistema funciona en casa, en la
universidad o con el hotspot de un celular sin tocar configuración ni regrabar
el ESP32.

La IP descubierta se guarda en data/esp32_descubierto.json y tiene prioridad
sobre esp32.ip de config.yaml mientras sea reciente.

Ejecutable de forma independiente:
    python -m src.comm.descubrimiento        # escucha 30 s y muestra qué anuncia el ESP32
"""
from __future__ import annotations

import json
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config as cfg  # noqa: E402

VALIDEZ_S = 600  # una IP descubierta hace más de 10 min ya no se da por buena


def ruta_registro() -> Path:
    return cfg.RAIZ / "data" / "esp32_descubierto.json"


def esp32_descubierto() -> dict | None:
    """Última IP anunciada por el ESP32, si es reciente."""
    ruta = ruta_registro()
    if not ruta.exists():
        return None
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        if time.time() - float(datos.get("epoch", 0)) > VALIDEZ_S:
            return None
        return datos
    except (ValueError, OSError):
        return None


def _guardar(datos: dict) -> None:
    ruta = ruta_registro()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(".tmp")
    tmp.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(ruta)


def ip_local() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


class Respondedor(threading.Thread):
    """Hilo que escucha los anuncios del ESP32 y le contesta con la URL de la laptop."""

    def __init__(self, puerto_udp: int, puerto_http: int, al_descubrir=None) -> None:
        super().__init__(daemon=True)
        self.puerto_udp = puerto_udp
        self.puerto_http = puerto_http
        self.al_descubrir = al_descubrir
        self._detener = threading.Event()
        self.ultimo: dict | None = None

    def run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1.0)
        try:
            sock.bind(("", self.puerto_udp))
        except OSError as exc:
            print(f"[descubrimiento] no se pudo escuchar en UDP {self.puerto_udp}: {exc}", file=sys.stderr)
            return

        while not self._detener.is_set():
            try:
                datos, origen = sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                anuncio = json.loads(datos.decode("utf-8"))
            except ValueError:
                continue
            if anuncio.get("dispositivo") != "semaforo-esp32":
                continue

            ip = anuncio.get("ip") or origen[0]
            registro = {
                "ip": ip,
                "puerto": int(anuncio.get("puerto", 80)),
                "led": anuncio.get("led"),
                "visto": datetime.now().isoformat(timespec="seconds"),
                "epoch": time.time(),
                "ip_laptop": ip_local(),
            }
            nuevo = self.ultimo is None or self.ultimo.get("ip") != ip
            self.ultimo = registro
            _guardar(registro)

            respuesta = json.dumps({
                "laptop_url": f"http://{ip_local()}:{self.puerto_http}/api/esp32/lectura",
            }).encode("utf-8")
            try:
                sock.sendto(respuesta, (origen[0], self.puerto_udp))
            except OSError:
                pass
            if nuevo:
                print(f"[descubrimiento] ESP32 en {ip}:{registro['puerto']}  (LED {registro['led']})")
                if self.al_descubrir:
                    self.al_descubrir(registro)
        sock.close()

    def detener(self) -> None:
        self._detener.set()


if __name__ == "__main__":
    conf = cfg.cargar()
    puerto = int(conf["dashboard"].get("puerto_descubrimiento", 8765))
    print(f"Escuchando anuncios del ESP32 en UDP {puerto} durante 30 s...")
    r = Respondedor(puerto, int(conf["dashboard"]["puerto"]))
    r.start()
    time.sleep(30)
    r.detener()
    if r.ultimo:
        print("Último anuncio:", r.ultimo)
    else:
        print("Ningún anuncio. ¿El ESP32 está encendido y en la misma red? ¿El firewall permite UDP", puerto, "?")
