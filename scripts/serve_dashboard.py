"""Servidor local: tablero web + recepción de datos del ESP32.

Hace dos cosas con un solo proceso de la biblioteca estándar:

1. Sirve la página de web/ y expone los reportes de reports/ como JSON.
2. Recibe las lecturas que el ESP32 envía por Wi-Fi (sus decibeles, el color que
   tiene puesto, la posición del servo) y las guarda en la base de datos SQLite.
   El tablero las muestra junto a los datos de la laptop.

Para que el ESP32 llegue, el servidor escucha en todas las interfaces
(`dashboard.escuchar_en: 0.0.0.0`). Solo tiene sentido en la red local del aula.

Endpoints
---------
    GET  /                          tablero
    GET  /api/reportes              todas las sesiones (sesiones/*/reporte.json)
    GET  /sesiones/<id>/fotos/<f>   fotos de evidencia de una sesión
    GET  /api/esp32/ultima          última lectura recibida del ESP32
    GET  /api/esp32/lecturas?n=300  últimas N lecturas, para graficar
    POST /api/esp32/lectura         el ESP32 envía {"nivel_db", "estado_led", "posicion_servo", "uptime_s"}
    GET  /api/salud                 comprobación de que el servidor responde

Uso:
    python scripts/serve_dashboard.py
    python scripts/serve_dashboard.py --puerto 8080 --sin-navegador
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg
from src import db
from src.comm.descubrimiento import Respondedor, esp32_descubierto
from src.fusion.reporter import listar_reportes

MAX_CUERPO = 64 * 1024  # un POST del ESP32 pesa unas decenas de bytes


class Servidor(ThreadingHTTPServer):
    # En Windows, Python permite por defecto que dos procesos se enganchen al
    # mismo puerto; el resultado es un segundo servidor que arranca "bien" pero
    # nunca recibe nada, porque las conexiones van al primero (aunque esté
    # colgado). Con esto, si el puerto está ocupado el arranque falla a la vista.
    allow_reuse_address = False
    daemon_threads = True


class Manejador(SimpleHTTPRequestHandler):
    """Sirve los archivos de web/ y añade la API."""

    # Una petición que se quede a medias (cliente que abre y no envía el cuerpo)
    # no debe colgar el servidor: se corta a los pocos segundos. Junto con el
    # servidor multihilo, cada cliente va por su lado y el ESP32 nunca espera
    # detrás del navegador ni al revés.
    timeout = 5

    def __init__(self, *args, directorio: str, **kwargs):
        super().__init__(*args, directory=directorio, **kwargs)

    # ------------------------------------------------------------ GET
    def do_GET(self) -> None:  # noqa: N802 - nombre impuesto por la clase base
        url = urlparse(self.path)
        if url.path == "/api/reportes":
            self._json({"reportes": listar_reportes()})
        elif url.path == "/api/esp32/ultima":
            self._json({"lectura": db.ultima_lectura_esp32()})
        elif url.path == "/api/esp32/lecturas":
            n = int(parse_qs(url.query).get("n", ["300"])[0])
            self._json({"lecturas": db.lecturas_esp32(max(1, min(n, 5000)))})
        elif url.path == "/api/esp32/descubierto":
            self._json({"esp32": esp32_descubierto()})
        elif url.path == "/api/salud":
            self._json({"ok": True, "servicio": "semaforo-del-silencio"})
        elif url.path.startswith("/sesiones/"):
            self._foto(url.path)
        else:
            super().do_GET()

    def _foto(self, ruta_url: str) -> None:
        """Fotos de sesiones/, sin dejar salir de esa carpeta ni servir otra cosa."""
        from urllib.parse import unquote

        base = cfg.ruta_de("sesiones").resolve()
        destino = (base / unquote(ruta_url[len("/sesiones/"):])).resolve()
        if base not in destino.parents or destino.suffix.lower() != ".jpg" or not destino.is_file():
            self.send_error(404, "Foto no encontrada")
            return
        datos = destino.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(datos)))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(datos)

    # ------------------------------------------------------------ POST
    def do_POST(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        if url.path != "/api/esp32/lectura":
            self._json({"ok": False, "error": "ruta desconocida"}, codigo=404)
            return

        largo = int(self.headers.get("Content-Length", "0") or 0)
        if largo <= 0 or largo > MAX_CUERPO:
            self._json({"ok": False, "error": "cuerpo vacío o demasiado grande"}, codigo=400)
            return
        try:
            datos = json.loads(self.rfile.read(largo).decode("utf-8"))
            if not isinstance(datos, dict):
                raise ValueError("se esperaba un objeto JSON")
        except (ValueError, UnicodeDecodeError) as exc:
            self._json({"ok": False, "error": f"JSON inválido: {exc}"}, codigo=400)
            return

        datos.setdefault("origen", "esp32")
        datos["ip"] = self.client_address[0]
        try:
            id_ = db.guardar_lectura_esp32(datos)
        except Exception as exc:  # noqa: BLE001
            self._json({"ok": False, "error": f"no se pudo guardar: {exc}"}, codigo=500)
            return
        self._json({"ok": True, "id": id_})

    # ------------------------------------------------------------ util
    def _json(self, datos: dict, codigo: int = 200) -> None:
        cuerpo = json.dumps(datos, ensure_ascii=False).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(cuerpo)

    def guess_type(self, path):
        """Declara UTF-8 en los archivos de texto.

        Sin esto el navegador adivina la codificación y los acentos salen rotos
        ("SemÃ¡foro" en lugar de "Semáforo").
        """
        tipo = super().guess_type(path)
        if tipo.startswith(("text/", "application/javascript")) and "charset=" not in tipo:
            return f"{tipo}; charset=utf-8"
        return tipo

    def log_message(self, formato: str, *args) -> None:
        # Solo se registra la API; el log por archivo estático ensucia la consola.
        if "/api/" in str(args[0] if args else ""):
            super().log_message(formato, *args)


def ip_local() -> str:
    """IP de esta laptop en la red local, para pegarla en el ESP32."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def main() -> int:
    conf = cfg.cargar()
    cd = conf["dashboard"]
    ap = argparse.ArgumentParser(description="Tablero de reportes por salón y receptor del ESP32")
    ap.add_argument("--puerto", type=int, default=int(cd["puerto"]))
    ap.add_argument("--escuchar-en", default=str(cd.get("escuchar_en", "127.0.0.1")))
    ap.add_argument("--sin-navegador", action="store_true")
    args = ap.parse_args()

    carpeta_web = cfg.ruta_de("web")
    if not (carpeta_web / "index.html").exists():
        print(f"No se encontró {carpeta_web / 'index.html'}", file=sys.stderr)
        return 1

    db.conectar().close()  # crea el archivo y el esquema si no existen

    n = len(listar_reportes())
    if n == 0:
        print("Aviso: todavía no hay sesiones en sesiones/.")
        print("Genera uno con: python scripts/run_session.py --minutos 2\n")

    try:
        servidor = Servidor((args.escuchar_en, args.puerto), partial(Manejador, directorio=str(carpeta_web)))
    except OSError as exc:
        print(f"No se pudo abrir el puerto {args.puerto}: {exc}", file=sys.stderr)
        print("Hay otro servidor corriendo en ese puerto. Ciérralo (o mira cuál es con:"
              f" netstat -ano | findstr :{args.puerto}) y vuelve a intentar.", file=sys.stderr)
        return 1
    url_local = f"http://localhost:{args.puerto}/"

    # El ESP32 se anuncia por UDP; se le contesta con la URL de esta laptop.
    puerto_udp = int(cd.get("puerto_descubrimiento", 8765))
    respondedor = Respondedor(puerto_udp, args.puerto)
    respondedor.start()
    visto = esp32_descubierto()
    if visto:
        print(f"ESP32 visto por última vez en {visto['ip']} ({visto['visto']})")

    print(f"Tablero en {url_local}  ({n} sesiones)")
    if args.escuchar_en == "0.0.0.0":
        print(f"Esta laptop:  http://{ip_local()}:{args.puerto}/api/esp32/lectura  "
              f"(el ESP32 la descubre solo por UDP {puerto_udp})")
    else:
        print("Escuchando solo en esta laptop; el ESP32 no podrá conectarse (usa --escuchar-en 0.0.0.0).")
    print("Ctrl+C para detener.")
    if not args.sin_navegador:
        webbrowser.open(url_local)

    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
    finally:
        respondedor.detener()
        servidor.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
