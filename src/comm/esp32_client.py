"""Cliente HTTP hacia el ESP32 — contrato de la sección 8.

La laptop es el cliente y el ESP32 el servidor. Se envía un POST solo cuando el
estado cambia, y como máximo una vez por segundo. El audio del INMP441 llega
aparte, por UDP: [número de bloque, 4 bytes][PCM de 16 bits a 16 kHz]. Se usa UDP
porque un bloque perdido no importa, y esperarlo atrasaría todo lo demás.

Regla de oro de este módulo: nunca bloquear el bucle principal esperando la red.
Si el ESP32 no responde se registra el error y se sigue; el ESP32 conserva su
propio control por decibeles, así que el semáforo no se apaga porque la laptop
pierda el Wi-Fi.

Ejecutable de forma independiente:
    python -m src.comm.esp32_client --probar
    python -m src.comm.esp32_client --estado AMARILLO
"""
from __future__ import annotations

import socket
import struct
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config as cfg  # noqa: E402


class ClienteESP32:
    """Envía el estado al ESP32 sin frenar el bucle que lo llama."""

    def __init__(self, configuracion: dict | None = None) -> None:
        conf = (configuracion or cfg.cargar())["esp32"]
        self.habilitado = bool(conf.get("habilitado", False))
        self._conf = conf
        self._resuelto_t = 0.0
        self._resolver()
        self.timeout = float(conf["timeout_s"])
        self.min_intervalo = float(conf["min_intervalo_envio_s"])
        # El firmware vuelve a su control por dB si la laptop calla un rato, así
        # que el mismo estado se repite cada tanto mientras dure la sesión.
        self.reenviar_cada = float(conf.get("reenviar_estado_s", 3.0))

        self._ultimo_envio = 0.0
        self._ultimo_estado: str | None = None
        self._lock = threading.Lock()
        self.enviados = 0
        self.fallidos = 0
        self.ultimo_error: str | None = None

        self.puerto_audio = int(conf.get("puerto_audio", 8766))
        self._audio: list = []
        self._audio_n = 0
        self._audio_cond = threading.Condition()
        self._audio_ultimo_bloque = -1
        self._audio_sock: socket.socket | None = None
        self.audio_bloques = 0
        self.audio_perdidos = 0

    def _resolver(self) -> None:
        """Si el ESP32 se anunció hace poco por la red, su IP real manda sobre la
        de config.yaml. Así no hay que editar nada al cambiar de red, y si la IP
        cambia a mitad de sesión (el hotspot da otra) se sigue llegando a él."""
        if time.time() - self._resuelto_t < 5.0:
            return
        self._resuelto_t = time.time()
        from src.comm.descubrimiento import esp32_descubierto

        conf = self._conf
        visto = esp32_descubierto()
        ip, puerto = (visto["ip"], visto["puerto"]) if visto else (conf["ip"], conf["puerto"])
        self.descubierto = visto is not None
        self.url = f"http://{ip}:{puerto}{conf['ruta_estado']}"
        self.url_servo = f"http://{ip}:{puerto}{conf.get('ruta_servo', '/servo')}"
        self.url_audio = f"http://{ip}:{puerto}{conf.get('ruta_audio', '/audio')}"

    # ------------------------------------------------------------------
    def _post(self, payload: dict) -> None:
        """Se ejecuta en un hilo aparte: aquí sí se puede esperar a la red."""
        import requests

        try:
            r = requests.post(self.url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            with self._lock:
                self.enviados += 1
                self.ultimo_error = None
        except Exception as exc:  # noqa: BLE001 - cualquier fallo de red se registra y se sigue
            with self._lock:
                self.fallidos += 1
                self.ultimo_error = f"{type(exc).__name__}: {exc}"
            print(f"[esp32] no se pudo enviar el estado: {exc}", file=sys.stderr)

    def enviar(self, payload: dict, forzar: bool = False) -> bool:
        """Envía el estado si cambió. Devuelve True si se despachó el envío."""
        if not self.habilitado:
            return False

        self._resolver()
        estado = payload.get("estado")
        ahora = time.time()
        with self._lock:
            mismo = estado == self._ultimo_estado
            muy_seguido = ahora - self._ultimo_envio < self.min_intervalo
            refrescar = mismo and ahora - self._ultimo_envio >= self.reenviar_cada
            if not forzar and (muy_seguido or (mismo and not refrescar)):
                return False
            self._ultimo_estado = estado
            self._ultimo_envio = ahora

        threading.Thread(target=self._post, args=(payload,), daemon=True).start()
        return True

    def girar(self, posicion: int, esperar: bool = True) -> bool:
        """Ordena al ESP32 mover el servo a una posición fija.

        A diferencia del estado, aquí sí conviene esperar la respuesta: la
        laptop pausa el análisis hasta que la cámara esté quieta, y necesita
        saber si la orden llegó. Con el ESP32 apagado devuelve False y el
        programa sigue analizando desde la posición actual.
        """
        from src import db

        if not self.habilitado:
            db.registrar_orden_servo(posicion, ok=False, error="esp32 deshabilitado en config")
            return False

        import requests

        self._resolver()
        url = self.url_servo
        try:
            r = requests.post(url, json={"posicion": int(posicion)}, timeout=self.timeout)
            r.raise_for_status()
            db.registrar_orden_servo(posicion, ok=True)
            return True
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self.fallidos += 1
                self.ultimo_error = f"{type(exc).__name__}: {exc}"
            db.registrar_orden_servo(posicion, ok=False, error=str(exc))
            print(f"[esp32] no se pudo ordenar el giro a {posicion}: {exc}", file=sys.stderr)
            return False

    # ------------------------------------------------------------------ audio
    def _escuchar_audio(self) -> None:
        import numpy as np

        sock = self._audio_sock
        while self._audio_sock is sock:
            try:
                datos, _ = sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(datos) < 6:
                continue
            n = struct.unpack("<I", datos[:4])[0]
            if n <= self._audio_ultimo_bloque and n != 0:
                continue                             # repetido o fuera de orden
            if self._audio_ultimo_bloque >= 0 and n > self._audio_ultimo_bloque + 1:
                self.audio_perdidos += n - self._audio_ultimo_bloque - 1
            self._audio_ultimo_bloque = n
            muestras = np.frombuffer(datos[4:], dtype="<i2").astype(np.float32) / 32768.0
            with self._audio_cond:
                self._audio.append(muestras)
                self._audio_n += len(muestras)
                # Como mucho 10 s en memoria: si nadie lee, se descarta lo viejo.
                while self._audio_n > 160000 and self._audio:
                    self._audio_n -= len(self._audio.pop(0))
                self.audio_bloques += 1
                self._audio_cond.notify_all()

    def _pedir_audio(self, activo: bool) -> bool:
        import requests

        self._resolver()
        try:
            r = requests.post(self.url_audio, json={"activo": int(activo), "puerto": self.puerto_audio},
                              timeout=self.timeout)
            r.raise_for_status()
            return bool(r.json().get("ok"))
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self.ultimo_error = f"{type(exc).__name__}: {exc}"
            print(f"[esp32] no se pudo {'pedir' if activo else 'cortar'} el audio: {exc}", file=sys.stderr)
            return False

    def iniciar_audio(self) -> bool:
        """Abre el puerto UDP y pide al ESP32 que mande el audio del INMP441."""
        if not self.habilitado:
            return False
        if self._audio_sock is None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
            sock.settimeout(0.5)
            sock.bind(("", self.puerto_audio))
            self._audio_sock = sock
            threading.Thread(target=self._escuchar_audio, daemon=True, name="audio-udp").start()
        with self._audio_cond:
            self._audio.clear()
            self._audio_n = 0
        self._audio_ultimo_bloque = -1
        return self._pedir_audio(True)

    def detener_audio(self) -> None:
        if self._audio_sock is not None:
            self._pedir_audio(False)

    def leer_audio(self, n_muestras: int, timeout: float = 2.0):
        """Devuelve exactamente n_muestras (float32) o None si no llegan a tiempo."""
        import numpy as np

        fin = time.time() + timeout
        with self._audio_cond:
            while self._audio_n < n_muestras:
                restante = fin - time.time()
                if restante <= 0:
                    return None
                self._audio_cond.wait(restante)
            juntas = np.concatenate(self._audio)
            salida, resto = juntas[:n_muestras], juntas[n_muestras:]
            self._audio = [resto] if len(resto) else []
            self._audio_n = len(resto)
        return salida

    def estadisticas(self) -> dict:
        with self._lock:
            return {
                "habilitado": self.habilitado,
                "transporte": "wifi",
                "url": self.url,
                "descubierto": self.descubierto,
                "enviados": self.enviados,
                "fallidos": self.fallidos,
                "audio_bloques": self.audio_bloques,
                "audio_bloques_perdidos": self.audio_perdidos,
                "ultimo_error": self.ultimo_error,
            }


    def lectura(self, timeout: float = 0.0) -> dict | None:
        """Última lectura del ESP32. Por Wi-Fi llegan a serve_dashboard.py y quedan
        en la base de datos, así que se leen de ahí (con su hora de recepción)."""
        from datetime import datetime

        from src import db

        fila = db.ultima_lectura_esp32()
        if not fila:
            return None
        datos = dict(fila)
        try:
            datos["recibida_t"] = datetime.fromisoformat(fila["recibido"]).timestamp()
        except (KeyError, ValueError):
            datos["recibida_t"] = 0.0
        return datos

    def finalizar(self, estado: str = "VERDE") -> bool:
        """Deja el semáforo en un color y espera la respuesta (para el cierre)."""
        if not self.habilitado:
            return False
        import requests

        try:
            r = requests.post(self.url, json={"estado": estado, "origen": "fin_sesion"}, timeout=self.timeout)
            r.raise_for_status()
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[esp32] no se pudo dejar el semáforo en {estado}: {exc}", file=sys.stderr)
            return False

    def cerrar(self) -> None:
        """Cierra el puerto de audio; cada envío HTTP abre y cierra su conexión."""
        sock, self._audio_sock = self._audio_sock, None
        if sock is not None:
            sock.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Prueba del cliente HTTP del ESP32")
    ap.add_argument("--estado", default="AMARILLO", choices=["VERDE", "AMARILLO", "ROJO"])
    ap.add_argument("--probar", action="store_true", help="enviar aunque esté deshabilitado")
    args = ap.parse_args()

    cliente = ClienteESP32()
    if args.probar:
        cliente.habilitado = True

    print(f"Destino: {cliente.url}")
    if not cliente.habilitado:
        print("esp32.habilitado está en false en config.yaml; usa --probar para forzar.")
        raise SystemExit(0)

    payload = {
        "estado": args.estado,
        "origen": "ia",
        "clase_audio": "murmullo",
        "n_de_pie": 4,
        "evento": None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    cliente.enviar(payload, forzar=True)
    time.sleep(cliente.timeout + 0.5)
    print(cliente.estadisticas())
