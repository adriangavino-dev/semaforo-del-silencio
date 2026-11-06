"""Cliente HTTP hacia el ESP32 — contrato de la sección 8.

La laptop es el cliente y el ESP32 el servidor. Se envía un POST solo cuando el
estado cambia, y como máximo una vez por segundo.

Regla de oro de este módulo: nunca bloquear el bucle principal esperando la red.
Si el ESP32 no responde se registra el error y se sigue; el ESP32 conserva su
propio control por decibeles, así que el semáforo no se apaga porque la laptop
pierda el Wi-Fi.

Ejecutable de forma independiente:
    python -m src.comm.esp32_client --probar
    python -m src.comm.esp32_client --estado AMARILLO
"""
from __future__ import annotations

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
        # Si el ESP32 se anunció hace poco por la red, su IP real manda sobre la
        # de config.yaml. Así no hay que editar nada al cambiar de red.
        from src.comm.descubrimiento import esp32_descubierto

        visto = esp32_descubierto()
        ip, puerto = (visto["ip"], visto["puerto"]) if visto else (conf["ip"], conf["puerto"])
        self.descubierto = visto is not None
        self.url = f"http://{ip}:{puerto}{conf['ruta_estado']}"
        self.url_servo = f"http://{ip}:{puerto}{conf.get('ruta_servo', '/servo')}"
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

    def estadisticas(self) -> dict:
        with self._lock:
            return {
                "habilitado": self.habilitado,
                "transporte": "wifi",
                "url": self.url,
                "descubierto": self.descubierto,
                "enviados": self.enviados,
                "fallidos": self.fallidos,
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
        """Nada que cerrar: cada envío abre y cierra su propia conexión."""


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
