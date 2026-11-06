"""Cliente del ESP32 por el cable USB (puerto serie).

Misma interfaz que ClienteESP32 (enviar, girar, estadisticas) para que
run_session.py no distinga por dónde habla. Ventajas frente al Wi-Fi: no hay
que configurar redes ni IPs, no depende del Wi-Fi del aula y la latencia es
de milisegundos.

Protocolo (una línea por mensaje, 115200 baudios):
    laptop -> ESP32   {"estado":"ROJO"}  {"servo":1}  {"prueba_led":1}  {"lectura":1}
                      {"audio":1} / {"audio":0}
    ESP32 -> laptop   ">{json}"  (respuestas a órdenes y una lectura por segundo)
                      "@<n>:<base64>"  bloque de audio de 32 ms (PCM 16 bits, 16 kHz)
                      del INMP441, solo mientras el audio está activado
    Las líneas sin ">" son mensajes del firmware para leer, se muestran y ya.

Las lecturas que llegan se guardan en SQLite (esp32_lecturas), así que el
tablero web las muestra igual que si hubieran llegado por la red.

Un solo programa puede tener el puerto abierto: run_session.py lo abre mientras
dura la sesión. Si el ESP32 no está conectado, todo devuelve False y el
programa sigue; el ESP32 conserva su propio control por decibeles.

Ejecutable de forma independiente:
    python -m src.comm.esp32_serial --listar
    python -m src.comm.esp32_serial --probar-led
    python -m src.comm.esp32_serial --estado ROJO
"""
from __future__ import annotations

import base64
import json
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config as cfg  # noqa: E402

# Chips USB-serie habituales en placas ESP32 DevKit.
_PISTAS_ESP32 = ("CP210", "Silicon Labs", "CH340", "CH9102", "USB-SERIAL", "USB Serial")


def puertos_disponibles() -> list[dict]:
    from serial.tools import list_ports

    return [
        {"puerto": p.device, "descripcion": p.description, "parece_esp32": any(x.lower() in (p.description or "").lower() for x in _PISTAS_ESP32)}
        for p in list_ports.comports()
    ]


def detectar_puerto() -> str | None:
    """El primer puerto cuyo chip parece el de un ESP32; si no, el único que haya."""
    puertos = puertos_disponibles()
    for p in puertos:
        if p["parece_esp32"]:
            return p["puerto"]
    return puertos[0]["puerto"] if len(puertos) == 1 else None


class ClienteESP32Serie:
    """Habla con el ESP32 por USB sin frenar el bucle que lo llama."""

    def __init__(self, configuracion: dict | None = None) -> None:
        conf = (configuracion or cfg.cargar())["esp32"]
        self.habilitado = bool(conf.get("habilitado", False))
        puerto = str(conf.get("puerto_serie", "auto"))
        self.puerto = detectar_puerto() if puerto.lower() == "auto" else puerto
        self.baudios = int(conf.get("baudios", 115200))
        self.timeout = float(conf["timeout_s"])
        self.min_intervalo = float(conf["min_intervalo_envio_s"])
        # El firmware vuelve a su control por dB si la laptop calla un rato, así
        # que el mismo estado se repite cada tanto mientras dure la sesión.
        self.reenviar_cada = float(conf.get("reenviar_estado_s", 3.0))

        self._serie = None
        self._lock = threading.Lock()
        self._lock_orden = threading.Lock()   # una orden en vuelo a la vez
        self._respuestas: queue.Queue[dict] = queue.Queue()
        self._ultimo_guardado = 0.0
        # Audio del INMP441 (solo con {"audio":1}): muestras float32 en [-1, 1].
        self._audio: list = []
        self._audio_n = 0
        self._audio_cond = threading.Condition()
        self._audio_ultimo_bloque = -1
        self.audio_bloques = 0
        self.audio_perdidos = 0
        # Defensa contra un fallo intermitente de pyserial en Windows: a veces
        # devuelve los últimos bytes de una línea cientos de veces por segundo.
        self._msg_anterior = ""
        self._msg_repeticiones = 0
        self.reaperturas = 0
        self._ultimo_envio = 0.0
        self._ultimo_estado: str | None = None
        self._vivo = False
        self.enviados = 0
        self.fallidos = 0
        self.recibidas = 0
        self.ultima_lectura: dict | None = None
        self.ultimo_error: str | None = None

        if self.habilitado:
            self._abrir()

    # ------------------------------------------------------------------
    def _abrir(self) -> None:
        if not self.puerto:
            self.ultimo_error = "no se encontró el puerto del ESP32 (¿está conectado el cable USB?)"
            print(f"[esp32] {self.ultimo_error}", file=sys.stderr)
            return
        try:
            import serial

            s = serial.Serial()
            s.port = self.puerto
            s.baudrate = self.baudios
            s.timeout = 0     # sin bloqueo: el lector consulta cuántos bytes hay y lee esos
            # Se abre con DTR/RTS por defecto, lo que REINICIA el ESP32 (como
            # hace el monitor serie de Arduino). Es a propósito: si el puerto
            # estuvo cerrado un rato mientras el ESP32 seguía enviando, el chip
            # USB-serie (CP2102) se queda atascado y devuelve el mismo paquete
            # cientos de veces por segundo; con el reinicio el ESP32 calla dos
            # segundos, el chip se vacía y todo llega limpio. Cuesta ~2.5 s de
            # arranque y el estado se vuelve a mandar solo (reenviar_estado_s).
            s.open()
            time.sleep(0.3)
            s.reset_input_buffer()
            self._serie = s
        except Exception as exc:  # noqa: BLE001
            self.ultimo_error = f"{type(exc).__name__}: {exc}"
            print(f"[esp32] no se pudo abrir {self.puerto}: {exc}", file=sys.stderr)
            return
        self._vivo = True
        self._hilo = threading.Thread(target=self._leer, daemon=True, name="esp32-serie")
        self._hilo.start()
        print(f"[esp32] conectado por USB en {self.puerto} a {self.baudios} baudios (el ESP32 se reinicia, ~3 s)")

    def _leer(self) -> None:
        """Hilo lector: separa respuestas de lecturas periódicas y guarda estas últimas."""
        from src import db

        pendiente = b""   # bytes de una línea que aún no terminó de llegar
        while self._vivo and self._serie is not None:
            try:
                # Lectura sin bloqueo: se leen exactamente los bytes que el
                # driver dice tener. Las lecturas con espera (read(1) con
                # timeout, readline) devolvían en Windows los mismos bytes
                # cientos de veces por segundo de forma intermitente.
                n = self._serie.in_waiting
                if not n:
                    time.sleep(0.02)
                    continue
                trozo = self._serie.read(n)
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self.ultimo_error = f"{type(exc).__name__}: {exc}"
                self._vivo = False
                print(f"[esp32] se perdió el puerto {self.puerto}: {exc}", file=sys.stderr)
                break
            if not trozo:
                continue
            pendiente += trozo
            if b"\n" not in pendiente:
                if len(pendiente) > 4096:
                    pendiente = b""
                continue
            *completas, pendiente = pendiente.split(b"\n")
            for cruda in completas:
                try:
                    self._procesar_linea(cruda.decode("utf-8", "replace").strip(), db)
                except Exception as exc:  # noqa: BLE001
                    print(f"[esp32] linea ignorada: {exc}", file=sys.stderr)

    def _procesar_linea(self, linea: str, db) -> None:
        """Una línea completa del ESP32: respuesta, lectura periódica o mensaje."""
        if not linea:
            return
        if linea.startswith("@"):
            self._recibir_audio(linea)
            return
        if not linea.startswith(">"):
            if linea == self._msg_anterior:
                self._msg_repeticiones += 1
            else:
                self._msg_anterior, self._msg_repeticiones = linea, 0
            if self._msg_repeticiones >= 5:
                self._sanear_puerto()
                return
            # La consola de Windows no siempre acepta todo (basura del arranque):
            # se imprime lo que se pueda y nunca se deja morir el hilo por esto.
            try:
                codificacion = sys.stdout.encoding or "utf-8"
                print(f"[esp32] {linea}".encode(codificacion, "replace").decode(codificacion, "replace"), flush=True)
            except Exception:  # noqa: BLE001
                pass
            return
        self._msg_repeticiones = 0
        try:
            datos = json.loads(linea[1:])
        except json.JSONDecodeError:
            return
        if "orden" in datos or ("ok" in datos and "nivel_db" not in datos):
            self._respuestas.put(datos)
            return
        # Lectura periódica del ESP32.
        datos["recibida_t"] = time.time()
        with self._lock:
            self.ultima_lectura = datos
            self.recibidas += 1
        ahora = time.time()
        if ahora - self._ultimo_guardado >= 1.0:
            self._ultimo_guardado = ahora
            datos.setdefault("origen", "esp32-usb")
            try:
                db.guardar_lectura_esp32(datos)
            except Exception as exc:  # noqa: BLE001
                print(f"[esp32] no se pudo guardar la lectura: {exc}", file=sys.stderr)

    def _sanear_puerto(self) -> None:
        """La misma línea llega en bucle: se vacía el búfer y, si insiste, se reabre."""
        if self._serie is None:
            return
        if self._msg_repeticiones == 5:
            print("[esp32] el puerto repite bytes; se vacía el búfer de entrada", file=sys.stderr)
            try:
                self._serie.reset_input_buffer()
            except Exception:  # noqa: BLE001
                pass
        elif self._msg_repeticiones >= 40:
            self._msg_repeticiones = 0
            self.reaperturas += 1
            print("[esp32] el puerto sigue repitiendo bytes; se reabre", file=sys.stderr)
            try:
                self._serie.close()
                self._serie.open()
                self._serie.reset_input_buffer()
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self.ultimo_error = f"{type(exc).__name__}: {exc}"
                print(f"[esp32] no se pudo reabrir {self.puerto}: {exc}", file=sys.stderr)

    def _recibir_audio(self, linea: str) -> None:
        """'@<n>:<base64>' -> muestras al búfer. Descarta repetidos y cuenta pérdidas."""
        import numpy as np

        try:
            cabecera, datos = linea[1:].split(":", 1)
            n = int(cabecera)
            pcm = np.frombuffer(base64.b64decode(datos, validate=True), dtype="<i2")
        except Exception:  # noqa: BLE001 - línea partida o con basura: se ignora
            return
        if n <= self._audio_ultimo_bloque and n != 0:
            return                                   # repetido (fallo del chip USB-serie)
        if self._audio_ultimo_bloque >= 0 and n > self._audio_ultimo_bloque + 1:
            self.audio_perdidos += n - self._audio_ultimo_bloque - 1
        self._audio_ultimo_bloque = n
        muestras = pcm.astype(np.float32) / 32768.0
        with self._audio_cond:
            self._audio.append(muestras)
            self._audio_n += len(muestras)
            # Como mucho 10 s en memoria: si nadie lee, se descarta lo viejo.
            while self._audio_n > 160000 and self._audio:
                self._audio_n -= len(self._audio.pop(0))
            self.audio_bloques += 1
            self._audio_cond.notify_all()

    def iniciar_audio(self) -> bool:
        """Pide al ESP32 que mande el audio del INMP441."""
        with self._audio_cond:
            self._audio.clear()
            self._audio_n = 0
        self._audio_ultimo_bloque = -1
        r = self.ordenar({"audio": 1}, "audio")
        return bool(r and r.get("ok"))

    def detener_audio(self) -> None:
        if self._vivo:
            self.ordenar({"audio": 0}, "audio")

    def leer_audio(self, n_muestras: int, timeout: float = 2.0):
        """Devuelve exactamente n_muestras (float32) o None si no llegan a tiempo."""
        import numpy as np

        fin = time.time() + timeout
        with self._audio_cond:
            while self._audio_n < n_muestras:
                restante = fin - time.time()
                if restante <= 0 or not self._vivo:
                    return None
                self._audio_cond.wait(restante)
            juntas = np.concatenate(self._audio)
            salida, resto = juntas[:n_muestras], juntas[n_muestras:]
            self._audio = [resto] if len(resto) else []
            self._audio_n = len(resto)
        return salida

    def _escribir(self, orden: dict) -> bool:
        if not self._vivo or self._serie is None:
            return False
        try:
            with self._lock:
                self._serie.write((json.dumps(orden) + "\n").encode("utf-8"))
            return True
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self.ultimo_error = f"{type(exc).__name__}: {exc}"
            print(f"[esp32] no se pudo escribir en {self.puerto}: {exc}", file=sys.stderr)
            return False

    def _esperar_respuesta(self, orden: str, timeout: float) -> dict | None:
        fin = time.time() + timeout
        while True:
            restante = fin - time.time()
            if restante <= 0:
                return None
            try:
                r = self._respuestas.get(timeout=restante)
            except queue.Empty:
                return None
            if r.get("orden") == orden:
                return r

    # ------------------------------------------------------------------
    def ordenar(self, orden: dict, nombre: str, timeout: float | None = None) -> dict | None:
        """Manda una orden y espera su respuesta. None si no hay ESP32 o no contestó."""
        with self._lock_orden:
            while not self._respuestas.empty():   # respuestas viejas que nadie recogió
                try:
                    self._respuestas.get_nowait()
                except queue.Empty:
                    break
            if not self._escribir(orden):
                return None
            return self._esperar_respuesta(nombre, timeout or self.timeout)

    def enviar(self, payload: dict, forzar: bool = False) -> bool:
        """Envía el estado si cambió. Devuelve True si se despachó el envío."""
        if not self.habilitado or not self._vivo:
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

        def _mandar() -> None:
            r = self.ordenar({"estado": estado}, "estado")
            with self._lock:
                if r and r.get("ok"):
                    self.enviados += 1
                    self.ultimo_error = None
                else:
                    self.fallidos += 1
                    self.ultimo_error = "el ESP32 no confirmó el estado"

        threading.Thread(target=_mandar, daemon=True).start()
        return True

    def girar(self, posicion: int, esperar: bool = True) -> bool:
        """Ordena girar la cámara a una posición fija y espera la confirmación."""
        from src import db

        if not self.habilitado:
            db.registrar_orden_servo(posicion, ok=False, error="esp32 deshabilitado en config")
            return False
        r = self.ordenar({"servo": int(posicion)}, "servo")
        if r and r.get("ok"):
            db.registrar_orden_servo(posicion, ok=True)
            return True
        error = (r or {}).get("error") or self.ultimo_error or "sin respuesta por USB"
        with self._lock:
            self.fallidos += 1
            self.ultimo_error = error
        db.registrar_orden_servo(posicion, ok=False, error=error)
        print(f"[esp32] no se pudo ordenar el giro a {posicion}: {error}", file=sys.stderr)
        return False

    def finalizar(self, estado: str = "VERDE") -> bool:
        """Deja el semáforo en un color y espera la confirmación (para el cierre)."""
        if not self.habilitado or not self._vivo:
            return False
        r = self.ordenar({"estado": estado}, "estado")
        return bool(r and r.get("ok"))

    def probar_leds(self) -> bool:
        r = self.ordenar({"prueba_led": 1}, "prueba_led")
        return bool(r and r.get("ok"))

    def lectura(self, timeout: float = 2.0) -> dict | None:
        """Última lectura recibida; si aún no llegó ninguna, espera una."""
        fin = time.time() + timeout
        while time.time() < fin:
            with self._lock:
                if self.ultima_lectura:
                    return dict(self.ultima_lectura)
            time.sleep(0.05)
        return None

    def estadisticas(self) -> dict:
        with self._lock:
            return {
                "habilitado": self.habilitado,
                "transporte": "usb",
                "puerto": self.puerto,
                "conectado": self._vivo,
                "enviados": self.enviados,
                "fallidos": self.fallidos,
                "lecturas_recibidas": self.recibidas,
                "audio_bloques": self.audio_bloques,
                "audio_bloques_perdidos": self.audio_perdidos,
                "ultimo_error": self.ultimo_error,
            }

    def cerrar(self) -> None:
        self._vivo = False
        hilo = getattr(self, "_hilo", None)
        if hilo is not None and hilo.is_alive():
            hilo.join(timeout=1.0)   # que el lector termine su vuelta antes de cerrar el puerto
        serie, self._serie = self._serie, None
        if serie is not None:
            try:
                serie.close()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Prueba del ESP32 por el cable USB")
    ap.add_argument("--listar", action="store_true", help="mostrar los puertos serie que hay")
    ap.add_argument("--estado", choices=["VERDE", "AMARILLO", "ROJO"], help="mandar un color")
    ap.add_argument("--servo", type=int, help="mandar una posición del servo")
    ap.add_argument("--probar-led", action="store_true", help="encender los tres LED por turno")
    args = ap.parse_args()

    if args.listar:
        for p in puertos_disponibles():
            print(f"  {p['puerto']:<6} {p['descripcion']}{'   <- parece el ESP32' if p['parece_esp32'] else ''}")
        raise SystemExit(0)

    cliente = ClienteESP32Serie()
    cliente.habilitado = True
    if not cliente._vivo:
        cliente.puerto = cliente.puerto or detectar_puerto()
        cliente._abrir()
    if not cliente._vivo:
        print("No hay ESP32 por USB. Revisa el cable y usa --listar.")
        raise SystemExit(1)

    lectura = cliente.lectura(timeout=3.0)
    print(f"Lectura: {lectura}")
    if args.probar_led:
        print("prueba de LED:", "ok" if cliente.probar_leds() else "sin respuesta")
    if args.estado:
        r = cliente.ordenar({"estado": args.estado}, "estado")
        print(f"estado {args.estado}:", r)
    if args.servo is not None:
        r = cliente.ordenar({"servo": args.servo}, "servo")
        print(f"servo {args.servo}:", r)
    time.sleep(1.2)
    print(cliente.estadisticas())
    cliente.cerrar()
