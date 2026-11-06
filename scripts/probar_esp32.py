"""Comprueba la conexión con el ESP32 real, paso por paso.

Con esp32.transporte: usb (lo normal) prueba por el cable: que el ESP32 mande
lecturas, que obedezca un color y que obedezca al servo.

Con esp32.transporte: wifi hace las cuatro cosas que tienen que funcionar por
la red, en orden, y dice cuál falla:

  1. ¿La laptop llega al ESP32?           GET  http://<ip>/lectura
  2. ¿El ESP32 obedece un estado?         POST /estado {"estado": "ROJO"} y comprueba el LED
  3. ¿El ESP32 obedece al servo?          POST /servo {"posicion": 1} (y vuelve a 0)
  4. ¿El ESP32 llega a la laptop?         mira si serve_dashboard.py recibió lecturas suyas

Si todo pasa, deja config.yaml apuntando al ESP32 (con --guardar).

Uso:
    python scripts/probar_esp32.py                       # USB (según config.yaml)
    python scripts/probar_esp32.py --puerto-serie COM3   # USB en un puerto concreto
    python scripts/probar_esp32.py --ip 192.168.1.50     # Wi-Fi
    python scripts/probar_esp32.py --ip 192.168.1.50 --guardar
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg


def paso(n: int, texto: str) -> None:
    print(f"\n[{n}] {texto}")


def ok(msg: str) -> None:
    print(f"    OK   {msg}")


def fallo(msg: str, pista: str = "") -> None:
    print(f"    FALLO  {msg}")
    if pista:
        print(f"           {pista}")


def probar_usb(conf: dict, puerto: str | None, sin_servo: bool) -> int:
    """Las tres cosas que tienen que funcionar por el cable."""
    from src.comm.esp32_serial import ClienteESP32Serie, puertos_disponibles

    if puerto:
        conf["esp32"]["puerto_serie"] = puerto
    conf["esp32"]["habilitado"] = True
    cliente = ClienteESP32Serie(conf)
    todo_bien = True

    paso(1, f"El ESP32 está conectado por USB ({cliente.puerto or 'puerto no encontrado'})")
    if not cliente.estadisticas()["conectado"]:
        fallo(cliente.ultimo_error or "no se pudo abrir el puerto",
              "puertos que ve Windows: " + (", ".join(f"{p['puerto']} ({p['descripcion']})" for p in puertos_disponibles()) or "ninguno")
              + "\n           ¿Está el monitor serie de Arduino abierto? Solo un programa puede usar el puerto.")
        return 1
    lectura = cliente.lectura(timeout=8.0)   # el ESP32 se reinicia al abrir el puerto
    if lectura:
        ok(f"manda lecturas: {lectura.get('nivel_db')} dB, LED {lectura.get('estado_led')}, "
           f"servo {lectura.get('posicion_servo')}, encendido hace {lectura.get('uptime_s')} s, modo {lectura.get('modo')}")
    else:
        fallo("el puerto abre pero no llega ninguna lectura en 8 s",
              "¿El firmware grabado es el nuevo (con protocolo por USB)? Regrabar desde firmware/semaforo_esp32.")
        cliente.cerrar()
        return 1

    paso(2, "El ESP32 obedece un estado (ROJO durante unos segundos)")
    r = cliente.ordenar({"estado": "ROJO"}, "estado")
    time.sleep(1.2)
    led = (cliente.lectura() or {}).get("estado_led")
    if r and r.get("ok") and led == "ROJO":
        ok("el LED rojo se encendió por orden de la laptop")
    else:
        fallo(f"respuesta {r}, LED dice {led}")
        todo_bien = False
    cliente.ordenar({"estado": "VERDE"}, "estado")

    if not sin_servo:
        paso(3, "El ESP32 obedece al servo (posición 1 y vuelta a 0)")
        r = cliente.ordenar({"servo": 1}, "servo")
        if r and r.get("ok"):
            time.sleep(r.get("tiempo_giro_ms", 800) / 1000 + 1.2)
            pos = (cliente.lectura() or {}).get("posicion_servo")
            if pos == 1:
                ok(f"giró a la posición 1 (tiempo de giro {r.get('tiempo_giro_ms')} ms)")
            else:
                fallo(f"aceptó la orden pero reporta posición {pos}")
                todo_bien = False
            cliente.ordenar({"servo": 0}, "servo")
        else:
            fallo(f"respuesta {r}", "si no hay servo conectado, usar --sin-servo")
            todo_bien = False

    cliente.cerrar()
    print()
    print("TODO FUNCIONA por USB." if todo_bien else "Hay pasos fallidos; arriba dice cuáles y qué revisar.")
    return 0 if todo_bien else 1


def main() -> int:
    import requests

    conf = cfg.cargar()
    ap = argparse.ArgumentParser(description="Prueba la conexión con el ESP32")
    ap.add_argument("--ip", default=None, help="IP del ESP32 (fuerza la prueba por Wi-Fi)")
    ap.add_argument("--puerto", type=int, default=80)
    ap.add_argument("--puerto-serie", default=None, help="COMx del ESP32 (fuerza la prueba por USB)")
    ap.add_argument("--guardar", action="store_true", help="dejar config.yaml apuntando a este ESP32 (Wi-Fi)")
    ap.add_argument("--sin-servo", action="store_true", help="omitir la prueba del servo")
    args = ap.parse_args()

    transporte = str(conf["esp32"].get("transporte", "usb")).lower()
    if args.puerto_serie or (not args.ip and transporte in ("usb", "serie", "serial", "cable")):
        return probar_usb(conf, args.puerto_serie, args.sin_servo)

    if not args.ip:
        from src.comm.descubrimiento import esp32_descubierto

        visto = esp32_descubierto()
        if not visto:
            print("No se indicó --ip y el ESP32 no se ha anunciado por la red todavía.")
            print("Arranca serve_dashboard.py, enciende el ESP32 y espera unos segundos; o pasa --ip.")
            return 1
        args.ip, args.puerto = visto["ip"], visto["puerto"]
        print(f"Usando el ESP32 descubierto en {args.ip}:{args.puerto} (visto {visto['visto']})")

    base = f"http://{args.ip}:{args.puerto}"
    todo_bien = True

    # ------------------------------------------------------------ 1
    paso(1, f"La laptop llega al ESP32 en {base}")
    try:
        r = requests.get(f"{base}/lectura", timeout=3)
        r.raise_for_status()
        lectura = r.json()
        ok(f"responde: {lectura.get('nivel_db')} dB, LED {lectura.get('estado_led')}, "
           f"servo {lectura.get('posicion_servo')}, encendido hace {lectura.get('uptime_s')} s")
        if lectura.get("nivel_db", 0) < 0:
            print("    AVISO  nivel_db negativo: el INMP441 no está entregando muestras (revisar cableado I2S).")
    except Exception as exc:  # noqa: BLE001
        fallo(f"no responde: {exc}",
              "¿Están la laptop y el ESP32 en la misma red Wi-Fi? ¿La IP es la que imprimió el monitor serie?\n"
              "           En redes de universidad los dispositivos suelen estar aislados: probar con hotspot del celular.")
        return 1

    # ------------------------------------------------------------ 2
    paso(2, "El ESP32 obedece un estado (ROJO durante unos segundos)")
    try:
        r = requests.post(f"{base}/estado", json={"estado": "ROJO", "origen": "prueba"}, timeout=3)
        r.raise_for_status()
        time.sleep(0.5)
        led = requests.get(f"{base}/lectura", timeout=3).json().get("estado_led")
        if led == "ROJO":
            ok("el LED rojo se encendió por orden de la laptop")
        else:
            fallo(f"aceptó la orden pero el LED dice {led}", "revisar que rutaEstado llame a encender()")
            todo_bien = False
        requests.post(f"{base}/estado", json={"estado": "VERDE", "origen": "prueba"}, timeout=3)
    except Exception as exc:  # noqa: BLE001
        fallo(f"error al mandar el estado: {exc}")
        todo_bien = False

    # ------------------------------------------------------------ 3
    if not args.sin_servo:
        paso(3, "El ESP32 obedece al servo (posición 1 y vuelta a 0)")
        try:
            r = requests.post(f"{base}/servo", json={"posicion": 1}, timeout=3)
            r.raise_for_status()
            resp = r.json()
            time.sleep(resp.get("tiempo_giro_ms", 800) / 1000 + 0.3)
            pos = requests.get(f"{base}/lectura", timeout=3).json().get("posicion_servo")
            if pos == 1:
                ok(f"giró a la posición 1 (tiempo de giro {resp.get('tiempo_giro_ms')} ms)")
            else:
                fallo(f"aceptó la orden pero reporta posición {pos}")
                todo_bien = False
            requests.post(f"{base}/servo", json={"posicion": 0}, timeout=3)
        except Exception as exc:  # noqa: BLE001
            fallo(f"error con el servo: {exc}", "si no hay servo conectado, usar --sin-servo")
            todo_bien = False

    # ------------------------------------------------------------ 4
    paso(4, "El ESP32 llega a la laptop (serve_dashboard.py debe estar corriendo)")
    puerto = int(conf["dashboard"]["puerto"])
    try:
        antes = requests.get(f"http://127.0.0.1:{puerto}/api/esp32/ultima", timeout=3).json().get("lectura")
        print(f"    esperando {3} s a que el ESP32 mande algo...")
        time.sleep(3.0)
        ahora = requests.get(f"http://127.0.0.1:{puerto}/api/esp32/ultima", timeout=3).json().get("lectura")
        if ahora and (not antes or ahora["id"] != antes["id"]):
            ok(f"la laptop recibió una lectura nueva: {ahora['nivel_db']} dB, LED {ahora['estado_led']}")
        elif ahora:
            fallo("la laptop tiene lecturas viejas pero no llegan nuevas",
                  "revisar LAPTOP_URL en config.h: debe ser la IP que imprime serve_dashboard.py")
            todo_bien = False
        else:
            fallo("la laptop no ha recibido ninguna lectura del ESP32",
                  "revisar LAPTOP_URL en config.h y que el firewall de Windows permita el puerto "
                  f"{puerto} (la primera vez Windows pregunta; hay que aceptar en red privada).")
            todo_bien = False
    except Exception as exc:  # noqa: BLE001
        fallo(f"no llego al servidor de la laptop en localhost:{puerto}: {exc}",
              "arrancar antes: python scripts/serve_dashboard.py")
        todo_bien = False

    # ------------------------------------------------------------ resultado
    print()
    if todo_bien:
        print("TODO FUNCIONA. Laptop -> ESP32 y ESP32 -> laptop.")
    else:
        print("Hay pasos fallidos; arriba dice cuáles y qué revisar.")

    if args.guardar and todo_bien:
        ruta = cfg.RUTA_CONFIG
        texto = ruta.read_text(encoding="utf-8")
        import re

        texto = re.sub(r'(?m)^(esp32:\n(?:.*\n)*?  habilitado: )\w+', r"\1true", texto, count=1)
        texto = re.sub(r'(?m)^  ip: ".*"', f'  ip: "{args.ip}"', texto, count=1)
        texto = re.sub(r'(?m)^  puerto: \d+', f"  puerto: {args.puerto}", texto, count=1)
        ruta.write_text(texto, encoding="utf-8")
        print(f"config.yaml actualizado: esp32.habilitado=true, ip={args.ip}, puerto={args.puerto}")
    elif args.guardar:
        print("No se guardó config.yaml porque hubo fallos.")

    return 0 if todo_bien else 1


if __name__ == "__main__":
    raise SystemExit(main())
