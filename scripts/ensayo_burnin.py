"""Ensayo Burn-in: registra la unidad durante horas sin tocarla.

Pregunta al ESP32 por Wi-Fi (GET /lectura) cada pocos segundos y deja el registro
que exige el expediente: tiempo continuo, reinicios, tasa de envío de datos e
intensidad de la señal. La temperatura interna y la tensión de la batería no las
mide el ESP32: se leen con termómetro y multímetro, y se escriben aquí mismo
mientras corre el ensayo.

    python scripts/ensayo_burnin.py --horas 2
    python scripts/ensayo_burnin.py --horas 2 --ip 172.20.10.3

Durante el ensayo:
    t 38.5   + Enter   anota 38.5 °C de temperatura interna
    v 3.92   + Enter   anota 3.92 V de batería
    fin      + Enter   termina antes de tiempo

Deja en reports/: burnin_<fecha>.csv (cada muestra), burnin_<fecha>.json (resumen
con los criterios de aprobación) y burnin_<fecha>.png (la gráfica).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg  # noqa: E402
from src.comm.descubrimiento import esp32_descubierto  # noqa: E402

COLUMNAS = ["hora", "segundo", "respondio", "uptime_s", "arranques", "nivel_db", "estado_led",
            "envios_ok", "envios_fallidos", "rssi", "temperatura_c", "bateria_v"]
TEMPERATURA_MAX_C = 45.0
TASA_MINIMA = 99.0
HORAS_MINIMAS = 2.0


def leer_consola(anotado: dict, detener: threading.Event) -> None:
    for linea in sys.stdin:
        partes = linea.strip().lower().split()
        if partes[:1] == ["fin"]:
            detener.set()
            return
        if len(partes) == 2 and partes[0] in ("t", "v"):
            try:
                valor = float(partes[1].replace(",", "."))
            except ValueError:
                print("  no entendí el número")
                continue
            anotado["temperatura_c" if partes[0] == "t" else "bateria_v"] = valor
            print(f"  anotado: {linea.strip()}")


def resumir(filas: list[dict], inicio: datetime) -> dict:
    respuestas = [f for f in filas if f["respondio"]]
    temperaturas = [f["temperatura_c"] for f in filas if f["temperatura_c"] != ""]
    tensiones = [f["bateria_v"] for f in filas if f["bateria_v"] != ""]
    rssi = [f["rssi"] for f in respuestas if f["rssi"] != ""]

    # Reinicio: el uptime retrocede o el contador de arranques sube.
    reinicios = sum(
        1 for a, b in zip(respuestas, respuestas[1:])
        if b["uptime_s"] < a["uptime_s"] or (b["arranques"] or 0) > (a["arranques"] or 0)
    )
    ok = fallidos = None
    if len(respuestas) >= 2 and respuestas[0]["envios_ok"] != "":
        ok = respuestas[-1]["envios_ok"] - respuestas[0]["envios_ok"]
        fallidos = respuestas[-1]["envios_fallidos"] - respuestas[0]["envios_fallidos"]
    tasa_esp32 = round(100.0 * ok / (ok + fallidos), 2) if ok is not None and ok + fallidos > 0 else None
    tasa_consulta = round(100.0 * len(respuestas) / len(filas), 2) if filas else 0.0
    # Si el contador se reinició a mitad de ensayo, la resta no vale: queda la de las consultas.
    tasa = tasa_esp32 if tasa_esp32 is not None and not reinicios else tasa_consulta
    horas = filas[-1]["segundo"] / 3600.0 if filas else 0.0

    return {
        "inicio": inicio.isoformat(timespec="seconds"),
        "horas": round(horas, 3),
        "muestras": len(filas),
        "consultas_respondidas_pct": tasa_consulta,
        "tasa_envio_esp32_pct": tasa_esp32,
        "tasa_envio_pct": tasa,
        "reinicios": reinicios,
        "temperatura_max_c": max(temperaturas) if temperaturas else None,
        "n_temperaturas": len(temperaturas),
        "bateria_inicial_v": tensiones[0] if tensiones else None,
        "bateria_final_v": tensiones[-1] if tensiones else None,
        "rssi_medio_dbm": round(sum(rssi) / len(rssi), 1) if rssi else None,
        "criterios": {
            "duracion_2h": horas >= HORAS_MINIMAS,
            "tasa_99": tasa >= TASA_MINIMA,
            "sin_reinicios": reinicios == 0,
            "temperatura_45c": (max(temperaturas) < TEMPERATURA_MAX_C) if temperaturas else None,
        },
    }


def graficar(filas: list[dict], ruta: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    min_ = [f["segundo"] / 60.0 for f in filas]
    fig, ejes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    ejes[0].plot(min_, [f["nivel_db"] if f["respondio"] else None for f in filas], lw=0.8, color="#1f6fb2")
    ejes[0].set_ylabel("Nivel (dB)")
    ejes[1].plot(min_, [f["rssi"] if f["rssi"] != "" else None for f in filas], lw=0.8, color="#6b7280")
    ejes[1].set_ylabel("RSSI (dBm)")
    puntos = [(m, f["temperatura_c"]) for m, f in zip(min_, filas) if f["temperatura_c"] != ""]
    if puntos:
        ejes[2].plot(*zip(*puntos), "o-", color="#c2410c", ms=3)
    ejes[2].axhline(TEMPERATURA_MAX_C, color="#b91c1c", ls="--", lw=0.8)
    ejes[2].set_ylabel("Temp. (°C)")
    ejes[2].set_xlabel("Minutos de ensayo")
    for eje in ejes:
        eje.grid(alpha=0.25)
    fig.tight_layout()
    # plt.savefig pasa por una ruta con acentos sin problema; cv2 es el que no.
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def main() -> int:
    import requests

    conf = cfg.cargar()["esp32"]
    ap = argparse.ArgumentParser(description="Ensayo Burn-in de la unidad")
    ap.add_argument("--horas", type=float, default=2.0)
    ap.add_argument("--cada", type=float, default=5.0, help="segundos entre muestras")
    ap.add_argument("--ip", help="IP del ESP32; por defecto la descubierta (serve_dashboard.py)")
    ap.add_argument("--puerto", type=int, default=None)
    args = ap.parse_args()

    visto = esp32_descubierto()
    ip = args.ip or (visto["ip"] if visto else conf["ip"])
    puerto = args.puerto or (visto["puerto"] if visto else conf["puerto"])
    url = f"http://{ip}:{puerto}/lectura"

    inicio = datetime.now()
    marca = inicio.strftime("%Y%m%d_%H%M")
    carpeta = cfg.ruta_de("reports")
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_csv = carpeta / f"burnin_{marca}.csv"

    anotado: dict = {}
    detener = threading.Event()
    threading.Thread(target=leer_consola, args=(anotado, detener), daemon=True).start()

    print(f"Ensayo de {args.horas:g} h contra {url}, una muestra cada {args.cada:g} s.")
    print("Anota con 't 38.5' (°C) y 'v 3.92' (V). 'fin' termina antes. Ctrl+C también.\n")

    filas: list[dict] = []
    t0 = time.time()
    fin = t0 + args.horas * 3600.0
    try:
        with ruta_csv.open("w", encoding="utf-8-sig", newline="") as fh:
            escritor = csv.DictWriter(fh, fieldnames=COLUMNAS)
            escritor.writeheader()
            while time.time() < fin and not detener.is_set():
                vuelta = time.time()
                fila = {c: "" for c in COLUMNAS}
                fila.update(hora=datetime.now().isoformat(timespec="seconds"),
                            segundo=round(vuelta - t0, 1), respondio=False)
                try:
                    r = requests.get(url, timeout=min(3.0, args.cada)).json()
                    fila["respondio"] = True
                    for c in ("uptime_s", "arranques", "nivel_db", "estado_led", "envios_ok", "envios_fallidos", "rssi"):
                        fila[c] = r.get(c, "")
                except Exception as exc:  # noqa: BLE001 - una falla es un dato del ensayo, no un error
                    print(f"  {fila['hora']} sin respuesta: {type(exc).__name__}")
                fila["temperatura_c"] = anotado.pop("temperatura_c", "")
                fila["bateria_v"] = anotado.pop("bateria_v", "")
                filas.append(fila)
                escritor.writerow(fila)
                fh.flush()   # si se va la luz, lo medido ya está en el disco
                if len(filas) % max(1, int(300 / args.cada)) == 0:
                    bien = sum(1 for f in filas if f["respondio"])
                    print(f"  {fila['segundo'] / 60:6.1f} min   respondió {100 * bien / len(filas):.1f} %   "
                          f"{fila['nivel_db']} dB   LED {fila['estado_led']}")
                detener.wait(max(0.0, args.cada - (time.time() - vuelta)))
    except KeyboardInterrupt:
        print("\nEnsayo cortado por el usuario.")

    if not filas:
        print("No se tomó ninguna muestra.", file=sys.stderr)
        return 1
    resumen = resumir(filas, inicio)
    (carpeta / f"burnin_{marca}.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    graficar(filas, carpeta / f"burnin_{marca}.png")

    print("\n" + json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"\nRegistro en {ruta_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
