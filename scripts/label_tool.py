"""Etiquetado de fragmentos asistido por consola — Etapa 2.

Tiene dos modos.

El modo por lote asigna una clase a todos los fragmentos de una sesión de una
sola vez. Es el que conviene en la práctica: se graba una sesión entera con una
sola situación (por ejemplo dos minutos de varias personas hablando a la vez) y
después se etiqueta completa con un comando.

El modo interactivo reproduce fragmento por fragmento y pide la clase. Sirve para
sesiones mezcladas y para revisar a mano lo que quedó dudoso. Puede mostrar una
sugerencia a partir de lo que YAMNet reconoce, que es solo una ayuda: la etiqueta
final la decide la persona.

Uso:
    python scripts/label_tool.py --resumen
    python scripts/label_tool.py --sesion sesion01 --asignar murmullo
    python scripts/label_tool.py --sesion sesion02
    python scripts/label_tool.py --sesion sesion02 --sugerir
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg
from src.audio.dataset import escribir_etiquetas, leer_etiquetas  # noqa: E402

# Palabras de las clases de AudioSet que empujan hacia cada clase nuestra.
# Es solo una sugerencia para el modo interactivo, no una regla de decisión.
PISTAS = {
    "silencio": ("silence", "inside", "room tone"),
    "voz_docente": ("speech", "narration", "monologue", "male speech", "female speech"),
    "murmullo": ("babble", "chatter", "crowd", "hubbub", "conversation"),
    "ruido_brusco": ("bang", "slam", "clap", "shout", "yell", "screaming", "thump", "knock"),
}


def sugerir(top: list[tuple[str, float]]) -> str | None:
    """Propone una de nuestras cuatro clases a partir de las de AudioSet."""
    puntos: Counter[str] = Counter()
    for nombre, puntaje in top:
        bajo = nombre.lower()
        for clase, palabras in PISTAS.items():
            if any(p in bajo for p in palabras):
                puntos[clase] += puntaje
    return puntos.most_common(1)[0][0] if puntos else None


def mostrar_resumen(filas: dict[str, dict[str, str]], clases: list[str]) -> None:
    por_clase: Counter[str] = Counter()
    por_sesion: dict[str, Counter[str]] = {}
    for fila in filas.values():
        clase = fila["clase"] or "(sin etiquetar)"
        por_clase[clase] += 1
        por_sesion.setdefault(fila["sesion"], Counter())[clase] += 1

    print("\nFragmentos por clase")
    print("-" * 52)
    for clase in clases:
        n = por_clase.get(clase, 0)
        falta = max(0, 150 - n)
        estado = "cumple" if n >= 150 else f"faltan {falta}"
        print(f"  {clase:<15} {n:>5}   {estado}")
    sin = por_clase.get("(sin etiquetar)", 0)
    if sin:
        print(f"  {'(sin etiquetar)':<15} {sin:>5}")

    print("\nSesiones")
    print("-" * 52)
    for sesion in sorted(por_sesion):
        detalle = ", ".join(f"{c}: {n}" for c, n in sorted(por_sesion[sesion].items()))
        print(f"  {sesion:<15} {detalle}")

    sesiones_con_datos = {
        f["sesion"] for f in filas.values() if f["clase"]
    }
    print(f"\nSesiones etiquetadas: {len(sesiones_con_datos)} (la meta son 3 o más)")


def main() -> int:
    conf = cfg.cargar()
    clases: list[str] = conf["audio"]["clases"]

    ap = argparse.ArgumentParser(description="Etiquetado de fragmentos de audio")
    ap.add_argument("--sesion", help="sesión sobre la que trabajar")
    ap.add_argument("--asignar", choices=clases, help="asignar esta clase a toda la sesión")
    ap.add_argument("--sugerir", action="store_true", help="mostrar sugerencia de YAMNet")
    ap.add_argument("--rehacer", action="store_true", help="incluir los ya etiquetados")
    ap.add_argument("--resumen", action="store_true", help="solo mostrar el estado y salir")
    args = ap.parse_args()

    ruta_labels = cfg.ruta_de("data_audio_labels")
    filas = leer_etiquetas(ruta_labels)
    if not filas:
        print(f"No hay fragmentos registrados en {ruta_labels}.", file=sys.stderr)
        print("Corre antes: python scripts/make_clips.py --todas", file=sys.stderr)
        return 1

    if args.resumen:
        mostrar_resumen(filas, clases)
        return 0

    # ---------------------------------------------------------- modo por lote
    if args.asignar:
        if not args.sesion:
            print("--asignar necesita también --sesion", file=sys.stderr)
            return 1
        objetivo = [
            f for f in filas.values()
            if f["sesion"] == args.sesion and (args.rehacer or not f["clase"])
        ]
        if not objetivo:
            print(f"No hay fragmentos pendientes en la sesión '{args.sesion}'.")
            return 0
        for fila in objetivo:
            fila["clase"] = args.asignar
        escribir_etiquetas(filas, ruta_labels)
        print(f"{len(objetivo)} fragmentos de '{args.sesion}' etiquetados como '{args.asignar}'.")
        mostrar_resumen(filas, clases)
        return 0

    # -------------------------------------------------------- modo interactivo
    import sounddevice as sd
    import soundfile as sf

    pendientes = [
        f for f in sorted(filas.values(), key=lambda x: x["clip_id"])
        if (not args.sesion or f["sesion"] == args.sesion) and (args.rehacer or not f["clase"])
    ]
    if not pendientes:
        print("No queda nada por etiquetar.")
        mostrar_resumen(filas, clases)
        return 0

    yam = None
    if args.sugerir:
        from src.audio.features import Yamnet

        print("Cargando YAMNet para las sugerencias...")
        yam = Yamnet(conf).cargar()

    carpeta_clips = cfg.ruta_de("data_audio_clips")
    print(f"\n{len(pendientes)} fragmentos por etiquetar.")
    print("Teclas: " + "  ".join(f"[{i + 1}] {c}" for i, c in enumerate(clases)))
    print("        [r] repetir   [s] saltar   [q] guardar y salir\n")

    etiquetados = 0
    try:
        for i, fila in enumerate(pendientes, 1):
            ruta = carpeta_clips / fila["sesion"] / f"{fila['clip_id']}.wav"
            if not ruta.exists():
                print(f"  falta el archivo {ruta}, se omite")
                continue

            onda, sr = sf.read(str(ruta), dtype="float32")
            pista = ""
            if yam is not None:
                propuesta = sugerir(yam.clases_principales(onda, 4))
                pista = f"   sugerencia: {propuesta}" if propuesta else ""

            while True:
                sd.play(onda, sr)
                sd.wait()
                print(f"[{i}/{len(pendientes)}] {fila['clip_id']}{pista}")
                tecla = input("  clase> ").strip().lower()

                if tecla == "r":
                    continue
                if tecla == "s":
                    break
                if tecla == "q":
                    raise KeyboardInterrupt
                if tecla.isdigit() and 1 <= int(tecla) <= len(clases):
                    fila["clase"] = clases[int(tecla) - 1]
                    etiquetados += 1
                    break
                if tecla in clases:
                    fila["clase"] = tecla
                    etiquetados += 1
                    break
                print("  opción no válida")

            if etiquetados and etiquetados % 20 == 0:
                escribir_etiquetas(filas, ruta_labels)
                print(f"  ...guardado ({etiquetados} etiquetados)")
    except KeyboardInterrupt:
        print("\nInterrumpido.")
    finally:
        escribir_etiquetas(filas, ruta_labels)

    print(f"\n{etiquetados} fragmentos etiquetados en esta ronda.")
    mostrar_resumen(filas, clases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
