# Firmware del ESP32

Carpeta `semaforo_esp32/`: el programa que va grabado en el ESP32. Cumple el
contrato de [`docs/contrato_esp32.md`](../docs/contrato_esp32.md).

## 1. Que Windows vea el ESP32

Al conectarlo por USB debe aparecer un puerto COM (Administrador de dispositivos
→ Puertos COM y LPT). Si no aparece:

- **Cable.** Muchos cables USB son solo de carga. Probar con otro; el que viene
  con los celulares suele servir.
- **Driver.** Según el chip USB-serie de la placa:
  - Silicon Labs CP210x → https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers
  - WCH CH340 → https://www.wch.cn/downloads/CH341SER_EXE.html
  Con el driver instalado, desconectar y volver a conectar.

Comprobar desde el proyecto:

```bash
.venv\Scripts\python.exe -c "import serial.tools.list_ports as l; [print(p) for p in l.comports()]"
```

## 2. Preparar el IDE de Arduino (una sola vez)

1. **Placas ESP32:** Archivo → Preferencias → *Gestor de URLs adicionales de
   tarjetas*, pegar
   `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`.
   Luego Herramientas → Placa → Gestor de tarjetas → buscar **esp32** (Espressif) → instalar.
2. **Bibliotecas:** Herramientas → Administrar bibliotecas → instalar
   **ArduinoJson** (Benoit Blanchon) y **ESP32Servo** (Kevin Harrington).

## 3. Configurar y grabar

1. Abrir `firmware/semaforo_esp32/semaforo_esp32.ino` en el IDE.
2. En la pestaña `config.h` poner:
   - `REDES`: las redes Wi-Fi conocidas (casa **y** hotspot del celular). El ESP32
     se conecta a la que encuentre, así el mismo firmware sirve en todos lados.
   - `PIN_SERVO` si usan otro pin (la especificación no lo define; está en 13).
   - **No hace falta la IP de la laptop**: el ESP32 la descubre solo (abajo).
3. Herramientas → Placa → **ESP32 Dev Module**. Herramientas → Puerto → el COM del ESP32.
4. Botón **Subir**. Si se queda en "Connecting...", mantener pulsado **BOOT** en la placa hasta que empiece a escribir.
5. Herramientas → **Monitor serie**, a **115200** baudios. Al conectarse imprime:

```
================================================
  IP del ESP32:  192.168.1.50
  Ponerla en config.yaml -> esp32.ip
================================================
```

y después una línea por segundo con dB, LED, servo y estado del Wi-Fi.

## 4. Conectar con la laptop: por el cable USB

Es el modo normal (`MODO_WIFI false` en `config.h`, `esp32.transporte: usb` en
`config.yaml`). El mismo cable que sirve para grabar sirve para hablar, a
**921 600 baudios** (`BAUDIOS_SERIE` en `config.h` = `esp32.baudios` en
`config.yaml`): el ESP32 manda una línea `>{json}` por segundo con su lectura y
acepta órdenes JSON (`{"estado":"ROJO"}`, `{"servo":0|1|2}`, `{"prueba_led":1}`,
`{"lectura":1}`, `{"audio":1}`/`{"audio":0}`). Con el audio activado manda
además el sonido del INMP441: una línea `@<n>:<base64>` cada 32 ms (512
muestras PCM de 16 bits a 16 kHz, ~43 KB/s). Si abres el monitor serie de
Arduino, ponlo en 921600 o verás caracteres raros.
`run_session.py` abre el puerto solo; no hay nada que configurar.

Para comprobarlo (con el monitor serie de Arduino **cerrado**):

```bash
.venv\Scripts\python.exe scripts\probar_esp32.py
```

Al abrir el puerto la placa se reinicia a propósito (como en el monitor serie
de Arduino): si el puerto estuvo cerrado un rato mientras el ESP32 seguía
enviando, el chip USB-serie CP2102 se atasca y repite el mismo paquete sin
parar; el reinicio lo limpia. Tarda unos 3 s en volver y el color se vuelve a
mandar solo.

## 4b. Por Wi-Fi (opcional): se descubren solos

Con `MODO_WIFI true` en `config.h` y `esp32.transporte: wifi` en `config.yaml`.
Ni el ESP32 necesita la IP de la laptop ni la laptop la del ESP32. El ESP32
anuncia por UDP (puerto 8765) "aquí estoy" cada 3 s; `serve_dashboard.py` lo
escucha, se guarda su IP en `data/esp32_descubierto.json` y le responde con la
dirección a la que enviar lecturas. El ESP32 la guarda en su memoria (sobrevive a
reinicios) y desde entonces manda ahí.

Por eso funciona igual en casa, en la universidad o con hotspot: basta con que los
dos estén en la misma red y `serve_dashboard.py` esté corriendo.

Para comprobarlo, con `serve_dashboard.py` en una terminal, en otra:

```bash
.venv\Scripts\python.exe scripts\probar_esp32.py --ip <IP del ESP32> --guardar
```

Prueba los cuatro sentidos (laptop→ESP32 lectura, estado, servo; ESP32→laptop) y
dice cuál falla y qué revisar. Con `--guardar` deja la IP en `config.yaml`.

## 5. En la universidad

Por cable USB no hay nada que hacer: enchufar y correr. Lo que sigue es solo
para el modo Wi-Fi.

El Wi-Fi institucional **no sirve**: pide usuario y contraseña (WPA2-Enterprise) y
aísla a los dispositivos entre sí. Lo que funciona:

1. Activar el **hotspot de un celular** (nombre y clave puestos en `REDES` de
   `config.h` desde antes).
2. Conectar la **laptop** a ese hotspot.
3. Encender el ESP32: se conecta al hotspot y en unos segundos se descubren.
4. `serve_dashboard.py` y `run_session.py` como siempre.

Nada que regrabar ni que editar. Si el hotspot tiene la opción "aislar
clientes" o "AP isolation", debe estar desactivada (por defecto lo está).

## Problemas típicos

| Síntoma | Causa habitual |
|---|---|
| No aparece puerto COM | Cable de solo carga, o falta el driver CP210x/CH340. |
| "Connecting..." eterno al subir | Mantener BOOT pulsado durante la subida. |
| Se conecta al Wi-Fi pero la laptop no le llega | Red que aísla clientes (universidad). Usar hotspot del celular para ambos. |
| La laptop llega al ESP32 pero no al revés | `LAPTOP_URL` mal, o el firewall de Windows bloqueó Python la primera vez: permitirlo en redes privadas. |
| `nivel_db` en 0 o `mic: ceros` | INMP441 sin señal: SCK D19, WS D22, SD D21, L/R a GND (o 3V3), VDD a 3V3. Con multímetro en DC: SCK y WS deben marcar ~1,6 V (reloj), SD entre 0,2 y 1,6 V (el micrófono responde). Si SD marca 3,3 V fijo, el micrófono no recibe reloj. |
| `nivel_db` marca ~78 en un cuarto tranquilo | Falta la calibración: `DB_OFFSET` en `config.h` (medido: -38 deja el reposo en ~40 dB). |
| dB muy distintos de un sonómetro | Ajustar `DB_OFFSET` en `config.h`. |
| El servo tiembla | Alimentarlo aparte (no del pin 3.3 V del ESP32) con GND común. |
