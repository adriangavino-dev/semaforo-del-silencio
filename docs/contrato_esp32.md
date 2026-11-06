# Contrato de comunicación laptop ↔ ESP32

Hay dos transportes con el mismo contenido. Todo va en JSON y ninguno de los
dos se bloquea esperando al otro: si un mensaje falla, se registra y se sigue.

## 0. Por cable USB (modo normal)

Puerto serie a 115200 baudios, un mensaje por línea. `MODO_WIFI false` en el
firmware y `esp32.transporte: usb` en `config.yaml`.

```
   LAPTOP (Python)                                  ESP32 (Arduino)
   ─────────────────                                ───────────────
   {"estado":"ROJO"}      ───────────────────────▶  enciende ese LED
                          ◀───────────────────────  >{"ok":true,"orden":"estado"}
   {"servo":1}            ───────────────────────▶  gira la cámara
                          ◀───────────────────────  >{"ok":true,"orden":"servo","posicion":1,"tiempo_giro_ms":800}
   {"prueba_led":1}       ───────────────────────▶  verde, amarillo, rojo x2
   {"lectura":1}          ───────────────────────▶  pide la lectura ahora
                          ◀───────────────────────  >{lectura}   (además, sola, cada 1 s)
```

Las respuestas y lecturas del ESP32 empiezan por `>`; las demás líneas son
mensajes para leer en el monitor (`[laptop] estado -> ROJO`, `[prueba] LED …`).
La lectura tiene los mismos campos que en el apartado 2 más `"modo":"usb"`.
En la laptop lo implementa `src/comm/esp32_serial.py`; las lecturas se guardan
en `esp32_lecturas` igual que si llegaran por la red.

## 0b. Por Wi-Fi (opcional)

Los dos dispositivos están en la misma red. Cada uno es servidor HTTP para lo
que recibe y cliente para lo que envía. `MODO_WIFI true` y `esp32.transporte: wifi`.

```
   LAPTOP (Python)                                  ESP32 (Arduino)
   servidor en :8000                                servidor en :80
   ─────────────────                                ───────────────
   POST /estado  ─────────────────────────────────▶  enciende el LED que le dicen
   POST /servo   ─────────────────────────────────▶  mueve la cámara a la posición
                 ◀─────────────────────────────────  POST /api/esp32/lectura
                                                     (cada 1 s: dB, LED, servo)
```

El ESP32 conserva **siempre** su propio control por decibeles: si la laptop se
apaga o pierde el Wi-Fi, el semáforo sigue funcionando con el INMP441.

---

## 1. Laptop → ESP32

### `POST http://<IP_ESP32>/estado`

Se envía **solo cuando el color cambia**, y como máximo una vez por segundo.

```json
{
  "estado": "AMARILLO",
  "origen": "ia",
  "clase_audio": "murmullo",
  "n_de_pie": 4,
  "evento": null,
  "timestamp": "2026-09-14T10:34:12"
}
```

| Campo | Valores | Qué hacer con él |
|---|---|---|
| `estado` | `VERDE` \| `AMARILLO` \| `ROJO` | Encender ese LED y apagar los otros dos. |
| `origen` | siempre `"ia"` | Distingue la orden de la laptop de la decisión propia. |
| `clase_audio` | `silencio` \| `voz_docente` \| `murmullo` \| `ruido_brusco` \| `null` | Informativo. |
| `n_de_pie` | entero o `null` | Informativo. |
| `evento` | `movimiento_anomalo` \| `ruido_brusco` \| `null` | Informativo. |
| `timestamp` | ISO 8601 | Informativo. |

Respuesta esperada: `200` con `{"ok": true}`.

**Regla de prioridad recomendada en el firmware:** respetar el estado de la
laptop durante ~10 s desde que llega; si en ese tiempo no llega otro, volver al
control propio por dB. Así la laptop manda mientras está viva y el ESP32 se
recupera solo si la laptop desaparece.

### `POST http://<IP_ESP32>/servo`

```json
{ "posicion": 1 }
```

`posicion` es un entero de `0` a `servo.posiciones - 1` (config.yaml). El
firmware mapea cada índice a un ángulo fijo (por ejemplo 0 → 45°, 1 → 135°).

Respuesta esperada: `200` con `{"ok": true, "posicion": 1, "tiempo_giro_ms": 800}`.

`tiempo_giro_ms` es opcional: cuánto tarda el servo en llegar y asentarse. La
laptop pausa el análisis durante `servo.tiempo_giro_s` (config.yaml), que debe
ser mayor que ese valor.

La laptop **espera esta respuesta** (hasta `esp32.timeout_s`). Si no llega,
asume que la cámara no se movió y sigue analizando en la posición anterior.

---

## 2. ESP32 → Laptop

### `POST http://<IP_LAPTOP>:8000/api/esp32/lectura`

Enviar **cada 1-2 segundos**.

```json
{
  "nivel_db": 61.2,
  "estado_led": "AMARILLO",
  "posicion_servo": 1,
  "uptime_s": 1234
}
```

| Campo | Tipo | Significado |
|---|---|---|
| `nivel_db` | número | Nivel sonoro medido con el INMP441. |
| `estado_led` | `VERDE` \| `AMARILLO` \| `ROJO` | El LED que está encendido ahora mismo (sea por orden de la laptop o por decisión propia). |
| `posicion_servo` | entero | Posición actual de la cámara. |
| `uptime_s` | entero | Segundos desde que arrancó. Sirve para detectar reinicios. |

Cualquier campo adicional se guarda tal cual en la base de datos.

Respuesta: `200` con `{"ok": true, "id": 123}`. Si la laptop no responde, el
ESP32 descarta la lectura y sigue: **nunca reintentar en bucle ni bloquear la
medición**.

La IP de la laptop la imprime `serve_dashboard.py` al arrancar:

```
El ESP32 debe enviar sus lecturas a:  http://192.168.1.37:8000/api/esp32/lectura
```

Conviene fijar la IP de la laptop en el router (reserva DHCP) para no tener
que cambiarla en el firmware cada vez.

### Endpoints de consulta (los usa el tablero)

- `GET /api/esp32/ultima` → última lectura recibida.
- `GET /api/esp32/lecturas?n=300` → últimas N lecturas.
- `GET /api/salud` → `{"ok": true}` para comprobar que el servidor responde.

---

## 2b. Descubrimiento (UDP 8765)

Para que ninguno necesite la IP del otro:

- El ESP32 envía por difusión UDP al puerto 8765, cada 3 s hasta que le
  respondan y luego cada 30 s:
  `{"dispositivo":"semaforo-esp32","ip":"192.168.43.12","puerto":80,"led":"VERDE"}`
- La laptop (`serve_dashboard.py`) contesta al mismo puerto:
  `{"laptop_url":"http://192.168.43.7:8000/api/esp32/lectura"}`
- El ESP32 guarda esa URL en memoria no volátil y la usa para sus envíos.
- La laptop guarda la IP del ESP32 en `data/esp32_descubierto.json`; tiene
  prioridad sobre `esp32.ip` de `config.yaml` durante 10 minutos desde el
  último anuncio.

## 3. Dónde queda todo guardado

`data/semaforo.db` (SQLite). Tablas:

| Tabla | Contenido |
|---|---|
| `sesiones` | una fila por sesión, con el resumen JSON completo |
| `lecturas` | traza segundo a segundo de cada sesión (estado, audio, personas, posición del servo) |
| `eventos` | movimientos anómalos y ruidos bruscos, con zona y posición |
| `esp32_lecturas` | todo lo que envió el ESP32 |
| `servo_ordenes` | cada orden de giro y si el ESP32 la aceptó |

Consultar desde consola:

```bash
python -m src.db
python -m src.db --sql "SELECT recibido, nivel_db, estado_led FROM esp32_lecturas ORDER BY id DESC LIMIT 20"
```

O abrir el archivo con [DB Browser for SQLite](https://sqlitebrowser.org/).

---

## 4. Probar sin el circuito

`scripts/simular_esp32.py` se comporta exactamente como este contrato: es
servidor en `:8081`, acepta `/estado` y `/servo`, mide dB inventados con
histéresis y manda lecturas cada segundo. Con `esp32.ip: 127.0.0.1` y
`esp32.puerto: 8081` en config.yaml, todo el sistema corre igual que con el
hardware.

Cuando el ESP32 real esté en la red: `esp32.ip` a su IP, `esp32.puerto: 80`, y
apagar el simulador. Nada más cambia.

---

## 5. Esbozo del firmware (Arduino, ESP32)

Lo mínimo para cumplir el contrato. Falta la lectura real del INMP441 por I2S,
que ya la tiene definida el equipo (pines 26/25/33), y la calibración de dB.

```cpp
#include <WiFi.h>
#include <WebServer.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <ESP32Servo.h>

const char* SSID = "...";
const char* PASS = "...";
const char* LAPTOP = "http://192.168.1.37:8000/api/esp32/lectura";

const int LED_VERDE = 19, LED_AMARILLO = 18, LED_ROJO = 23, PIN_SERVO = 13;
const float UMBRAL_AMARILLO = 55.0, UMBRAL_ROJO = 65.0, HISTERESIS = 2.0;
const int ANGULOS[] = {45, 135};              // posición 0 y 1

WebServer servidor(80);
Servo servo;
String led = "VERDE";
String ordenLaptop = "";
unsigned long ordenHasta = 0;
int posicion = 0;

void encender(const String& estado) {
  led = estado;
  digitalWrite(LED_VERDE,    estado == "VERDE");
  digitalWrite(LED_AMARILLO, estado == "AMARILLO");
  digitalWrite(LED_ROJO,     estado == "ROJO");
}

float medirDb() {
  // TODO: RMS del INMP441 por I2S -> dB. Aquí solo el esqueleto.
  return 50.0;
}

void decidirPropio(float db) {
  if (db > UMBRAL_ROJO + HISTERESIS)          encender("ROJO");
  else if (db > UMBRAL_AMARILLO + HISTERESIS) encender("AMARILLO");
  else if (db < UMBRAL_AMARILLO - HISTERESIS) encender("VERDE");
}

void rutaEstado() {
  StaticJsonDocument<256> doc;
  if (deserializeJson(doc, servidor.arg("plain"))) {
    servidor.send(400, "application/json", "{\"ok\":false,\"error\":\"json\"}");
    return;
  }
  String estado = doc["estado"] | "";
  if (estado != "VERDE" && estado != "AMARILLO" && estado != "ROJO") {
    servidor.send(400, "application/json", "{\"ok\":false,\"error\":\"estado\"}");
    return;
  }
  ordenLaptop = estado;
  ordenHasta = millis() + 10000;              // respetarla 10 s
  encender(estado);
  servidor.send(200, "application/json", "{\"ok\":true}");
}

void rutaServo() {
  StaticJsonDocument<64> doc;
  if (deserializeJson(doc, servidor.arg("plain")) || !doc.containsKey("posicion")) {
    servidor.send(400, "application/json", "{\"ok\":false,\"error\":\"posicion\"}");
    return;
  }
  posicion = doc["posicion"];
  servo.write(ANGULOS[posicion]);
  servidor.send(200, "application/json",
    "{\"ok\":true,\"posicion\":" + String(posicion) + ",\"tiempo_giro_ms\":800}");
}

void enviarLectura(float db) {
  HTTPClient http;
  http.begin(LAPTOP);
  http.setTimeout(1500);
  http.addHeader("Content-Type", "application/json");
  String cuerpo = "{\"nivel_db\":" + String(db, 1) +
                  ",\"estado_led\":\"" + led + "\"" +
                  ",\"posicion_servo\":" + String(posicion) +
                  ",\"uptime_s\":" + String(millis() / 1000) + "}";
  http.POST(cuerpo);                          // se ignora el resultado: nunca bloquear
  http.end();
}

void setup() {
  pinMode(LED_VERDE, OUTPUT); pinMode(LED_AMARILLO, OUTPUT); pinMode(LED_ROJO, OUTPUT);
  servo.attach(PIN_SERVO);
  servo.write(ANGULOS[0]);
  WiFi.begin(SSID, PASS);
  while (WiFi.status() != WL_CONNECTED) delay(200);
  servidor.on("/estado", HTTP_POST, rutaEstado);
  servidor.on("/servo",  HTTP_POST, rutaServo);
  servidor.begin();
  encender("VERDE");
}

unsigned long ultimoEnvio = 0;

void loop() {
  servidor.handleClient();
  float db = medirDb();
  if (ordenLaptop == "" || millis() > ordenHasta) {
    ordenLaptop = "";
    decidirPropio(db);                        // control propio si la laptop calla
  }
  if (millis() - ultimoEnvio > 1000) {
    ultimoEnvio = millis();
    enviarLectura(db);
  }
}
```

Bibliotecas: `ArduinoJson`, `ESP32Servo` (gestor de bibliotecas del IDE de
Arduino). `WiFi`, `WebServer` y `HTTPClient` vienen con el núcleo del ESP32.
