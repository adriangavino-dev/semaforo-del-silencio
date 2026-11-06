// Configuración del ESP32 del Semáforo del Silencio.
// Lo único que hay que tocar para poner el equipo en marcha está en este archivo.
#pragma once

// ---------------------------------------------------------------- Modo
// Cómo habla el ESP32 con la laptop:
//   MODO_WIFI true  -> Wi-Fi + servidor HTTP + descubrimiento UDP + audio por
//                      UDP. Es la unidad autónoma: basta el switch, sin cable a
//                      la PC. Lo que está configurado.
//   MODO_WIFI false -> solo por el cable USB (para depurar en banco).
#define MODO_WIFI true

// ---------------------------------------------------------------- Wi-Fi
// (solo se usa si MODO_WIFI es true)
// Lista de redes conocidas. El ESP32 se conecta a la primera que encuentre, así
// que el mismo firmware sirve en casa y en la universidad sin regrabar.
//
// En la universidad el Wi-Fi institucional NO sirve: pide usuario y contraseña
// (WPA2-Enterprise) y aísla a los dispositivos entre sí. Lo que funciona es el
// HOTSPOT DE UN CELULAR: la laptop y el ESP32 se conectan a él.
struct RedWifi
{
  const char *ssid;
  const char *clave;
};
static const RedWifi REDES[] = {
    {"NOMBRE_WIFI_CASA", "CLAVE_WIFI_CASA"},
    {"NOMBRE_HOTSPOT", "CLAVE_HOTSPOT"},
};
#define N_REDES (sizeof(REDES) / sizeof(REDES[0]))
// Cuánto espera la red al arrancar antes de seguir sin ella. El semáforo no
// depende del Wi-Fi: la reconexión sigue en segundo plano.
#define ESPERA_WIFI_MS 8000

// ---------------------------------------------------------------- Laptop
// NO hace falta poner la IP de la laptop: el ESP32 la descubre solo. Cada pocos
// segundos anuncia por UDP "aquí estoy" y serve_dashboard.py le responde con la
// dirección a la que debe enviar las lecturas. Esa dirección queda guardada en
// la memoria del ESP32 y sobrevive a reinicios.
//
// LAPTOP_URL_INICIAL es solo un respaldo por si nunca llega respuesta.
#define LAPTOP_URL_INICIAL "http://192.168.18.58:8000/api/esp32/lectura"
#define PUERTO_DESCUBRIMIENTO 8765 // el mismo que dashboard.puerto_descubrimiento en config.yaml
#define ANUNCIAR_CADA_MS 3000
#define ENVIAR_CADA_MS 1000 // cada cuánto se manda una lectura a la laptop

// ---------------------------------------------------------------- Cable USB
// 921600 baudios: hace falta para mandar el audio del INMP441 a la laptop
// (16 kHz x 16 bits = 32 KB/s, ~43 KB/s en base64). Debe coincidir con
// esp32.baudios en config.yaml.
#define BAUDIOS_SERIE 921600
// Audio a la laptop: el INMP441 entrega 24 bits; se pasan a 16 desplazando
// AUDIO_DESPLAZAMIENTO bits. 14 deja un aula tranquila en ~-55 dBFS y una voz
// fuerte en ~-20 dBFS, sin saturar. Menos = más ganancia.
#define AUDIO_DESPLAZAMIENTO 14

// ---------------------------------------------------------------- Pines
// Los definió el equipo (sección 2.1 de la especificación).
#define PIN_I2S_SCK 19 // INMP441 SCK / BCLK
#define PIN_I2S_WS 22  // INMP441 WS / LRC
#define PIN_I2S_SD 21  // INMP441 SD      (L/R a GND o a 3V3: se leen los dos canales)
#define PIN_LED_VERDE    13   // D13 (D12 es pin de arranque del ESP32; se evita)
#define PIN_LED_AMARILLO 14   // D14 — cableado real del equipo
#define PIN_LED_ROJO     27   // D27 — cableado real del equipo (el PDF decía 19/18/23)
#define PIN_SERVO 26        // D26 — cableado real del equipo

// ---------------------------------------------------------------- Sonido
// Los mismos umbrales que audio.db en config.yaml de la laptop.
#define UMBRAL_AMARILLO 55.0
#define UMBRAL_ROJO 65.0
#define HISTERESIS_DB 2.0 // evita parpadeo alrededor del umbral

// Calibración: el INMP441 entrega amplitud, no dB SPL. Este desplazamiento se
// ajusta una vez comparando con un sonómetro (o una app de celular) en el aula:
// si el ESP32 marca 48 y el sonómetro 60, sumar 12.
#define DB_OFFSET -38.0 // medido 2026-09-20: el INMP441 marca ~78 en un cuarto tranquilo; -38 lo deja en ~40

// ---------------------------------------------------------------- Servo
// Ángulo de cada posición fija. El índice es lo que manda la laptop en /servo.
static const int ANGULOS_SERVO[] = {30, 90, 150}; // izquierda, centro, derecha
#define N_POSICIONES (sizeof(ANGULOS_SERVO) / sizeof(ANGULOS_SERVO[0]))
// Giro suave: el servo avanza de grado en grado en vez de saltar al ángulo final.
// MS_POR_GRADO = 25 -> 60° (de una posición a la siguiente) en 1,5 s.
#define MS_POR_GRADO 25
// Tiempo extra para que la cámara deje de vibrar al llegar.
#define ASENTAMIENTO_MS 400

// ---------------------------------------------------------------- Prioridad
// Cuánto dura la orden de color de la laptop sin que se refresque. La laptop la
// repite cada esp32.reenviar_estado_s (3 s), así que el color se mantiene todo
// lo que dure la sesión (1 h, 1 día). Si la laptop se apaga o se cae el Wi-Fi,
// a los 15 s el ESP32 vuelve a decidir solo por dB y no se queda congelado.
// 0 = la orden nunca caduca (solo para depurar por cable).
#define ORDEN_LAPTOP_MS 15000
