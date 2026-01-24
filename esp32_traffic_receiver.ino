/*
ESP32 MQTT Receiver for Traffic Detection Data
Install libraries:
- Sketch -> Include Library -> Manage Libraries -> Search and install:
  1. "PubSubClient" by Nick O'Leary
  2. "ArduinoJson" by Benoit Blanchon

Then configure WiFi and MQTT settings below
*/

#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// WiFi Configuration
const char* ssid = "YOUR_WIFI_SSID";        // Change this
const char* password = "YOUR_WIFI_PASSWORD"; // Change this

// MQTT Configuration
const char* mqtt_server = "192.168.1.100";  // Change to your MQTT broker IP
const int mqtt_port = 1883;
const char* mqtt_topic = "traffic/detection";
const char* client_id = "esp32-traffic-receiver";

WiFiClient espClient;
PubSubClient client(espClient);

// Car count variables
int count_a = 0, count_b = 0, count_c = 0, count_d = 0, total = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n\nStarting ESP32 Traffic Detection Receiver");
  
  // Initialize pins if using LED indicators
  pinMode(LED_BUILTIN, OUTPUT);
  
  // Connect to WiFi
  setup_wifi();
  
  // Set MQTT server
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(mqtt_callback);
}

void setup_wifi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nFailed to connect to WiFi");
  }
}

void mqtt_callback(char* topic, byte* payload, unsigned int length) {
  // Parse JSON payload
  DynamicJsonDocument doc(256);
  deserializeJson(doc, payload, length);
  
  // Extract values
  count_a = doc["A"];
  count_b = doc["B"];
  count_c = doc["C"];
  count_d = doc["D"];
  total = doc["total"];
  
  // Print to Serial Monitor
  Serial.println("=== Traffic Detection Data Received ===");
  Serial.print("Square A (Top-Left): ");
  Serial.println(count_a);
  Serial.print("Square B (Top-Right): ");
  Serial.println(count_b);
  Serial.print("Square C (Bottom-Left): ");
  Serial.println(count_c);
  Serial.print("Square D (Bottom-Right): ");
  Serial.println(count_d);
  Serial.print("Total Cars: ");
  Serial.println(total);
  Serial.println("======================================");
  
  // Blink LED when data received
  digitalWrite(LED_BUILTIN, HIGH);
  delay(100);
  digitalWrite(LED_BUILTIN, LOW);
  
  // You can add custom logic here (e.g., trigger relay, display on LCD, etc.)
  handle_detection_data(count_a, count_b, count_c, count_d, total);
}

void handle_detection_data(int a, int b, int c, int d, int total) {
  // Add your custom logic here
  // Example: Send data to a display, trigger an alarm, control lights, etc.
  
  if (total > 4) {
    Serial.println("Alert: High traffic detected!");
  }
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection... ");
    
    if (client.connect(client_id)) {
      Serial.println("connected");
      client.subscribe(mqtt_topic);
      Serial.print("Subscribed to: ");
      Serial.println(mqtt_topic);
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

void loop() {
  // Check WiFi connection
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected, reconnecting...");
    setup_wifi();
  }
  
  // Check MQTT connection
  if (!client.connected()) {
    reconnect();
  }
  
  client.loop();
  delay(100);
}
