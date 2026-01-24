#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>

// WiFi credentials
const char* ssid = "Njambe";
const char* password = "1234567890";

// Create web server on port 80
WebServer server(80);

// Traffic light pins for each road (Red, Yellow, Green)
// Road A
const int A_RED = 2;
const int A_YELLOW = 4;
const int A_GREEN = 5;

// Road B
const int B_RED = 27;
const int B_YELLOW = 26;
const int B_GREEN = 25;

// Road C
const int C_RED = 33;
const int C_YELLOW = 32;
const int C_GREEN = 14;

// Road D
const int D_RED = 23;
const int D_YELLOW = 22;
const int D_GREEN = 21;

// Detection counts
int count_A = 0;
int count_B = 0;
int count_C = 0;
int count_D = 0;
int emergency_road = -1; // -1 = none, 0=A, 1=B, 2=C, 3=D

// Traffic control state
enum State { DETECTING, CYCLING, EMERGENCY };
State currentState = DETECTING;

// Road priority array (indices: 0=A, 1=B, 2=C, 3=D)
int roadPriority[4];
int currentRoadIndex = 0;
int interruptedRoad = -1;

// Timing variables
unsigned long stateStartTime = 0;
unsigned long detectionDuration = 10000; // 10 seconds detection
unsigned long greenDuration = 25000;     // 25 seconds green
unsigned long yellowDuration = 5000;     // 5 seconds yellow

// Emergency handling
int emergencyAlternateIndex = 0;
int rotationOrder[4] = {3, 0, 1, 2}; // D, A, B, C

void setup() {
  Serial.begin(115200);
  
  // Setup all LED pins
  pinMode(A_RED, OUTPUT);
  pinMode(A_YELLOW, OUTPUT);
  pinMode(A_GREEN, OUTPUT);
  pinMode(B_RED, OUTPUT);
  pinMode(B_YELLOW, OUTPUT);
  pinMode(B_GREEN, OUTPUT);
  pinMode(C_RED, OUTPUT);
  pinMode(C_YELLOW, OUTPUT);
  pinMode(C_GREEN, OUTPUT);
  pinMode(D_RED, OUTPUT);
  pinMode(D_YELLOW, OUTPUT);
  pinMode(D_GREEN, OUTPUT);
  
  // All red initially
  setAllRed();
  
  // Connect to WiFi
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);
  
  WiFi.begin(ssid, password);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println();
  Serial.println("WiFi connected!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
  
  // Define server routes
  server.on("/update", HTTP_POST, handleUpdate);
  server.on("/", HTTP_GET, handleRoot);
  
  // Start server
  server.begin();
  Serial.println("HTTP server started");
  
  stateStartTime = millis();
}

void loop() {
  server.handleClient();
  
  unsigned long currentTime = millis();
  unsigned long elapsedTime = currentTime - stateStartTime;
  
  switch (currentState) {
    case DETECTING:
      handleDetecting(elapsedTime);
      break;
      
    case CYCLING:
      handleCycling(elapsedTime);
      break;
      
    case EMERGENCY:
      handleEmergency(elapsedTime);
      break;
  }
}

void handleDetecting(unsigned long elapsed) {
  // Wait for detection period to complete
  if (elapsed >= detectionDuration) {
    // Sort roads by priority
    sortRoadsByPriority();
    
    Serial.println("Detection complete. Priority order:");
    for (int i = 0; i < 4; i++) {
      Serial.print("Position ");
      Serial.print(i + 1);
      Serial.print(": Road ");
      Serial.print(getRoadName(roadPriority[i]));
      Serial.print(" (");
      Serial.print(getRoadCount(roadPriority[i]));
      Serial.println(" cars)");
    }
    
    // Start cycling
    currentRoadIndex = 0;
    currentState = CYCLING;
    stateStartTime = millis();
    startGreenPhase(roadPriority[currentRoadIndex]);
  }
}

void handleCycling(unsigned long elapsed) {
  // Check for emergency override
  if (emergency_road != -1) {
    Serial.print("EMERGENCY DETECTED on Road ");
    Serial.println(getRoadName(emergency_road));
    
    // Save interrupted road
    interruptedRoad = roadPriority[currentRoadIndex];
    
    // Start emergency protocol
    currentState = EMERGENCY;
    stateStartTime = millis();
    startYellowPhase(roadPriority[currentRoadIndex]);
    emergencyAlternateIndex = 0;
    return;
  }
  
  int currentRoad = roadPriority[currentRoadIndex];
  
  // Yellow warning for next road at 20s
  if (elapsed >= 20000 && elapsed < greenDuration) {
    int nextRoadIndex = (currentRoadIndex + 1) % 4;
    int nextRoad = roadPriority[nextRoadIndex];
    setRoadYellow(nextRoad);
  }
  
  // Transition to next road at 25s
  if (elapsed >= greenDuration) {
    setRoadRed(currentRoad);
    
    currentRoadIndex++;
    
    if (currentRoadIndex >= 4) {
      // Cycle complete - return to detection
      Serial.println("Cycle complete. Starting new detection phase.");
      currentState = DETECTING;
      stateStartTime = millis();
      setAllRed();
    } else {
      // Next road gets green
      stateStartTime = millis();
      startGreenPhase(roadPriority[currentRoadIndex]);
    }
  }
}

void handleEmergency(unsigned long elapsed) {
  // Phase 1: Yellow transition (5s)
  if (elapsed < yellowDuration) {
    // Yellow already set when entering emergency state
    return;
  }
  
  // Phase 2: Emergency road green OR alternate road green
  if (elapsed < yellowDuration + greenDuration) {
    if (elapsed == yellowDuration) {
      // Determine which road gets green
      if (emergencyAlternateIndex % 2 == 0) {
        // Emergency road's turn
        Serial.print("Emergency road ");
        Serial.print(getRoadName(emergency_road));
        Serial.println(" getting green light");
        setAllRed();
        setRoadGreen(emergency_road);
      } else {
        // Alternate road's turn
        int altIndex = (emergencyAlternateIndex / 2) % 4;
        int altRoad = rotationOrder[altIndex];
        Serial.print("Alternate road ");
        Serial.print(getRoadName(altRoad));
        Serial.println(" getting green light");
        setAllRed();
        setRoadGreen(altRoad);
      }
    }
    
    // Yellow warning for next phase at 20s
    if (elapsed >= yellowDuration + 20000 && elapsed < yellowDuration + greenDuration) {
      int nextRoad;
      if (emergencyAlternateIndex % 2 == 0) {
        // Next is alternate road
        int altIndex = (emergencyAlternateIndex / 2) % 4;
        nextRoad = rotationOrder[altIndex];
      } else {
        // Next is emergency road
        nextRoad = emergency_road;
      }
      setRoadYellow(nextRoad);
    }
    
    return;
  }
  
  // Phase 3: Transition complete
  if (elapsed >= yellowDuration + greenDuration) {
    // Check if emergency still exists
    if (emergency_road == -1) {
      // Emergency cleared!
      Serial.println("Emergency cleared. Giving interrupted road its turn.");
      currentState = CYCLING;
      stateStartTime = millis();
      setAllRed();
      startGreenPhase(interruptedRoad);
      
      // Rebuild priority with interrupted road first
      int tempPriority[4];
      tempPriority[0] = interruptedRoad;
      int idx = 1;
      for (int i = 0; i < 4; i++) {
        if (roadPriority[i] != interruptedRoad) {
          tempPriority[idx++] = roadPriority[i];
        }
      }
      for (int i = 0; i < 4; i++) {
        roadPriority[i] = tempPriority[i];
      }
      currentRoadIndex = 0;
      interruptedRoad = -1;
    } else {
      // Emergency continues - alternate pattern
      emergencyAlternateIndex++;
      stateStartTime = millis();
      
      // Yellow phase for next
      if (emergencyAlternateIndex % 2 == 0) {
        setRoadYellow(emergency_road);
      } else {
        int altIndex = (emergencyAlternateIndex / 2) % 4;
        setRoadYellow(rotationOrder[altIndex]);
      }
    }
  }
}

void handleUpdate() {
  if (server.hasArg("plain")) {
    String body = server.arg("plain");
    
    StaticJsonDocument<300> doc;
    DeserializationError error = deserializeJson(doc, body);
    
    if (error) {
      Serial.print("JSON parse failed: ");
      Serial.println(error.c_str());
      server.send(400, "text/plain", "Invalid JSON");
      return;
    }
    
    // Extract regular counts
    count_A = doc["A"] | 0;
    count_B = doc["B"] | 0;
    count_C = doc["C"] | 0;
    count_D = doc["D"] | 0;
    
    // Check for emergency (assuming special field in JSON)
    if (doc.containsKey("emergency")) {
      String emergencyRoadName = doc["emergency"].as<String>();
      if (emergencyRoadName == "A") emergency_road = 0;
      else if (emergencyRoadName == "B") emergency_road = 1;
      else if (emergencyRoadName == "C") emergency_road = 2;
      else if (emergencyRoadName == "D") emergency_road = 3;
      else emergency_road = -1;
    } else {
      emergency_road = -1;
    }
    
    Serial.print("A: ");
    Serial.print(count_A);
    Serial.print(" | B: ");
    Serial.print(count_B);
    Serial.print(" | C: ");
    Serial.print(count_C);
    Serial.print(" | D: ");
    Serial.print(count_D);
    if (emergency_road != -1) {
      Serial.print(" | EMERGENCY: ");
      Serial.print(getRoadName(emergency_road));
    }
    Serial.println();
    
    server.send(200, "text/plain", "OK");
  } else {
    server.send(400, "text/plain", "No data received");
  }
}

void handleRoot() {
  String html = "<html><head><meta http-equiv='refresh' content='1'></head><body>";
  html += "<h1>Smart Traffic Light System</h1>";
  html += "<h2>Current State: ";
  if (currentState == DETECTING) html += "DETECTING";
  else if (currentState == CYCLING) html += "CYCLING";
  else html += "EMERGENCY";
  html += "</h2>";
  html += "<h3>Detection Counts:</h3>";
  html += "<p>Road A: " + String(count_A) + "</p>";
  html += "<p>Road B: " + String(count_B) + "</p>";
  html += "<p>Road C: " + String(count_C) + "</p>";
  html += "<p>Road D: " + String(count_D) + "</p>";
  if (emergency_road != -1) {
    html += "<h3 style='color:red'>EMERGENCY on Road " + String(getRoadName(emergency_road)) + "</h3>";
  }
  html += "</body></html>";
  
  server.send(200, "text/html", html);
}

void sortRoadsByPriority() {
  // Create array of road indices and counts
  int counts[4] = {count_A, count_B, count_C, count_D};
  for (int i = 0; i < 4; i++) {
    roadPriority[i] = i;
  }
  
  // Bubble sort (descending)
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3 - i; j++) {
      if (counts[roadPriority[j]] < counts[roadPriority[j + 1]]) {
        int temp = roadPriority[j];
        roadPriority[j] = roadPriority[j + 1];
        roadPriority[j + 1] = temp;
      }
    }
  }
}

void startGreenPhase(int road) {
  setAllRed();
  setRoadGreen(road);
  Serial.print("Road ");
  Serial.print(getRoadName(road));
  Serial.println(" - GREEN");
}

void startYellowPhase(int road) {
  setRoadYellow(road);
  Serial.print("Road ");
  Serial.print(getRoadName(road));
  Serial.println(" - YELLOW");
}

void setAllRed() {
  digitalWrite(A_RED, HIGH);
  digitalWrite(A_YELLOW, LOW);
  digitalWrite(A_GREEN, LOW);
  
  digitalWrite(B_RED, HIGH);
  digitalWrite(B_YELLOW, LOW);
  digitalWrite(B_GREEN, LOW);
  
  digitalWrite(C_RED, HIGH);
  digitalWrite(C_YELLOW, LOW);
  digitalWrite(C_GREEN, LOW);
  
  digitalWrite(D_RED, HIGH);
  digitalWrite(D_YELLOW, LOW);
  digitalWrite(D_GREEN, LOW);
}

void setRoadRed(int road) {
  switch(road) {
    case 0:
      digitalWrite(A_RED, HIGH);
      digitalWrite(A_YELLOW, LOW);
      digitalWrite(A_GREEN, LOW);
      break;
    case 1:
      digitalWrite(B_RED, HIGH);
      digitalWrite(B_YELLOW, LOW);
      digitalWrite(B_GREEN, LOW);
      break;
    case 2:
      digitalWrite(C_RED, HIGH);
      digitalWrite(C_YELLOW, LOW);
      digitalWrite(C_GREEN, LOW);
      break;
    case 3:
      digitalWrite(D_RED, HIGH);
      digitalWrite(D_YELLOW, LOW);
      digitalWrite(D_GREEN, LOW);
      break;
  }
}

void setRoadYellow(int road) {
  switch(road) {
    case 0:
      digitalWrite(A_YELLOW, HIGH);
      break;
    case 1:
      digitalWrite(B_YELLOW, HIGH);
      break;
    case 2:
      digitalWrite(C_YELLOW, HIGH);
      break;
    case 3:
      digitalWrite(D_YELLOW, HIGH);
      break;
  }
}

void setRoadGreen(int road) {
  switch(road) {
    case 0:
      digitalWrite(A_RED, LOW);
      digitalWrite(A_YELLOW, LOW);
      digitalWrite(A_GREEN, HIGH);
      break;
    case 1:
      digitalWrite(B_RED, LOW);
      digitalWrite(B_YELLOW, LOW);
      digitalWrite(B_GREEN, HIGH);
      break;
    case 2:
      digitalWrite(C_RED, LOW);
      digitalWrite(C_YELLOW, LOW);
      digitalWrite(C_GREEN, HIGH);
      break;
    case 3:
      digitalWrite(D_RED, LOW);
      digitalWrite(D_YELLOW, LOW);
      digitalWrite(D_GREEN, HIGH);
      break;
  }
}

char getRoadName(int road) {
  return 'A' + road;
}

int getRoadCount(int road) {
  switch(road) {
    case 0: return count_A;
    case 1: return count_B;
    case 2: return count_C;
    case 3: return count_D;
    default: return 0;
  }
}
