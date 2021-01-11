/*
  Serial Event example

  When new serial data arrives, this sketch adds it to a String.
  When a newline is received, the loop prints the string and clears it.
  This example is extended to control a NES Classic modified controller.

  NOTE: The serialEvent() feature is not available on the Leonardo, Micro, or
  other ATmega32U4 based boards.

  created 09 Jan 2021
  by Conrad Salinas

  This example code is in the public domain.
  http://www.arduino.cc/en/Tutorial/SerialEvent

  Board: Arduino BLE 33 Sense

  Notes:
  NES Controller action mapping
  1 [['NOOP'], 
  2 ['right'], 
  3 ['right', 'A'], 
  4 ['right', 'B'], 
  5 ['right', 'A', 'B'], 
  6 ['A'], 
  7 ['left'], 
  8 ['left', 'A'], 
  9 ['left', 'B'], 
  10 ['left', 'A', 'B'], 
  11 ['down'], 
  12 ['up']]
  NES Controller action mapping misc
  13 [['start'],
  14 ['select']]
  NES Classic IO mapping
  15 [['power'],
  16 ['reset']
*/

/* DEFINES */
#define NES_CTRL_UP       12
#define NES_CTRL_DOWN     11
#define NES_CTRL_LEFT     10
#define NES_CTRL_RIGHT    9
#define NES_CTRL_A        8
#define NES_CTRL_B        7
#define NES_CTRL_START    6
#define NES_CTRL_SELECT   5
#define NES_IO_POWER      4
#define NES_IO_RESET      3

#define ACTION_1          1
#define ACTION_2          2
#define ACTION_3          3
#define ACTION_4          4
#define ACTION_5          5
#define ACTION_6          6
#define ACTION_7          7
#define ACTION_8          8
#define ACTION_9          9
#define ACTION_10         10
#define ACTION_11         11
#define ACTION_12         12
#define ACTION_13         13
#define ACTION_14         14
#define ACTION_15         15
#define ACTION_16         16

#define BUTTON_ENABLE     LOW 
#define BUTTON_DISABLE    HIGH   

/* GLOBALS */
String inputString = "";         // a String to hold incoming data
bool stringComplete = false;  // whether the string is complete

/* PROTOTYPES */
void serialEvent(void);
void action_output(int);
void action_clear(int);

/* INIT */
void setup() {
  // initialize serial:
  Serial.begin(9600);
  // reserve 200 bytes for the inputString:
  inputString.reserve(200);
  // init IO
  pinMode(12, OUTPUT);
  pinMode(11, OUTPUT);
  pinMode(10, OUTPUT);
  pinMode(9, OUTPUT);
  pinMode(8, OUTPUT);
  pinMode(7, OUTPUT);
  pinMode(6, OUTPUT);
  pinMode(5, OUTPUT);
  pinMode(4, OUTPUT);
  pinMode(3, OUTPUT);  
  digitalWrite(12, BUTTON_DISABLE);
  digitalWrite(11, BUTTON_DISABLE);
  digitalWrite(10, BUTTON_DISABLE);
  digitalWrite(9, BUTTON_DISABLE);
  digitalWrite(8, BUTTON_DISABLE); 
  digitalWrite(7, BUTTON_DISABLE);
  digitalWrite(6, BUTTON_DISABLE);
  digitalWrite(5, BUTTON_DISABLE);
  digitalWrite(4, BUTTON_DISABLE);
  digitalWrite(3, BUTTON_DISABLE);
}

/* LOOP */
void loop() {
  while (Serial.available()) {
    // get the new byte:
    char inChar = (char)Serial.read();
    // got new line so complete string
    if (inChar == '\n') {
      stringComplete = true;
    }
    else {
      // add it to the inputString:
      inputString += inChar;      
    }
    // string is complete so execute actions
    if(stringComplete) {
      // Serial.println(inputString);
      int action = inputString.toInt();
      switch(action) {
        case ACTION_1: // ['NOOP']
          action_clear(NES_CTRL_UP);
          action_clear(NES_CTRL_DOWN);
          action_clear(NES_CTRL_LEFT);
          action_clear(NES_CTRL_RIGHT);
          action_clear(NES_CTRL_A);
          action_clear(NES_CTRL_B);
          action_clear(NES_CTRL_START);
          action_clear(NES_CTRL_SELECT);
          action_clear(NES_IO_POWER);
          action_clear(NES_IO_RESET);                                        
          break;
        case ACTION_2: // ['right']
          action_clear(NES_CTRL_UP);
          action_clear(NES_CTRL_DOWN);
          action_clear(NES_CTRL_LEFT);
          action_output(NES_CTRL_RIGHT);
          action_clear(NES_CTRL_A);
          action_clear(NES_CTRL_B);
          action_clear(NES_CTRL_START);
          action_clear(NES_CTRL_SELECT);
          action_clear(NES_IO_POWER);
          action_clear(NES_IO_RESET);                                        
          break;
        case ACTION_3: // ['right', 'A']
          action_clear(NES_CTRL_UP);
          action_clear(NES_CTRL_DOWN);
          action_clear(NES_CTRL_LEFT);
          action_output(NES_CTRL_RIGHT);
          action_output(NES_CTRL_A);
          action_clear(NES_CTRL_B);
          action_clear(NES_CTRL_START);
          action_clear(NES_CTRL_SELECT);
          action_clear(NES_IO_POWER);
          action_clear(NES_IO_RESET);                                        
          break;  
        case ACTION_4: // ['right', 'B']
          action_clear(NES_CTRL_UP);
          action_clear(NES_CTRL_DOWN);
          action_clear(NES_CTRL_LEFT);
          action_output(NES_CTRL_RIGHT);
          action_clear(NES_CTRL_A);
          action_output(NES_CTRL_B);
          action_clear(NES_CTRL_START);
          action_clear(NES_CTRL_SELECT);
          action_clear(NES_IO_POWER);
          action_clear(NES_IO_RESET);                                        
          break;
        case ACTION_5: // ['right', 'A', 'B']
          action_clear(NES_CTRL_UP);
          action_clear(NES_CTRL_DOWN);
          action_clear(NES_CTRL_LEFT);
          action_output(NES_CTRL_RIGHT);
          action_output(NES_CTRL_A);
          action_output(NES_CTRL_B);
          action_clear(NES_CTRL_START);
          action_clear(NES_CTRL_SELECT);
          action_clear(NES_IO_POWER);
          action_clear(NES_IO_RESET);                                        
          break; 
        case ACTION_6: // ['A']
          action_clear(NES_CTRL_UP);
          action_clear(NES_CTRL_DOWN);
          action_clear(NES_CTRL_LEFT);
          action_clear(NES_CTRL_RIGHT);
          action_output(NES_CTRL_A);
          action_clear(NES_CTRL_B);
          action_clear(NES_CTRL_START);
          action_clear(NES_CTRL_SELECT);
          action_clear(NES_IO_POWER);
          action_clear(NES_IO_RESET);                                        
          break;
        case ACTION_7: // ['left']
          action_clear(NES_CTRL_UP);
          action_clear(NES_CTRL_DOWN);
          action_output(NES_CTRL_LEFT);
          action_clear(NES_CTRL_RIGHT);
          action_clear(NES_CTRL_A);
          action_clear(NES_CTRL_B);
          action_clear(NES_CTRL_START);
          action_clear(NES_CTRL_SELECT);
          action_clear(NES_IO_POWER);
          action_clear(NES_IO_RESET);                                        
          break;
        case ACTION_8: // ['left', 'A']
          action_clear(NES_CTRL_UP);
          action_clear(NES_CTRL_DOWN);
          action_output(NES_CTRL_LEFT);
          action_clear(NES_CTRL_RIGHT);
          action_output(NES_CTRL_A);
          action_clear(NES_CTRL_B);
          action_clear(NES_CTRL_START);
          action_clear(NES_CTRL_SELECT);
          action_clear(NES_IO_POWER);
          action_clear(NES_IO_RESET);                                        
          break;
        case ACTION_9: // ['left', 'B']
          action_clear(NES_CTRL_UP);
          action_clear(NES_CTRL_DOWN);
          action_output(NES_CTRL_LEFT);
          action_clear(NES_CTRL_RIGHT);
          action_clear(NES_CTRL_A);
          action_output(NES_CTRL_B);
          action_clear(NES_CTRL_START);
          action_clear(NES_CTRL_SELECT);
          action_clear(NES_IO_POWER);
          action_clear(NES_IO_RESET);                                        
          break;
        case ACTION_10: // ['left', 'A', 'B']
          action_clear(NES_CTRL_UP);
          action_clear(NES_CTRL_DOWN);
          action_output(NES_CTRL_LEFT);
          action_clear(NES_CTRL_RIGHT);
          action_output(NES_CTRL_A);
          action_output(NES_CTRL_B);
          action_clear(NES_CTRL_START);
          action_clear(NES_CTRL_SELECT);
          action_clear(NES_IO_POWER);
          action_clear(NES_IO_RESET);                                        
          break; 
        case ACTION_11: // ['down']
          action_clear(NES_CTRL_UP);
          action_output(NES_CTRL_DOWN);
          action_clear(NES_CTRL_LEFT);
          action_clear(NES_CTRL_RIGHT);
          action_clear(NES_CTRL_A);
          action_clear(NES_CTRL_B);
          action_clear(NES_CTRL_START);
          action_clear(NES_CTRL_SELECT);
          action_clear(NES_IO_POWER);
          action_clear(NES_IO_RESET);                                        
          break; 
        case ACTION_12: // ['up']
          action_output(NES_CTRL_UP);
          action_clear(NES_CTRL_DOWN);
          action_clear(NES_CTRL_LEFT);
          action_clear(NES_CTRL_RIGHT);
          action_clear(NES_CTRL_A);
          action_clear(NES_CTRL_B);
          action_clear(NES_CTRL_START);
          action_clear(NES_CTRL_SELECT);
          action_clear(NES_IO_POWER);
          action_clear(NES_IO_RESET);                                        
          break; 
        case ACTION_13: // ['start']
          action_clear(NES_CTRL_UP);
          action_clear(NES_CTRL_DOWN);
          action_clear(NES_CTRL_LEFT);
          action_clear(NES_CTRL_RIGHT);
          action_clear(NES_CTRL_A);
          action_clear(NES_CTRL_B);
          action_output(NES_CTRL_START);
          action_clear(NES_CTRL_SELECT);
          action_clear(NES_IO_POWER);
          action_clear(NES_IO_RESET);                                        
          break;
        case ACTION_14: // ['select']
          action_clear(NES_CTRL_UP);
          action_clear(NES_CTRL_DOWN);
          action_clear(NES_CTRL_LEFT);
          action_clear(NES_CTRL_RIGHT);
          action_clear(NES_CTRL_A);
          action_clear(NES_CTRL_B);
          action_clear(NES_CTRL_START);
          action_output(NES_CTRL_SELECT);
          action_clear(NES_IO_POWER);
          action_clear(NES_IO_RESET);                                        
          break;
        case ACTION_15: // ['power']
          action_clear(NES_CTRL_UP);
          action_clear(NES_CTRL_DOWN);
          action_clear(NES_CTRL_LEFT);
          action_clear(NES_CTRL_RIGHT);
          action_clear(NES_CTRL_A);
          action_clear(NES_CTRL_B);
          action_clear(NES_CTRL_START);
          action_clear(NES_CTRL_SELECT);
          action_output(NES_IO_POWER);
          action_clear(NES_IO_RESET);                                        
          break; 
        case ACTION_16: // ['reset']
          action_clear(NES_CTRL_UP);
          action_output(NES_CTRL_DOWN);
          action_clear(NES_CTRL_LEFT);
          action_clear(NES_CTRL_RIGHT);
          action_clear(NES_CTRL_A);
          action_clear(NES_CTRL_B);
          action_clear(NES_CTRL_START);
          action_output(NES_CTRL_SELECT);
          action_clear(NES_IO_POWER);
          action_output(NES_IO_RESET);                                        
          delay(2000);    
          action_clear(NES_CTRL_START);
          action_clear(NES_CTRL_SELECT);                                           
          break;                                                                                                                     
        default:
          action_clear(NES_CTRL_UP);
          action_clear(NES_CTRL_DOWN);
          action_clear(NES_CTRL_LEFT);
          action_clear(NES_CTRL_RIGHT);
          action_clear(NES_CTRL_A);
          action_clear(NES_CTRL_B);
          action_clear(NES_CTRL_START);
          action_clear(NES_CTRL_SELECT);
          action_clear(NES_IO_POWER);
          action_clear(NES_IO_RESET);                                        
          break;
      }
      // clear the string:
      inputString = "";
      stringComplete = false;
    }
  }
}

/* FUNCTIONS */

//
void action_output(int pin) {
  if(BUTTON_ENABLE) {
    digitalWrite(pin, HIGH);
  }
  else {
    digitalWrite(pin, LOW);   
  }
}

//
void action_clear(int pin) {
  if(BUTTON_ENABLE) {
    digitalWrite(pin, LOW);
  }
  else {
    digitalWrite(pin, HIGH);   
  }
}
