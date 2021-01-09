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
#define NES_CTRL_UP       13
#define NES_CTRL_DOWN     12
#define NES_CTRL_LEFT     11
#define NES_CTRL_RIGHT    10
#define NES_CTRL_A        9
#define NES_CTRL_B        8
#define NES_CTRL_START    7
#define NES_CTRL_SELECT   6
#define NES_IO_POWER      5
#define NES_IO_RESET      4

/* GLOBALS */
String inputString = "";         // a String to hold incoming data
bool stringComplete = false;  // whether the string is complete

/* PROTOTYPES */
void serialEvent(void);

/* INIT */
void setup() {
  // initialize serial:
  Serial.begin(9600);
  // reserve 200 bytes for the inputString:
  inputString.reserve(200);
}

/* LOOP */
void loop() {
  // print the string when a newline arrives:
  if (stringComplete) {
    Serial.println(inputString);
    // clear the string:
    inputString = "";
    stringComplete = false;
  }
}

/* FUNCTIONS */
/*
  SerialEvent occurs whenever a new data comes in the hardware serial RX. This
  routine is run between each time loop() runs, so using delay inside loop can
  delay response. Multiple bytes of data may be available.
*/
void serialEvent() {
  while (Serial.available()) {
    // get the new byte:
    char inChar = (char)Serial.read();
    // add it to the inputString:
    inputString += inChar;
    // if the incoming character is a newline, set a flag so the main loop can
    // do something about it:
    if (inChar == '\n') {
      stringComplete = true;
    }
  }
}
