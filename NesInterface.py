
"""
@author: Conrad Salinas <csalinasonline@gmail.com>
"""

#  Notes:
#  NES Controller action mapping
#  COMPLEX
#  1 [['NOOP'],
#  2 ['right'],
#  3 ['right', 'A'],
#  4 ['right', 'B'],
#  5 ['right', 'A', 'B'],
#  6 ['A'],
#  7 ['left'],
#  8 ['left', 'A'],
#  9 ['left', 'B'],
#  10 ['left', 'A', 'B'],
#  11 ['down'],
#  12 ['up']]
#  NES Controller action mapping misc
#  13 [['start'],
#  14 ['select']]
#  NES Classic IO mapping
#  15 [['power'],
#  16 ['reset']
#  SIMPLE
#  1  ['NOOP'],
#  2  ['right'],
#  3  ['right', 'A'],
#  4  ['right', 'B'],
#  5  ['right', 'A', 'B'],
#  6  ['A'],
#  7  ['left'],


NES_NOOP = 1
NES_RIGHT = 2
NES_RIGHT_A = 3
NES_RIGHT_B = 4
NES_RIGHT_A_B = 5
NES_A = 6
NES_LEFT = 7
NES_LEFT_A = 8
NES_LEFT_B = 9
NES_LEFT_A_B = 10
NESP_DOWN = 11
NES_UP = 12
NES_START = 13
NES_SELECT = 14
NES_POWER = 15
NES_RESET = 16

# method that reset nes to main menu
def nes_button(ser, button):
   try:
      if(ser.isOpen()):
         msg = str(button) +'\n'
         msg = msg.encode('utf_8')
         ser.write(msg)
   except e:
      print(e)

