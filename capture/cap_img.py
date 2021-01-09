import cv2

CONST_CAP_RES_WIDTH = 640
CONST_CAP_RES_HEIGHT = 480
CONST_NES_RES_WIDTH = 256
CONST_NES_RES_HEIGHT = 240
CONST_OFFSET_RES_WIDTH = (20, 476)
CONST_OFFSET_RES_HEIGHT = (128, 512)
CONST_FEATURE_RES_WIDTH = 84
CONST_FEATURE_RES_HEIGHT = 84

print(f'Capture Resolution: {(CONST_CAP_RES_WIDTH, CONST_CAP_RES_HEIGHT)}')
print(f'NES Resolution: {(CONST_NES_RES_WIDTH, CONST_NES_RES_HEIGHT)}')
print(f'Offsets 1: {(CONST_OFFSET_RES_WIDTH, CONST_OFFSET_RES_HEIGHT)}')

# Open the device at the ID 2
cap = cv2.VideoCapture(2)

#Check whether user selected camera is opened successfully.
if not cap.isOpened():
   print("Could not open video device")

#To set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONST_CAP_RES_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONST_CAP_RES_HEIGHT)

while(True):
   # Capture frame-by-frame
   ret, frame = cap.read()
   #
   frame_2 = frame[20:476, 128:512]
   #
   frame_3 = cv2.resize(frame_2, (CONST_NES_RES_WIDTH, CONST_NES_RES_HEIGHT))
   #
   frame_4 = cv2.cvtColor(frame_3, cv2.COLOR_RGB2GRAY)
   #
   frame_5 = cv2.resize(frame_4, (CONST_FEATURE_RES_WIDTH, CONST_FEATURE_RES_HEIGHT))
   # Display the resulting frame
   cv2.imshow('preview', frame_5)
   #Waits for a user input to quit the application
   if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
