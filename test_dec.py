import cv2

img = cv2.imread('reward_plot.png')
print(img.shape)
cv2.startWindowThread()
cv2.imshow('img', img)

# wait forever, if Q is pressed then close cv image window
if cv2.waitKey(0) & 0xFF == ord('q'):
   cv2.destroyAllWindows()
   