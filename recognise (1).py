import cv2
import numpy as np
import pyttsx3

def nothing(x):
    pass


image_x, image_y = 64, 64

from keras.models import load_model
from tkinter import *
root = Tk()
root.geometry('25x25')

classifier = load_model('model.h5')


def predictor():
    import numpy as np
    import keras.utils as image
    test_image = image.load_img('1.png', target_size=(240, 195))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)

    if result[0][0] == 1:
        return 'call_me'
    elif result[0][1] == 1:
        return 'fingers_crossed'
    elif result[0][2] == 1:
        return 'okay'
    elif result[0][3] == 1:
        return 'paper'
    elif result[0][4] == 1:
        return 'peace'
    elif result[0][5] == 1:
        return 'rock'
    elif result[0][6] == 1:
        return 'rock_on'
    elif result[0][7] == 1:
        return 'scissor'
    elif result[0][8] == 1:
        return 'thumbs'
    elif result[0][9] == 1:
        return 'up'
    
cam = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

cv2.namedWindow("test")

img_counter = 0

img_text = ''
last_text = 'call me'

while True:
   
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8, shift=0)

    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    imcrop = img[102:298, 427:623]
    hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    cv2.putText(frame, img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
    cv2.imshow("test", frame)
    cv2.imshow("mask", mask)

    img_name = "1.png"
    save_img = cv2.resize(mask, (image_x, image_y))
    cv2.imwrite(img_name, save_img)
    print("{} written!".format(img_name))
    img_text = predictor()
    
    def buttonFunction():
        
        engine = pyttsx3.init()
        last_text = img_text
        engine.say(img_text)
        engine.runAndWait()
        
    b= Button(root, text = "SPEECH", command=buttonFunction)
    root.update()
    b.pack()
    
    
    if cv2.waitKey(1) == 27:
        root.quit()
        break



cam.release()
cv2.destroyAllWindows()