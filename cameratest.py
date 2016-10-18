from picamera import PiCamera
from time import sleep

camera = PiCamera()

camera.start_preview()
sleep(2)
camera.capture('/home/pi/lab4/test.jpg')
sleep(2)
#camera.start_recording('/home/pi/lab4/testvideo.h264')
#sleep(5)
#camera.stop_recording()
camera.stop_preview()
