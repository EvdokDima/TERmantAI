from pynput.mouse import Button, Controller
from time import sleep

mouse = Controller()

#Bounce the cube by clicking on the screen
def bounce():
    mouse.position = (180, 350)
    mouse.press(Button.left)
    sleep(0.3)
    mouse.release(Button.left)