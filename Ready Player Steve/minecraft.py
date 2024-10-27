import pyautogui
import time
from pynput import mouse

# Simulate key presses
def move_forward():
    pyautogui.keyDown('w')
    time.sleep(0.3)
    pyautogui.keyUp('w')


def stop_moving_forward():
    pyautogui.keyUp('w')

def move_backward():
    pyautogui.keyDown('s')

def stop_moving_backward():
    pyautogui.keyUp('s')

def jump():
    pyautogui.press('space')

def crouch():
    pyautogui.keyDown('shift')
    time.sleep(0.3)
    pyautogui.keyUp('shift')


def stop_crouching():
    pyautogui.keyUp('shift')

def attack():
    pyautogui.click(button='left')

def place_block():
    pyautogui.click(button='right')

def go_inventory():
    pyautogui.keyDown('E')

def leave_inventory():
    pyautogui.keyDown('esc')
