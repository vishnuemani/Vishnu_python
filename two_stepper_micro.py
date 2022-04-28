from time import sleep
import RPi.GPIO as GPIO

DIR = 19   # Direction GPIO Pin
STEP = 26  # Step GPIO Pin
CW = 1     # Clockwise Rotation
CCW = 0    # Counterclockwise Rotation
SPR = 48   # Steps per Revolution (360 / 7.5)

DIR2 = 20   # Direction GPIO Pin TOP
STEP2 = 21  # Step GPIO Pin TOP

GPIO.setmode(GPIO.BCM)
GPIO.setup(DIR, GPIO.OUT)
GPIO.setup(STEP, GPIO.OUT)
GPIO.output(DIR, CW)

GPIO.setup(DIR2, GPIO.OUT)
GPIO.setup(STEP2, GPIO.OUT)
GPIO.output(DIR2, CW)

MODE = (14, 15, 18)   # Microstep Resolution GPIO Pins
MODE2 = (2,3,4)
GPIO.setup(MODE, GPIO.OUT)
GPIO.setup(MODE2, GPIO.OUT)

RESOLUTION = {'Full': (0, 0, 0),
              'Half': (1, 0, 0),
              '1/4': (0, 1, 0),
              '1/8': (1, 1, 0),
              '1/16': (0, 0, 1),
              '1/32': (1, 0, 1)}
GPIO.output(MODE, RESOLUTION['1/32'])
GPIO.output(MODE2, RESOLUTION['1/32'])

step_count = SPR * 32
delay = .0416 / 32

for x in range(step_count):
    GPIO.output(STEP, GPIO.HIGH)
    GPIO.output(STEP2, GPIO.HIGH)
    sleep(delay)
    GPIO.output(STEP, GPIO.LOW)
    GPIO.output(STEP2, GPIO.LOW)
    sleep(delay)

sleep(.5)


GPIO.output(DIR, CCW)
GPIO.output(DIR2, CCW)
for x in range(step_count):
    GPIO.output(STEP, GPIO.HIGH)
    GPIO.output(STEP2, GPIO.HIGH)
    sleep(delay)
    GPIO.output(STEP, GPIO.LOW)
    GPIO.output(STEP2, GPIO.LOW)
    sleep(delay)

GPIO.cleanup()
