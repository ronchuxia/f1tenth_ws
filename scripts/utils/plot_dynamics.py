"""
Plot the maximum speed of the car as a function of the steering angle, based on:
1. Rollover limit.
2. Tire friction limit.
"""


import numpy as np
import matplotlib.pyplot as plt


# parameters
g = 9.81
b = 0.295   # track width: the distance between the wheel contact patches
h = 0.2 # center of mass height
l = 0.32    # wheel base
mu = 0.7

approximate = False # approximate tanx=x

steering_angles = np.linspace(0.0, 1.57, 100)

r = l / np.tan(np.fabs(steering_angles)) if not approximate else l / np.fabs(steering_angles)
centrifugal = 0.5 * b * g * r / h
friction = mu * g * r

c_max_speed = np.sqrt(centrifugal)
f_max_speed = np.sqrt(friction)

plt.plot(steering_angles, c_max_speed, label='centrifugal')
plt.plot(steering_angles, f_max_speed, label='friction')
plt.xlabel('steering angle (rad)')
plt.ylabel('max speed (m/s)')
plt.title('max speed vs steering angle')
plt.legend()
plt.grid()
plt.show()
