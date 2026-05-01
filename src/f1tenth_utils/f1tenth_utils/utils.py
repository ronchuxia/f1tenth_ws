import math

def max_speed_from_steering_angle(steering_angle, g=9.81, b=0.295, h=0.2, l=0.32, mu=0.7):
    """
    Computes the maximum speed at a steering angle that prevents the car from:
    1. Tipping over.
    2. Skidding.

    g = 9.81
    b = 0.295   # track width: the distance between the wheel contact patches
    h = 0.2 # center of mass height
    l = 0.32    # wheel base
    mu = 0.7
    
    NOTE: math.tan typically takes 50-150 ns. No need to approximate.
    """

    r = l / math.tan(math.fabs(steering_angle))
    centrifugal = 0.5 * b * g * r / h
    friction = mu * g * r

    max_speed = math.sqrt(min(centrifugal, friction))
    return max_speed


def dead_reckon(x, y, yaw, speed, steering_angle, dt, wheelbase=0.3302):
    """
    Forward-predict pose using the bicycle model over time dt.

    Call this after receiving a PF pose to compensate for PF processing latency:
        x, y, yaw = dead_reckon(x, y, yaw, last_speed, last_steering_angle, dt)

    x, y           : position (m)
    yaw            : heading (rad)
    speed          : longitudinal speed (m/s)
    steering_angle : front wheel steering angle (rad)
    dt             : time to predict forward (s)
    wheelbase      : distance between axles (m)
    """
    yaw_rate = (speed / wheelbase) * math.tan(steering_angle)
    if math.fabs(yaw_rate) < 1e-6:
        # straight line: Euler integration is exact
        x += speed * math.cos(yaw) * dt
        y += speed * math.sin(yaw) * dt
    else:
        # arc integration: exact solution for constant speed and steering angle
        x += (speed / yaw_rate) * (math.sin(yaw + yaw_rate * dt) - math.sin(yaw))
        y += (speed / yaw_rate) * (-math.cos(yaw + yaw_rate * dt) + math.cos(yaw))
    yaw += yaw_rate * dt
    return x, y, yaw