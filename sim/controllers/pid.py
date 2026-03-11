import mujoco
import numpy as np

class PID:
    """
    Basic PID which can be used to act as a PID for 
    position, angle, etc
    """
    def __init__(self, 
                 kp: float, 
                 ki: float, 
                 kd: float, 
                 dt: float, 
                 output_limits=None,
                 integral_limits=None,
                 is_angle:bool = False):
        self.kp=kp
        self.ki=ki
        self.kd=kd
        self.dt=dt

        self.integral = 0.0
        self.prev_error = 0.0
        
        # output limits and integral limits should be (low,high) tuples
        self.output_limits = output_limits
        self.integral_limits = integral_limits
        self.is_angle = is_angle


    def update (self, 
                measurement: float, 
                target: float,
                derivative_override:float = None):
        
        error = target - measurement
        if self.is_angle:
            # if error is an angle, wrap it to be in the range [-pi,pi]
            error = (error+np.pi) % (2*np.pi) - np.pi
        # integral
        self.integral += error * self.dt

        if self.integral_limits is not None:
            low, high = self.integral_limits
            self.integral = np.clip(self.integral, low, high)

        # derivative
        if derivative_override is None:
            derivative = (error - self.prev_error) / self.dt
        else:
            derivative = -derivative_override

        output = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        self.prev_error = error

        if self.output_limits is not None:
            low, high = self.output_limits
            output = np.clip(output, low, high)

        return output

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0