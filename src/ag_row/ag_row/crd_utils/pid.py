from math import atan2, sqrt

class PIDController:
    def __init__(self, kp, ki, kd, target):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.error_sum = 0
        self.last_error = 0

    def update(self, current):
        error = self.target - current
        self.error_sum += error
        derivative = error - self.last_error
        self.last_error = error
        output = self.kp * error + self.ki * self.error_sum + self.kd * derivative
        return output


'''
Here are the steps to tune the PID controller using MATLAB:

1.Define the plant model: First, you need to create a mathematical model of the robot's dynamics. This model should include the relationship between the input (angular velocity command) and output (heading error). You can use a simple model, such as a first-order model, or a more complex model, depending on the robot's dynamics.

2.Design the PID controller: Next, you can use the Control System Toolbox in MATLAB to design the PID controller. The pidtune function in the toolbox provides a convenient way to tune the controller based on the desired closed-loop performance specifications, such as the rise time, settling time, and overshoot.

3.Simulate the closed-loop system: Once you have designed the PID controller, you can simulate the closed-loop system using MATLAB. You can use the sim function to simulate the plant and controller and observe the system's response to different inputs.

4.Evaluate the performance: After simulating the system, you can evaluate the controller's performance by comparing the actual response to the desired response. You can use various performance metrics, such as the rise time, settling time, overshoot, and steady-state error, to assess the controller's performance.

5.Fine-tune the controller: If the controller's performance is not satisfactory, you can fine-tune the gains manually or use the pidtune function to refine the gains further. You can also adjust other parameters in the system, such as the reference input, disturbance input, and noise, to achieve the desired performance.

MATLAB provides a wealth of tools and functions for analyzing and designing control systems, so it's an excellent choice for tuning a PID controller. However, it's important to keep in mind that tuning a controller requires a good understanding of the plant's dynamics and the desired closed-loop performance specifications. It may also require multiple iterations of design, simulation, and testing to achieve the desired results.


Set all gains to zero.
Increase the proportional gain until the system starts to oscillate.
Reduce the proportional gain slightly until the oscillation stops.
Increase the integral gain until the steady-state error is minimized.
Add a derivative gain to improve the response time and reduce overshoot, if needed.
Repeat steps 2-5 until the desired performance is achieved.

'''