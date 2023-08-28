# simulator.py
# Application to simulate the rocket flight.

import csv
import numpy as np

from math import pi
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

# Local imports
from sensor import Sensor

class Simulator:
    # Define Constants
    GRAVITY = 9.80665
    AIR_DENSITY = 1.225

    def __init__(self, config):
        self.config = config
        self.rocket = Rocket(config["rocket"])
        self.airbrakes = Airbrakes(config["airbrakes"])
    
    def simulate(self):
        # Read the thrust curve file
        with open(self.config["rocket"]["thrust_curve"], 'r') as f:
            thrust_curve = [{
                "time": float(i[0]),
                "thrust": float(i[1])} for i in list(csv.reader(f))[1:]]
        burn_time = thrust_curve[-1]["time"]
        # Create a matrix of time steps
        # All matrix calculations are done based on these time steps
        if "dt" in self.config["simulation"]:
            dt = self.config["simulation"]["dt"]
        else:
            dt = burn_time/5000
        time_steps = np.arange(0, burn_time, dt)
        # Calculate drag conditions
        # Area is given in mm^2, so convert to m^2
        fin_drag_force = self.config["airbrakes"]["drag_coefficient"] * self.AIR_DENSITY * self.config["airbrakes"]["fins"] * self.config["airbrakes"]["area"] * (10**-6) * 0.5
        # Diameter is also given in mm, so convert to m
        rocket_area = (self.config["rocket"]["diameter"] * (10**-3))**2 * pi * 0.25
        rocket_drag_force = self.config["rocket"]["drag_coefficient"] * self.AIR_DENSITY * rocket_area * 0.5
        airbrakes_area = self.config["airbrakes"]["area"] * (10**-6)
        # Calculate mass over time
        rocket_full_mass = self.config["rocket"]["mass_empty"] + self.config["rocket"]["engine_mass"]
        rocket_mass = np.interp(time_steps, [0, burn_time], [rocket_full_mass, rocket_full_mass - self.config["rocket"]["propellant_mass"]])
        # TODO: interpolate mass based on propellant burn rate
        #       this should be directly propotional to force of thrust
        # Calculate forces acting on the rocket
        weight = rocket_mass * self.GRAVITY
        x, y = zip(*[(i["time"], i["thrust"]) for i in thrust_curve])
        thrust = CubicSpline(x, y, bc_type="natural")
        # Setup zero matrices for acceleration, velocity, and position
        acceleration = np.zeros(len(time_steps))
        velocity = np.zeros(len(time_steps))
        position = np.zeros(len(time_steps))
        # Loop over time steps while the engine is burning
        print("Burning engine...")
        for time_index in range(1, len(time_steps)):
            time = time_steps[time_index]
            drag_force = self.config["rocket"]["drag_coefficient"] * self.AIR_DENSITY * rocket_area * 0.5 * velocity[time_index-1]**2
            # Calculate acceleration and thus velocity and position based on forces
            acceleration[time_index] = (thrust(time) - weight[time_index] - drag_force) / rocket_mass[time_index]
            velocity[time_index] = velocity[time_index-1] + acceleration[time_index] * dt
            position[time_index] = position[time_index-1] + velocity[time_index] * dt
        # Continue the simulation after burn out, until the rocket hits the ground
        print("Coasting rocket...")
        while position[-1] > 0:
            time_index += 1
            drag_force = self.config["rocket"]["drag_coefficient"] * self.AIR_DENSITY * rocket_area * 0.5 * velocity[time_index-1]**2
            if position[-1] > self.config["simulation"]["deployment_altitude"]:
                drag_force += self.config["airbrakes"]["drag_coefficient"] * self.AIR_DENSITY * airbrakes_area * 0.5 * velocity[time_index-1]**2
            acceleration = np.append(acceleration, (-weight[-1] - drag_force) / rocket_mass[-1])
            velocity = np.append(velocity, velocity[time_index-1] + acceleration[time_index] * dt)
            position = np.append(position, position[time_index-1] + velocity[time_index] * dt)
        # expand the time array too
        time_steps = np.arange(0, len(acceleration)*dt, dt)
        print("LANDED!")
        self.simulation_results = {
            "time": time_steps,
            "acceleration": acceleration,
            "velocity": velocity,
            "position": position,
            "mass": np.pad(rocket_mass, (0, len(acceleration)-len(rocket_mass)), "edge")
        }
        return self.simulation_results

    def plot(self, simulation=None):
        # Normalize all vectors to plot on a 2D graph
        acceleration = [np.linalg.norm(a) for a in self.simulation_results["acceleration"]]
        velocity = [np.linalg.norm(v) for v in self.simulation_results["velocity"]]
        position = [np.linalg.norm(p) for p in self.simulation_results["position"]]
        if simulation is None:
            simulation = self.simulation_results
        fig, ax = plt.subplots()
        # Create seperate axis for the velocity and position
        velocity_ax = ax.twinx()
        position_ax = ax.twinx()
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Acceleration (m/s^2)")
        velocity_ax.set_ylabel("Velocity (m/s)")
        position_ax.set_ylabel("Position (m)")

        ax.plot(simulation["time"], acceleration, label="Acceleration")
        velocity_ax.plot(simulation["time"], velocity, label="Velocity", color="orange")
        position_ax.plot(simulation["time"], position, label="Position", color="green")
        position_ax.spines["right"].set_position(("axes", 1.2))
        ax.plot(simulation["time"], simulation["mass"], label="Mass", color="red")

        fig.tight_layout()
        plt.show()

class Rocket:
    def __init__(self, config):
        # Create sensors for the emulator
        self.accelerometer = Sensor(axes=3, noise=config["accelerometer_noise"], axes_names=["x", "y", "z"])
        self.barometer = Sensor(noise=config["barometer_noise"])
        # Create thrust curve
        with open(config["thrust_curve"], 'r') as f:
            thrust_curve = [{
                "time": float(i[0]),
                "thrust": float(i[1])} for i in list(csv.reader(f))[1:]]
            self.thrust = CubicSpline([i["time"] for i in thrust_curve], [i["thrust"] for i in thrust_curve], bc_type="natural")
        self.burn_time = thrust_curve[-1]["time"]
        # Set basic parameters
        self.config = config
        self.mass = config["mass_empty"] + config["engine_mass"]
        self.direction = np.array([0, 1, 0], dtype=np.float64)
        self.previous_acceleration, self.previous_velocity, self.previous_position = np.array([0, 0, 0], dtype=np.float64), np.array([0, 0, 0], dtype=np.float64), np.array([0, 0, 0], dtype=np.float64)
        self.acceleration, self.velocity, self.position = np.array([0, 0, 0], dtype=np.float64), np.array([0, 0, 0], dtype=np.float64), np.array([0, 0, 0], dtype=np.float64)
        self.previous_time = 0
        # Calculated properties
        # Diameter is given in mm, so convert to m
        self.area = (self.config["diameter"] * (10**-3))**2 * pi * 0.25
        # Calculate mass over time
        rocket_full_mass = self.config["mass_empty"] + self.config["engine_mass"]
        self.mass = CubicSpline([0, self.burn_time], [rocket_full_mass, rocket_full_mass - self.config["propellant_mass"]])
        # TODO: interpolate mass based on propellant burn rate
        #       this should be directly propotional to force of thrust

    def update(self, dt=0.1, forces=[]):
        time = self.previous_time + dt
        # Calculate the new acceleration based on the forces
        # TODO: ensure vectors are calculated properly
        if time < self.burn_time:
            forces.append(self.thrust(time))
        # Forces will act in the direction of the rocket
        forces = sum(forces) * self.direction
        # Sum these forces to get acceleration
        self.acceleration = forces / self.mass(time)
        # Calculate the new velocity and position
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt
        # Push the new acceleration to the accelerometer
        self.accelerometer.push(self.acceleration)
        # Push the new position to the barometer
        # TODO: fix me
        self.barometer.push(self.position)
        # Update the previous values
        self.previous_acceleration = self.acceleration
        self.previous_velocity = self.velocity
        self.previous_position = self.position
        self.previous_time = time

class Airbrakes:
    def __init__(self, config):
        self.config = config

__all__ = ["Simulator"]
