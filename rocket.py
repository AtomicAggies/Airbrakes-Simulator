# rocket.py
# Rocket data class to contain all rocket data and calculations.

import csv
import numpy as np

from math import pi
from scipy.interpolate import CubicSpline

# Local imports
from sensor import Sensor

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

__all__ = ["Airbrakes", "Rocket"]
