# simulator.py
# Application to simulate the rocket flight.

import csv
import numpy as np
import pandas as pd

from datetime import datetime
from math import pi
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline

# Local imports
from rocket import Airbrakes, Rocket

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
        thrust = UnivariateSpline(x, y, k=1, ext="const")
        # Setup zero matrices for acceleration, velocity, and position
        self.dataframe = pd.DataFrame(index=pd.TimedeltaIndex(time_steps, freq="s", name="time"), columns=["positionx", "positiony", "positionz", "velocityx", "velocityy", "velocityz", "accelerationx", "accelerationy", "accelerationz", "mass"])
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
    
    def save(self):
        self.dataframe.to_csv(f"data-{datetime.now().isoformat(timespec='seconds').replace(':', '')}.csv")

__all__ = ["Simulator"]
