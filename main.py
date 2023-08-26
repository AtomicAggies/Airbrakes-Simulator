# main.py

import argparse
import csv
import yaml

import numpy as np

from math import pi
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

# Define Constants
GRAVITY = 9.80665
AIR_DENSITY = 1.225

# Define Arg Parser
parser = argparse.ArgumentParser(description='Airbrake simulator.')
parser.add_argument('-c', '--config', dest="config", help='Config file path', required=True)

def main():
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # Based on the config, calculate simulation
    simulation = simulate(config)
    # Then plot the results
    plot(simulation)

def simulate(config):
    # Read the thrust curve file
    with open(config["rocket"]["thrust_curve"], 'r') as f:
        thrust_curve = [{
            "time": float(i[0]),
            "thrust": float(i[1])} for i in list(csv.reader(f))[1:]]
    burn_time = thrust_curve[-1]["time"]
    # Create a matrix of time steps
    # All matrix calculations are done based on these time steps
    if "dt" in config["simulation"]:
        dt = config["simulation"]["dt"]
    else:
        dt = burn_time/5000
    time_steps = np.arange(0, burn_time, dt)
    # Calculate drag conditions
    # Area is given in mm^2, so convert to m^2
    fin_drag_force = config["airbrakes"]["drag_coefficient"] * AIR_DENSITY * config["airbrakes"]["fins"] * config["airbrakes"]["area"] * (10**-6) * 0.5
    # Diameter is also given in mm, so convert to m
    rocket_area = (config["rocket"]["diameter"] * (10**-3))**2 * pi * 0.25
    rocket_drag_force = config["rocket"]["drag_coefficient"] * AIR_DENSITY * rocket_area * 0.5
    # Calculate mass over time
    rocket_full_mass = config["rocket"]["mass_empty"] + config["rocket"]["engine_mass"]
    rocket_mass = np.interp(time_steps, [0, burn_time], [rocket_full_mass, rocket_full_mass - config["rocket"]["propellant_mass"]])
    # TODO: interpolate mass based on propellant burn rate
    #       this should be directly propotional to force of thrust
    # Calculate forces acting on the rocket
    weight = rocket_mass * GRAVITY
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
        drag_force = config["rocket"]["drag_coefficient"] * AIR_DENSITY * rocket_area * 0.5 * velocity[time_index-1]**2
        # Calculate acceleration and thus velocity and position based on forces
        acceleration[time_index] = (thrust(time) - weight[time_index] - drag_force) / rocket_mass[time_index]
        velocity[time_index] = velocity[time_index-1] + acceleration[time_index] * dt
        position[time_index] = position[time_index-1] + velocity[time_index] * dt
    # Continue the simulation after burn out, until the rocket hits the ground
    print("Coasting rocket...")
    while position[-1] > 0:
        time_index += 1
        if position[-1] > config["simulation"]["deployment_altitude"]:
            drag_force = config["airbrakes"]["drag_coefficient"] * AIR_DENSITY * rocket_area * 0.5 * velocity[time_index-1]**2
        else:
            drag_force = config["rocket"]["drag_coefficient"] * AIR_DENSITY * rocket_area * 0.5 * velocity[time_index-1]**2
        acceleration = np.append(acceleration, (-weight[-1] - drag_force) / rocket_mass[-1])
        velocity = np.append(velocity, velocity[time_index-1] + acceleration[time_index] * dt)
        position = np.append(position, position[time_index-1] + velocity[time_index] * dt)
    # increase the time array too
    time_steps = np.arange(0, len(acceleration)*dt, dt)
    print("LANDED!")
    return {
        "time": time_steps,
        "acceleration": acceleration,
        "velocity": velocity,
        "position": position
    }

def plot(simulation):
    fig, ax = plt.subplots()
    # Create seperate axis for the velocity and position
    velocity_ax = ax.twinx()
    position_ax = ax.twinx()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (m/s^2)")
    velocity_ax.set_ylabel("Velocity (m/s) | Position (m)")
    # position_ax.set_ylabel("Position (m)")

    ax.plot(simulation["time"], simulation["acceleration"], label="Acceleration")
    velocity_ax.plot(simulation["time"], simulation["velocity"], label="Velocity", color="orange")
    position_ax.plot(simulation["time"], simulation["position"], label="Position", color="green")

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
