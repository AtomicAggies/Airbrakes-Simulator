# projectile.py
# Following this guide: https://physics.weber.edu/schroeder/scicomp/PythonManual.pdf
# Projectile motion including advanced variables such as drag and Magnus force

# Imports
import numpy as np
import pandas as pd
import threading
import matplotlib.pyplot as plt

from datetime import datetime
from math import cos, sin, radians

# Constants
g = 9.81 # m/s^2
dt = 0.001 # s
cd = 0.24 # Drag coefficient
launch_angle = 10 # degrees

class Projectile:
    def __init__(self, mass, launch_angle=0, initial_velocity=10):
        # Launch angle is measured in degrees from the vertical
        self.position = np.array([0, 0, 0])
        self.velocity = initial_velocity * np.array([sin(radians(launch_angle)), cos(radians(launch_angle)), 0])
        self.acceleration = np.array([0, 0, 0])
        self.mass = mass

    def update(self, dt):
        # Calculate the new acceleration
        drag = -cd * self.velocity * np.absolute(self.velocity) / self.mass
        self.acceleration = np.add(np.array([0, -g, 0]), drag)
        # Calculate the new velocity and position
        self.position = np.add(self.position, self.velocity * dt)
        self.velocity = np.add(self.velocity, self.acceleration * dt)
    
    def __str__(self):
        return f"<Projectile: position={self.position}, velocity={self.velocity}, acceleration={self.acceleration}, mass={self.mass}>"

def main():
    # Initialize the projectile
    p = Projectile(mass=1, launch_angle=launch_angle)
    p.update(dt)
    t = dt
    # Create the dataframe
    time = pd.TimedeltaIndex([0], freq=f"s", name="time")
    columns = ["positionx", "positiony", "positionz", "velocityx", "velocityy", "velocityz", "accelerationx", "accelerationy", "accelerationz"]
    df = pd.DataFrame(index=time, columns=columns)
    df = df.iloc[1:, :]
    while p.position[1] > 0:
        p.update(dt)
        # Append data to the dataframe
        df.at[t, "positionx"] = p.position[0]
        df.at[t, "positiony"] = p.position[1]
        df.at[t, "positionz"] = p.position[2]
        df.at[t, "velocityx"] = p.velocity[0]
        df.at[t, "velocityy"] = p.velocity[1]
        df.at[t, "velocityz"] = p.velocity[2]
        df.at[t, "accelerationx"] = p.acceleration[0]
        df.at[t, "accelerationy"] = p.acceleration[1]
        df.at[t, "accelerationz"] = p.acceleration[2]
        t += dt
    print(f"Projectile lands at {t} seconds with a velocity of {p.velocity[1]} m/s")
    # Plot the results
    time = np.arange(0, t, dt)
    ax = plt.figure().add_subplot(projection="3d")
    # Plot the data from the dataframe
    ax.plot(df.positionx, df.positionz, df.positiony)

    # Plot the velocity and acceleration
    ax2 = plt.figure().add_subplot()
    args = {}
    n = int(len(df)/30)
    for i in ["x", "y"]:
        args[i] = getattr(df, "position"+i).to_numpy()[::n].astype(float)
        args["d"+i] = getattr(df, "velocity"+i).to_numpy()[::n].astype(float)
        args["d2"+i] = getattr(df, "acceleration"+i).to_numpy()[::n].astype(float)
    ax2.quiver(args["x"], args["y"], args["dx"], args["dy"], color="blue")
    ax2.quiver(args["x"], args["y"], args["d2x"], args["d2y"], color="red")

    # Start saving the data since it sometimes takes awhile
    save_thread = threading.Thread(target=save, args=(df,))
    save_thread.start()

    plt.show()

    if save_thread.is_alive():
        print("Still saving data...")
        save_thread.join()

def save(df):
    df.to_csv(f"data-{datetime.now().isoformat(timespec='seconds').replace(':', '')}.csv")

if __name__ == "__main__":
    main()
