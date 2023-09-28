# projectile.py
# Following this guide: https://physics.weber.edu/schroeder/scicomp/PythonManual.pdf
# Projectile motion including advanced variables such as drag and Magnus force

# Imports
import numpy as np
import pandas as pd
import threading
import traceback
import matplotlib.pyplot as plt

from datetime import datetime
from math import cos, sin, radians
from scipy.interpolate import UnivariateSpline

# Constants
g = 9.81 # m/s^2
rho = 1.091 # kg/m^3
area = 0.01824 # m^2
dt = 0.01 # s
# Parameters of the rocket
cd = 0.445 # Drag coefficient
launch_angle = 10 # degrees
thrust_file = "Cesaroni_12066N2200-P.csv"
mass = 33.69 # kg

class Projectile:
    drag_coefficient = rho * area * cd / 2

    def __init__(self, mass, launch_angle=0, data_lock=None):
        # Read the thrust curve file
        thrust_data = pd.read_csv(thrust_file)
        self.burn_time = thrust_data["Time (s)"].iloc[-1]
        self.thrust = UnivariateSpline(thrust_data["Time (s)"], thrust_data["Thrust (N)"], k=1, ext="const")
        # Launch angle is measured in degrees from the vertical
        self.position = np.array([0, 0, 0])
        self.velocity = np.array([0, 0, 0])
        self.acceleration = np.array([0, 0, 0])
        self.mass = mass
        self.launch_angle = launch_angle
        self.df = pd.DataFrame(index=pd.TimedeltaIndex([0], freq="s", name="time"), columns=["positionx", "positiony", "positionz", "velocityx", "velocityy", "velocityz", "accelerationx", "accelerationy", "accelerationz"])
        self.df = self.df.iloc[1:, :]
        self.data_lock = data_lock

    @property
    def direction(self):
        if self.position[1] < 10:
            return np.array([sin(radians(self.launch_angle)), cos(radians(self.launch_angle)), 0])
        return self.velocity / np.linalg.norm(self.velocity)

    def burn(self, time, dt):
        self.data_lock.acquire()
        try:
            # Calculate the new acceleration
            drag = (-self.drag_coefficient * self.velocity * np.absolute(self.velocity)) / self.mass
            # Include the thrust in the direction the projectile is moving
            thrust = self.thrust(time) * self.direction
            self.acceleration = np.array([0, -g, 0]) + drag + thrust
            if time < dt*4:
                print(drag)
                print(self.direction)
                print(self.thrust(time) * self.direction)
            # Calculate the new velocity and position
            self.position = np.add(self.position, self.velocity * dt)
            self.velocity = np.add(self.velocity, self.acceleration * dt)
            # Append data to the dataframe
            self.df.loc[time] = self.position.tolist() + self.velocity.tolist() + self.acceleration.tolist()
        finally:
            self.data_lock.release()

    def update(self, time, dt):
        self.data_lock.acquire()
        try:
            # Calculate the new acceleration
            drag = (-self.drag_coefficient * self.velocity * np.absolute(self.velocity)) / self.mass
            self.acceleration = np.add(np.array([0, -g, 0]), drag)
            # Calculate the new velocity and position
            self.position = np.add(self.position, self.velocity * dt)
            self.velocity = np.add(self.velocity, self.acceleration * dt)
            # Append data to the dataframe
            self.df.loc[time] = self.position.tolist() + self.velocity.tolist() + self.acceleration.tolist()
        finally:
            self.data_lock.release()
    
    def __str__(self):
        return f"<Projectile: position={self.position}, velocity={self.velocity}, acceleration={self.acceleration}, mass={self.mass}>"

def update_plots(lock, ax, ax2, ax3, df):
    global running
    try:
        lock.acquire()
        ax.clear()
        ax.plot(df.positionx, df.positionz, df.positiony)

        ax2.clear()
        args = {}
        n = int(len(df)/30)+1
        for i in ["x", "y"]:
            args[i] = getattr(df, "position"+i).to_numpy()[::n].astype(float)
            args["d"+i] = getattr(df, "velocity"+i).to_numpy()[::n].astype(float)
            args["d2"+i] = getattr(df, "acceleration"+i).to_numpy()[::n].astype(float)
        
        ax2.quiver(args["x"], args["y"], args["dx"], args["dy"], color="blue")
        ax2.quiver(args["x"], args["y"], args["d2x"], args["d2y"], color="red")

        ax3.clear()
        ax3.plot(df.index, df.velocityy)

        plt.draw()
        print(threading.Event().wait(0.5))
    finally:
        lock.release()
        if running:
            return threading.Timer(0.5, update_plots, args=(ax, ax2, ax3, df,)).start()

def main():
    global running
    # Initialize the projectile
    data_lock = threading.Lock()
    p = Projectile(mass=mass, launch_angle=launch_angle, data_lock=data_lock)
    # Plot the initial position and start a thread to update it
    ax = plt.figure().add_subplot(projection="3d")
    # Plot the data from the dataframe
    ax.plot(p.df.positionx, p.df.positionz, p.df.positiony)

    # Plot the velocity and acceleration
    ax2 = plt.figure().add_subplot()
    args = {}
    n = int(len(p.df)/30)+1
    for i in ["x", "y"]:
        args[i] = getattr(p.df, "position"+i).to_numpy()[::n].astype(float)
        args["d"+i] = getattr(p.df, "velocity"+i).to_numpy()[::n].astype(float)
        args["d2"+i] = getattr(p.df, "acceleration"+i).to_numpy()[::n].astype(float)
    ax2.quiver(args["x"], args["y"], args["dx"], args["dy"], color="blue")
    ax2.quiver(args["x"], args["y"], args["d2x"], args["d2y"], color="red")

    ax3 = plt.figure().add_subplot()
    ax3.plot(p.df.index, p.df.velocityy)

    plt.show(block=False)

    # Start a thread to update it
    running = True
    update_timer = threading.Timer(0.5, update_plots, args=(data_lock, ax, ax2, ax3, p.df,))

    try:
        # Run the simulation
        t = 0
        burn_time = p.burn_time
        while t < burn_time:
            p.burn(t, dt)
            t += dt
        while p.position[1] > 0:
            p.update(t, dt)
            t += dt
    except KeyboardInterrupt as e:
        running = False
        print(traceback.format_exc())
        update_thread.join()
    print(f"Projectile lands at {t} seconds with a velocity of {p.velocity[1]} m/s")

    # Start saving the data since it sometimes takes awhile
    save_thread = threading.Thread(target=save, args=(p.df,))
    save_thread.start()

    plt.show()

    if save_thread.is_alive():
        print("Still saving data...")
        save_thread.join()
    
    running = False
    update_thread.join()

def save(df):
    df.to_csv(f"data-{datetime.now().isoformat(timespec='seconds').replace(':', '')}.csv")

if __name__ == "__main__":
    main()
