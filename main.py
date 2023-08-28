# main.py

import argparse
import yaml

# Local Imports
from simulator import Simulator

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
    simulator = Simulator(config)
    # Then plot the results
    simulator.simulate()
    simulator.plot()

if __name__ == '__main__':
    main()
