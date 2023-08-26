# Airbrakes Simulator

This is a simulator for the airbrakes of a rocket.
It is written in Python.

## Usage

Make sure you have Python 3 installed.
It is suggested to run the program within a virtual environment.
The steps to do this are below.

```bash
# Create the Virtual Environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the program
python3 main.py --config config.yml
```

Make sure all dependancies are met before running the program.
These are listed in the `requirements.txt` file.

## Configuration

The simulator is configured using a YAML file.
An example configuration file is provided in `config.yml`.
A full list of possible keys, their values, and what they do can be found in the `config_full.yml` file.
