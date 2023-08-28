# sensor.py
# Classes to help facilitate sensor emulation.

import numpy as np

class Sensor:
    def __init__(self, axes=1, buffer_size=100, noise=0, axes_names=None):
        self.noise = noise
        self.datastreams = [np.zeros(buffer_size) for _ in range(axes)]
        self.data_indexes = [-1 for _ in range(axes)]
        # Setup the axes properties
        for a in range(axes):
            if axes_names is not None:
                attribute_name = axes_names[a]
            else:
                attribute_name = f"data_{a}"
            data_property = property(fget=lambda s: s.get_data(a), doc=f"Get the current value of the {attribute_name} axis.")
            self.__setattr__(attribute_name, data_property)
    
    def push(self, value, axis=0):
        self.data_indexes[axis] = (self.data_indexes[axis] + 1) % len(self.datastreams[axis])
        if type(value) == list or type(value) == tuple or type(value) == np.ndarray:
            padding = len(self.datastreams[axis]) - len(value)
            self.datastreams[axis] = np.pad(np.array(value), padding, mode="constant", constant_values=0)
            # Set the data index to the last value of this array
            self.data_indexes[axis] = len(value)
        else:
            self.datastreams[axis][self.data_indexes] = value
    
    def get_data(self, axis=0):
        return self.datastreams[axis][self.data_indexes[axis]] #+ np.random.normal(0, self.noise)

__all__ = ["Sensor"]
