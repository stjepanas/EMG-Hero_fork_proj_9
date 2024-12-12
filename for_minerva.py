import h5py
import numpy as np
import pandas as pd

my_dataset = h5py.File('dataset_pretrain.h5', 'r')
a = my_dataset.keys()
print(a)