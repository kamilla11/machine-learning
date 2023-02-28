from easydict import EasyDict
import numpy as np
cfg = EasyDict()
cfg.dataframe_path = 'linear_regression_dataset.csv'

max_degree_of_polynomials = 8
cfg.base_functions = [lambda x, i=i: x ** i for i in range(1, max_degree_of_polynomials+1)]# list of basis functions

#for f in cfg.base_functions:
#    print(f(2))
