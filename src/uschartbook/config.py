from pathlib import Path

import pandas as pd
import numpy as np
from uschartbook.api_key import *
import statsmodels.api as sm
import re
import os
os.environ['X13PATH'] = '/home/brian/Documents/econ_data/micro/x13as/'

from statsmodels.tsa.x13 import x13_arima_analysis

data_dir = Path('../chartbook/data/')
text_dir = Path('../chartbook/text/')
cps_dir = Path('/home/brian/Documents/CPS/data/clean/')
