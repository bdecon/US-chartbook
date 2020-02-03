from pathlib import Path

import pandas as pd
import numpy as np
from uschartbook.api_key import *
import statsmodels.api as sm
import re
import itertools
import os
os.environ['X13PATH'] = '/home/brian/Documents/econ_data/micro/x13as/'

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Polygon
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize

plt.rc('font', family='Lato')

from statsmodels.tsa.x13 import x13_arima_analysis

data_dir = Path('../chartbook/data/')
text_dir = Path('../chartbook/text/')
cps_dir = Path('/home/brian/Documents/CPS/data/clean/')
acs_dir = Path('/home/brian/Documents/ACS/')
