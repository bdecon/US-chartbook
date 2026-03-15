from pathlib import Path
import os

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
from uschartbook.api_key import *
import statsmodels.api as sm
import re
import itertools

# X13 seasonal adjustment path
x13_path = os.getenv('X13_PATH', '')
if x13_path:
    os.environ['X13PATH'] = x13_path

# HTTP request headers for government data sources
# Centralized headers to prevent requests from being blocked
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:148.0) Gecko/20100101 Firefox/148.0',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive'
}


def get_headers(referrer=None):
    """Return headers dict, optionally with a Referrer field."""
    headers = REQUEST_HEADERS.copy()
    if referrer:
        headers['Referrer'] = referrer
    return headers

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Polygon
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize

plt.rc('font', family='Lato')

from statsmodels.tsa.x13 import x13_arima_analysis

# Directory paths - configurable via environment variables with sensible defaults
_project_root = Path(__file__).parent.parent.parent
cb_dir = Path(os.getenv('CHARTBOOK_DIR', _project_root / 'notebooks'))
data_dir = _project_root / 'chartbook' / 'data'
text_dir = _project_root / 'chartbook' / 'text'
db_path = Path(os.getenv('DB_PATH', data_dir / 'chartbook.db'))
raw_dir = _project_root / 'notebooks' / 'raw'
cps_dir = Path(os.getenv('CPS_DATA_DIR', ''))
acs_dir = Path(os.getenv('ACS_DATA_DIR', ''))
web_dir = Path(os.getenv('WEBSITE_DIR')) if os.getenv('WEBSITE_DIR') else None
