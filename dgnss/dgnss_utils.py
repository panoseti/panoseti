"""
Utilities for differential GNSS timing experiments.
Includes routines for the following tasks:
    - Extracting timing information from PH files produced by the experiments.
    - Interfaces for manipulating timing information.
"""
import typing
import os
import json
import sys
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import seaborn_image as isns
import matplotlib.pyplot as plt

sys.path.append("../util")

import panoseti_file_interfaces

import config_file
import pff
import image_quantiles