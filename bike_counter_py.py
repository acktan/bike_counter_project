from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

data = pd.read_parquet(Path('data') / 'train.parquet')
