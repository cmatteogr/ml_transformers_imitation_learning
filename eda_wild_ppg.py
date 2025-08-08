import scipy.io
import pandas as pd

from utils import generate_profiling_report

# read data
data_filepath = 'data/WildPPG_data_sample_56.csv'

# apply profiling
title = "PPG Profiling"
report_name = 'ppg_profiling'
report_filepath = f"data/{report_name}.html"

# Now you can pass the specific arguments for reading your CSV file.
generate_profiling_report(
    report_filepath=report_filepath,
    title=title,
    data_filepath=data_filepath,
    minimal=False,
    tsmode=True
)