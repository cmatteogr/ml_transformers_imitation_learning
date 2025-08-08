"""
Utils
"""
import pandas as pd
from ydata_profiling import ProfileReport


def generate_profiling_report(
    title: str,
    report_filepath: str,
    df: pd.DataFrame = None,
    data_filepath: str = None,
    type_schema: dict = None,
    minimal: bool = True,
    tsmode: bool=False,
    **read_csv_kwargs
):
    """
    Generates and saves a ydata-profiling report for a DataFrame.

    The data can be provided either as a pandas DataFrame directly or via a filepath.

    Args:
        title: The title for the report.
        report_filepath: The path where the HTML report will be saved.
        df: An existing pandas DataFrame to profile.
        data_filepath: The path to the CSV file to read and profile.
        type_schema: A dictionary specifying column types for the profiler.
        minimal: If True, generates a minimal report.
        tsmode: df is a time-series
        **read_csv_kwargs: Additional keyword arguments to pass to pd.read_csv()
    """
    # Ensure data is provided from only one source.
    if df is not None and data_filepath:
        raise ValueError("Data should be provided via 'df' or 'data_filepath', not both.")

    # Read data from filepath if a DataFrame isn't provided directly.
    if data_filepath:
        try:
            df = pd.read_csv(data_filepath, **read_csv_kwargs).head(50000)
        except FileNotFoundError:
            print(f"Error: The file was not found at {data_filepath}")
            return
        except Exception as e:
            print(f"An error occurred while reading the CSV file: {e}")
            return

    if df is None:
        raise ValueError("No data provided. Please specify 'df' or 'data_filepath'.")

    # Generate data profiling report.
    print(f"Generating profile report for '{title}'...")
    df_profile = ProfileReport(df, tsmode=tsmode, title=title, minimal=minimal, type_schema=type_schema)

    # Export profiling report.
    df_profile.to_file(report_filepath)
    print(f"Report saved to {report_filepath}")
