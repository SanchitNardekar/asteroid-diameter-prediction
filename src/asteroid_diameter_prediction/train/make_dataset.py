# src/asteroid-diameter-prediction/train/make_dataset.py

import click
from loguru import logger
from pathlib import Path
import os

import pandas as pd


def drop_negative(df):
    """ Drops all rows with negative values in a column.

        Args:
            df = dataframe to be transformed

        Returns:
            Transfored dataframe
    """
    for col in df:
        if df[col].dtype == 'object':
            continue
        else:
            if df[col][df[col] < 0].count() > 0:
                df.drop(df[df[col] < 0].index, axis=0, inplace=True)
            else:
                continue
    return df


def VIF_calc(df):
    """ Calculate VIF (Variance inflation factor) for all columns in a dataframe.

        Args:
            df = dataframe to calculate VIF

        Returns:
            pd.Series with df.columns as index and VIF as values.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    aux_df = df.assign(const=1)
    vif = pd.Series([variance_inflation_factor(aux_df.values, i) 
               for i in range(len(aux_df.columns))], 
              index=aux_df.columns)
    return vif


@click.command()
@click.option('--reprocess', default=False, help='Runs process pipeline from scratch')
def main(reprocess: bool = False):
    """ Runs data processing scripts to turn raw data from into
        cleaned data ready to be analyzed.

        Args:
            reprocess, bool = TRUE if reprocessing raw data from scratch, else False
    """

    input_path = '/Users/sanchitnardekar/asteroid-diameter-prediction/data/Asteroid.csv'
    output_path = '/Users/sanchitnardekar/asteroid-diameter-prediction/data/Asteroid_processed.csv'

    if reprocess or not Path(output_path).exists():
        df = pd.read_csv(input_path)

        logger.info('Processing data...')

        converted_vals = []
        for idx, val in enumerate(df.diameter.tolist()):
            if isinstance(val, str):
                converted_vals.append(float(val))
            else:
                converted_vals.append(float(val))

        # Convert response to numerical
        df.diameter = converted_vals
        df.diameter = pd.to_numeric(df.diameter)

        # Drop columns with more than 5% of NULLs and rows with NULLs TODO: replace with median imputer
        df = df[[col for col in df.columns if (df[col].count() / len(df)) > 0.95]]
        df = df.dropna()

        # Drop rows with negative values
        df = drop_negative(df)

        # Get only numeric columns
        df = df[[col for col, dtype in zip(df.columns, df.dtypes) if dtype != object]]

        # Get the variables with the least multi-collinearity
        df = df[[col for col, vif in zip(df.columns, VIF_calc(df).values) if vif < 5]]

        df.to_csv(output_path, index=False)
        logger.success(f'Created output file @ {output_path}')
    else:
        logger.info("Output file already exists.")



if __name__ == '__main__':

    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
