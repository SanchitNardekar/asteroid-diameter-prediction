# src/asteroid-diameter-prediction/train/make_dataset.py

import click
from loguru import logger
import os

import pandas as pd

@click.command()
@click.option('--reprocess', default=False, help='Runs process pipeline from scratch')
def main(reprocess: bool = False):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    input_path = '~/asteroid-diameter-prediction/data/Asteroid.csv'
    output_path = '~/asteroid-diameter-prediction/data/Asteroid_processed.csv'

    if reprocess:
        df = pd.read_csv(input_path)

        logger.info('Processing data...')

        converted_vals = []
        for idx, val in enumerate(df.diameter.tolist()):
            if isinstance(val, str):
                converted_vals.append(float(val))
            else:
                converted_vals.append(float(val))

        df.diameter = converted_vals
        df.diameter = pd.to_numeric(df.diameter)

        df_numerical = df[[col for col, dtype in zip(df.columns, df.dtypes) if dtype != object]]

        # Use imputer here
        # df_cleaned = df[df["diameter"].notnull()]
        # df_cleaned = df_cleaned[[col for col in df_cleaned.columns if (df_cleaned[col].count() / len(df_cleaned)) > 0.95]]

        df_numerical.to_csv(output_path, index=False)
        logger.success(f'Created output file @ {output_path}')
    else:
        logger.info("Output file already exists.")



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
