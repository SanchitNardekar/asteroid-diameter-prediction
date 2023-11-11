# src/asteroid-diameter-prediction/train/enrich_features.py

import click
from loguru import logger
from pathlib import Path
import dotenv
import os

import pandas as pd
import numpy as np
from asteroid_diameter_prediction.utils import params

@click.command()
@click.option('--reprocess', default=False, help='Runs process pipeline from scratch')
@click.option('--env', default='dev', help='Defines the environment to run the pipeline')
def main(reprocess: bool = False, env: str = 'dev'):

    dotenv.load_dotenv()
    
    processed_path = os.environ["DEV_PROCESSED_DATA"]
    enriched_path = os.environ["DEV_ENRICHED_DATA"]

    if reprocess or not Path(enriched_path).exists():

        df = pd.read_csv(processed_path)

        df_features = df.drop(['diameter'], axis=1)

        logger.debug(f"Features: {df_features.columns}")
        # X = df_features.drop(['diameter', 'a', 'q', 'ad', 'per'], axis=1)

        # df_features['e:n'] = df_features.e * df_features.n
        if params['enrich']['log_features']:
            for col in df_features.columns:
                df_features["log "+ col] = df_features[col].apply(np.log)

        # df_features['log e:log n'] = df_features['log e'] * df_features['log n']
        # df_features['log H: log n_obs_used'] = df_features['log H'] * df_features['log n_obs_used']
        # df_features = df_features.drop('log e:n', axis=1)

        logger.debug(f"feature cols: {df_features.columns}")

        df = pd.concat([df_features, df.diameter], axis=1)

        df.to_csv(enriched_path)
        logger.success(f'Enriched dataset created at {enriched_path}')
    else:
        logger.info("Enriched file already exists")


