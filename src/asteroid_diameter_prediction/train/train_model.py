# src/asteroid-diameter-prediction/train/train_model.py

import click
from loguru import logger
from pathlib import Path
import dotenv
import os
import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics, linear_model, model_selection
from scipy import stats
from dvclive import Live

from asteroid_diameter_prediction import utils

def standard_error(df, col, rss_calc):
    x_bar = np.mean(df[col].tolist())
    feature_std = np.sqrt(sum([(xi - x_bar)**2 for xi in df[col].tolist()]))
    rse = np.sqrt(rss_calc / (len(df) - 2))
    return rse / feature_std


@click.command()
@click.option('--env', default='dev', help='Defines the environment to run the pipeline')
def main(env: str = 'dev'):

    dotenv.load_dotenv()

    df = pd.read_csv(os.environ["DEV_ENRICHED_DATA"])
    logger.debug('Loaded enriched dataset')

    X = df.drop(["diameter"], axis=1)
    Y = df.diameter
    
    reg = linear_model.LinearRegression()

    kfold = model_selection.KFold(n_splits = 10, shuffle=True, random_state = utils.params['train']['seed'])
    logger.debug('Created K-Fold splits.')

    scores = []
    rmses = []
    for train_index, test_index in kfold.split(X):
        x_train, x_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], Y.iloc[train_index], Y.iloc[test_index]

        reg.fit(x_train, y_train)
        scores.append([reg.score(x_test, y_test),
                       (np.sqrt(metrics.mean_squared_error(y_test, reg.predict(x_test))))])

    df_kfold = pd.DataFrame(data = np.array(scores), columns = ['RSquared', 'RMSE'])
    logger.info(f"K-Fold CV: {df_kfold.RSquared.values}")

    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size = utils.params['train']['split'], random_state = utils.params['train']['seed'], shuffle = True)

    reg.fit(x_train, y_train)
    logger.success('Fitted model on training set')
    
    df_coef = pd.DataFrame(data = np.array([[m, n] for m, n in zip(X.columns, reg.coef_)]), columns = ['column', 'coef'])
    rss_calc = metrics.mean_squared_error(y_test, reg.predict(x_test)) * len(X)
    df_coef['SE'] = [standard_error(X, col, rss_calc) for col in X.columns]
    df_coef.coef = pd.to_numeric(df_coef.coef)
    df_coef['t'] = df_coef['coef'] / df_coef['SE']
    df_coef['pvalue'] = df_coef.apply(lambda row: stats.t.sf(np.abs(row['t']), len(X)-1)*100, axis=1)

    logger.info("Coefficient Analysis:\n", df_coef)
    
    rsquared = reg.score(x_test, y_test)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, reg.predict(x_test)))

    logger.info("R-Squared: {0:.3f} \nRMSE: {1:.3f}".format(rsquared, rmse))

    with Live() as live:
        live.log_metric("$R^2$", rsquared)
        live.log_metric("RMSE", rmse)

        coef_live = df_coef[['column', 'coef']].to_dict('records')
        live.log_plot(
            "Coefficient values",
            coef_live,
            x='column',
            y='coef',
            template='bar_horizontal',
            title='Coefficient values for Asteroid diameter prediction',
            x_label='Feature',
            y_label='Coefficient value'
        )

        fig_rsquared, ax_rsquared = plt.subplots()
        sns.kdeplot(df_kfold['RSquared'].values, ax=ax_rsquared)
        plt.title('K-Fold $R^2$')
        live.log_image('K-Fold $R^2$', fig_rsquared)

    model_path = utils.get_git_root() + '/models/diameter_prediction_model.joblib'
    joblib.dump(reg, model_path)
    logger.success(f'Model saved @ {model_path}')
