# src/asteroid-diameter-prediction/train/train_model.py

import click
from loguru import logger
from pathlib import Path
import dotenv
import os

import pandas as pd
import numpy as np
from sklearn import metrics, linear_model, model_selection
from scipy import stats
import pickle

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

    kfold = model_selection.KFold(n_splits = 10, shuffle=True, random_state = 42)
    logger.debug('Created K-Fold splits.')

    scores = []
    rmses = []
    for train_index, test_index in kfold.split(X):
        x_train, x_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], Y.iloc[train_index], Y.iloc[test_index]

        reg.fit(x_train, y_train)
        scores.append([reg.score(x_test, y_test),
                       (np.sqrt(metrics.mean_squared_error(y_test, reg.predict(x_test))))])

    df_kfold = pd.DataFrame(data = np.array(scores), columns = ['RSquared', 'RMSE'])
    logger.info("K-Fold CV:\n", df_kfold)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state = 42, shuffle = True)

    reg.fit(x_train, y_train)
    logger.success('Fitted model on training set')
    
    df_coef = pd.DataFrame(data = np.array([[m, n] for m, n in zip(X.columns, reg.coef_)]), columns = ['column', 'coef'])
    rss_calc = metrics.mean_squared_error(y_test, reg.predict(x_test)) * len(X)
    df_coef['SE'] = [standard_error(X, col, rss_calc) for col in X.columns]
    df_coef.coef = pd.to_numeric(df_coef.coef)
    df_coef['t'] = df_coef['coef'] / df_coef['SE']
    df_coef['pvalue'] = df_coef.apply(lambda row: stats.t.sf(np.abs(row['t']), len(X)-1)*100, axis=1)

    logger.info("Coefficient Analysis:\n", df_coef)
    
    logger.info("R-Squared: {0:.3f} \nRMSE: {1:.3f}".format(reg.score(x_test, y_test), 
                                    np.sqrt(metrics.mean_squared_error(y_test, reg.predict(x_test)))))

    # sns.distplot(y_test.values, kde=True, label = 'True')
    # sns.distplot(reg.predict(x_test), kde=True, label = 'Prediction')
    # plt.legend()
    # plt.show()

    model_path = utils.get_git_root() + '/src/asteroid_diameter_prediction/models/diameter_prediction_model.sav'
    pickle.dump(reg, open(model_path, 'wb'))
    logger.success(f'Model saved @ {model_path}')
