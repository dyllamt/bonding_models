import numpy as np

from pandas import DataFrame

from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


"""This module implements models for analyzing pairwise interactions in half-
Heusler materials. Features are assumed to be pariwise elemental features
and the targets are assumed to be thermodynamic quantities.
"""


class GSLinearModel(GridSearchCV):
    """A linear model trained using gradient descent. A cross-vaidation loop
    optimizes the model hyperparameters using the r2 scoring metric.

    Pipeline:
        1. StandardScaler
        2. SGDRegressor
    """
    def __init__(self, gs_params):
        """
        Args:
            gs_params (dict) search parameters for hyperparameter optimization.
        """

        # constructs pipline steps
        scale = StandardScaler()
        regressor = SGDRegressor()
        pipe = Pipeline(steps=[('scale', scale), ('regressor', regressor)])

        # constructs the gridsearch estimator
        GridSearchCV.__init__(pipe, gs_params, refit=True, cv=5, scoring='r2')

    def estimate_partial_energies(self, features):
        """Returns the partial energies in each bond. Estimating the partial
        energies is possible because each feature describes only one of the
        bonding interactions.

        Note:
            The fit() method must be called before calling this method.

        Args:
            features (array-like) Pairwise bonding features that are ordered
                according to their pairwise interactions. In addition, the
                number of features for each bonding pair must be the same.

        Returns:
            (DataFrame) The estimated partial energies for each sample.
        """

        # applies the standard scaling to the features
        A, B, C = np.split(
            self.best_estimator_.steps[0][1].transform(features), 3, axis=1)

        # collects the coefficients of the linear model (last pipeline step)
        a, b, c = np.split(
            self.best_estimator_.named_steps['regressor'].coef_, 3)

        # computes the partial energies, which sum to the total energy
        partials = DataFrame()
        partials['partial_energy bcc_tet1'] = np.sum(A * a, axis=1)
        partials['partial_energy bcc_tet2'] = np.sum(B * b, axis=1)
        partials['partial_energy tet1_tet2'] = np.sum(C * c, axis=1)
        return partials


class GSRandomForest(GridSearchCV):
    """A tree-based model with a cross-validation loop for optimizing model
    hyperparameters using the r2 scoring metric.

    Pipeline:
        1. StandardScaler
        2. RandomForestRegressor
    """
    def __init__(self, gs_params):
        """
        Args:
            gs_params (dict) search parameters for hyperparameter optimization.
        """

        # constructs pipline steps
        scale = StandardScaler()
        regressor = RandomForestRegressor()
        pipe = Pipeline(steps=[('scale', scale), ('regressor', regressor)])

        # constructs the gridsearch estimator
        GridSearchCV.__init__(pipe, gs_params, refit=True, cv=5, scoring='r2')
