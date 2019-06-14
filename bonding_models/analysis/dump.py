import matplotlib.pyplot as plt

from dataspace.workspaces.local_db import MongoFrame

from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

"""Holdover from model refactoring
"""


class AFLOWFeatures(MongoFrame):
    """Featurizes half-Heusler structures in a mongodb collection.

    Feature vectors contain pairwise features of elemental attributes.

    Attributes:
        see MongoFrame for a list of attributes.
    """
    def __init__(self, path='/data/db', database='aflow_HH',
                 collection='entries'):
        """Initializes the workspace from a pymongo.Collection spec.

        Args:
            path: (str) Path to the local mongodb directory.
            database: (str) Name of the pymongo database.
            collection: (str) Name of the pymongo collection.
        """
        MongoFrame.__init__(self, path=path, database=database,
                            collection=collection)

    def predict_partial_energies(self, target='energy_cell'):
        """2. Extracts partial formation energies using linear models.

        Args:
            target: (st) The column name of the target property.
        """

        # cleanses data of rows containing nan
        self.memory.dropna(inplace=True)

        # selects features, targets, and the analysis pipeline
        features, target = self._select_features_target(target)
        pipe = self._select_analyis_pipeline()

        # trains optimal model (based on cross-validation scores)
        model = self._train_best_model(pipe, features, target)

        # reports the scoring for the model (regression plot and r2)
        self._report_scoring(model, features, target)

        # computes partial formation energies
        self._compute_partial_energies(model, features)

    def meta_analysis(self, target='enthalpy_formation_cell', limit=0.0):
        """3. Performs meta analysis of the target with the partial energies.

        Args:
            target: (str) The column name of the target property.
            limit: (float) Only analyze entries with targets less than limit.
        """

        # selects partial energies and targets
        partials, target = self._select_partials_target(target, limit)

        # plots target variable in latent space
        self._plot_target_vs_partials(target, partials)

    def to_csv(self, fname='energetics_data.csv', fdir='.'):
        """Exports current DataFrame memory to a csv file.

        Args:
            fname: (str) Name of the saved file.
            fdir: (str) Directory to save the file in (working dir by default).
        """
        self.memory.to_csv('{}/{}'.format(fdir, fname))

    def _select_features_target(self, target):
        """Returns a features DataFrame and a target DataFrame.
        """
        return (self.memory.filter(like='pairwise_feature'),
                self.memory.filter(items=[target]))

    def _select_partials_target(self, target, limit):
        """Returns a features DataFrame and a target DataFrame.
        """
        data = self.memory.loc[self.memory[target].values < limit]
        return (data.filter(like='partial_energy'),
                data.filter(items=[target]))

    def _select_analyis_pipeline(self):
        """Returns a sklearn Pipeline.
        """
        scale = StandardScaler()
        regressor = SGDRegressor()
        return Pipeline(steps=[('scale', scale), ('regressor', regressor)])

    def _train_best_model(self, pipe, features, target, print_scores=True):
        """Returns a model fit to the data.

        This method first trains models using a grid of parameters. Then the
        model that scored the highest in cv is refit with all the data.
        """
        gs_params = {
            'regressor__penalty': ['l1', 'l2'],
            'regressor__alpha': [1e-7, 1e-6, 0.00001, 0.0001, 0.001, 0.01],
            'regressor__tol': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]}
        model = GridSearchCV(pipe, gs_params, refit=True, cv=3, scoring='r2')
        model.fit(features, target.values.flatten())
        return model

    def _report_scoring(self, model, features, target):
        """Reports the model performance with a regression plot and r2.
        """
        print(model.best_score_, model.best_params_)
        plt.plot(model.predict(features), target.values.flatten(), 'ko')
        plt.xlabel('Predicted Value')
        plt.ylabel('Actual Value')
        plt.show()

    def _plot_target_vs_partials(self, target, partials):
        """Plots a target variable vs the partial energies.
        """

        # loads 3d plotting objects
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # plots the data
        scat = ax.scatter(
            partials.filter(like='bcc_tet1').values.flatten(),
            partials.filter(like='bcc_tet2').values.flatten(),
            partials.filter(like='tet1_tet2').values.flatten(),
            c=target.values.flatten())
        ax.set_xlabel('eV bcc-tet1')
        ax.set_ylabel('eV bcc-tet2')
        ax.set_zlabel('eV tet1-tet2')
        fig.colorbar(scat)
        plt.show()
