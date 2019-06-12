import re
import numpy as np
import matplotlib.pyplot as plt

from pandas import DataFrame, concat

from pymatgen.core.composition import Composition
from pymatgen.core.composition import Element

from dataspace.workspaces.local_db import MongoFrame

from matminer.featurizers.composition import Miedema, ElementProperty
from matminer.featurizers.utils.stats import PropertyStats

from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


"""This module implements models for the energetics of half-Heuslers. The
models are NOT intended to be "black-box" estimators, and therefore sacrifice
estimation power for physical interpretability.
"""


class PairwiseElementProperty(ElementProperty):
    """Computes features for binary compositions.
    """
    def __init__(self, data_source, features, stats):
        ElementProperty.__init__(self, data_source, features, stats)

    def featurize(self, comp):
        """This featurizer requires a string representation of a composition.
        """

        all_attributes = []

        pstats = ExtendedPropertyStats()

        elements = [Element(i.group(0)) for i in re.finditer(
            r"([A-Z][a-z]*)\s*([-*\.\d]*)", comp)]

        for attr in self.features:
            element_data = [self.data_source.get_elemental_property(
                e, attr) for e in elements]

            for stat in self.stats:
                all_attributes.append(pstats.calc_stat(element_data, stat))

        return all_attributes


class ExtendedPropertyStats(PropertyStats):
    """Extends the Property Stats featurizer to include "difference"
    """

    @staticmethod
    def calc_stat(data_lst, stat, weights=None):
        """
        Compute a property statistic
        Args:
            data_lst (list of floats): list of values
            stat (str) - Name of property to be compute. If there are arguments
                to the statistics function, these should be added after the
                name and separated by two colons. For example, the 2nd Holder
                mean would be "holder_mean::2"
            weights (list of floats): (Optional) weights for each element
        Returns:
            float - Desired statistic
        """
        statistics = stat.split("::")
        return getattr(ExtendedPropertyStats, statistics[0])(data_lst, weights,
                                                             *statistics[1:])

    @staticmethod
    def difference(data_lst, weights=None):
        """Calculates the difference between the first two list elements
        Args:
            data_lst (list of floats): List of values to be assessed
            weights: (ignored)
        Returns:
            minimum value
        """
        return (data_lst[0] - data_lst[1]) if not np.any(
            np.isnan(data_lst)) else float("nan")


class EnergeticsModel(MongoFrame):
    """Analyzes the energetics of half-Heusler materials from a local database.

    Attributes:
        same as MongoFrame.
    """
    def __init__(self, path='/data/db', database='aflow',
                 collection='candidates'):
        """Initializes the workspace from a pymongo.Collection spec.

        Args:
            path: (str) Path to the local mongodb directory.
            database: (str) Name of the pymongo database.
            collection: (str) Name of the pymongo collection.
        """
        MongoFrame.__init__(self, path=path, database=database,
                            collection=collection)

    def prepare_features(self, criteria=None, remove_selective_features=True):
        """1. Prepares pairwise composition features.

        Args:
            criteria: (son) A pymongo query operator used to select entries.
            remove_selective_features: (bool) Whether to remove features.
        """

        # loads requested data from storage
        self.from_storage(filter=criteria)

        # generates new columns of pairwise compositions
        self._generate_site_collections()
        self._generate_pairwise_compositions()

        # generates new columns of pairwise features
        self._generate_pairwise_features()
        self._symmetrize_tet1_tet2_features()

        # removes features that are nan for a significant number of entries
        if remove_selective_features:
            self._remove_fere()
            self._remove_miedema()
        else:
            pass

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

    def _generate_site_collections(self):
        """Generates columns for bcc and tet atoms.

        Columns are labeled "bcc_atom" (str) and "tet_atoms" (list of str).
        """

        # determines the atoms in tet sites
        species = self.memory['species']
        bcc_atom = self.memory['bcc_atom']
        tet_atoms = [
            list(set(i) - set([j])) for i, j in zip(species, bcc_atom)]

        # orders each set of tet elements by group
        tet_groups = [[Element(j).group for j in i] for i in tet_atoms]
        for i, element_group in enumerate(tet_groups):

            # reverses the order if the order is wrong
            if element_group[0] > element_group[1]:
                tet_atoms[i] = tet_atoms[i][::-1]
            else:
                pass
        self.memory['tet_atoms'] = tet_atoms

    def _generate_pairwise_compositions(self):
        """Generates columns of pairwise compositions.

        Column labels contain the prefix "pairwise_composition {}", where the
        string in parethesis is either "bcc_tet1", "bcc_tet2", or "tet1_tet2".
        """
        site_collections = list(zip(
            self.memory['bcc_atom'], self.memory['tet_atoms']))
        self.memory['pairwise_composition bcc_tet1'] = [
            '{}{}'.format(bcc, tet[0]) for bcc, tet in site_collections]
        self.memory['pairwise_composition bcc_tet2'] = [
            '{}{}'.format(bcc, tet[1]) for bcc, tet in site_collections]
        self.memory['pairwise_composition tet1_tet2'] = [
            '{}{}'.format(tet[0], tet[1]) for bcc, tet in site_collections]

    def _generate_pairwise_features(self):
        """Generates features for each bonding pair.

        Column labels contain a prefix "pairwise_feature {}", where the
        string in parethesis is either "bcc_tet1", "bcc_tet2", or "tet1_tet2".
        """

        # initializes requested matminer featurizers
        feat_element_property = PairwiseElementProperty(
            data_source='deml',
            features=['atom_radius', 'electronegativity', 'first_ioniz',
                      'col_num', 'row_num', 'molar_vol', 'heat_fusion',
                      'melting_point', 'GGAU_Etot', 'mus_fere',
                      'FERE correction'],
            stats=['difference', 'mean'])
        feat_miedema = Miedema(
            struct_types=['inter', 'amor'])

        # generates freatures for each bonding pair
        for bonding_pair in ['bcc_tet1', 'bcc_tet2', 'tet1_tet2']:

            # gets the string composition of the bonding pairs
            composition_index = 'pairwise_composition {}'.format(
                bonding_pair)
            composition = self.memory[[composition_index]]

            # adds ElementProperty features
            features = feat_element_property.featurize_dataframe(
                df=composition,
                col_id=composition_index,
                inplace=False
            ).drop(
                labels=composition_index,
                axis=1
            ).add_prefix(
                prefix='pairwise_feature {} '.format(bonding_pair))
            self.memory = concat([self.memory, features], axis=1)

            # gets the pymatgen.Composition of the bonding pairs
            composition[composition_index] = [
                Composition(i) for i in composition[composition_index]]

            # adds Miedema features
            features = feat_miedema.featurize_dataframe(
                df=composition,
                col_id=composition_index,
                inplace=False
            ).drop(
                labels=composition_index,
                axis=1
            ).add_prefix(
                prefix='pairwise_feature {} '.format(bonding_pair))
            self.memory = concat([self.memory, features], axis=1)

    def _symmetrize_tet1_tet2_features(self):
        """Takes the absolute value of any "difference" features.
        """
        for column in self.memory.columns:
            if ' tet1_tet2 difference ' in column:
                self.memory[column] = np.abs(self.memory[column].values)
            else:
                pass

    def _remove_fere(self):
        """Deletes feature columns involving FERE corrections.
        """
        labels = self.memory.columns
        labels = labels[np.array([('fere' in label) or ('FERE' in label) or
                                  ('GGAU' in label) for label in labels])]
        self.memory.drop(labels, axis=1, inplace=True)

    def _remove_miedema(self):
        """Deletes feature columns involving the miedema featurizer.
        """
        labels = self.memory.columns
        labels = labels[np.array(['Miedema' in label for label in labels])]
        self.memory.drop(labels, axis=1, inplace=True)

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

    def _compute_partial_energies(self, model, features):
        """Computes the partial formation energies, which sum to the total.
        """

        # transforms the features using the first step of the pipeline
        A, B, C = np.split(
            model.best_estimator_.steps[0][1].transform(features), 3, axis=1)

        # collects the coefficients of the linear model (last pipeline step)
        a, b, c = np.split(
            model.best_estimator_.named_steps['regressor'].coef_, 3)

        # computes the partial energies, which sum to the total energy
        self.memory['partial_energy bcc_tet1'] = np.sum(A * a, axis=1)
        self.memory['partial_energy bcc_tet2'] = np.sum(B * b, axis=1)
        self.memory['partial_energy tet1_tet2'] = np.sum(C * c, axis=1)

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


if __name__ == '__main__':

    MODEL = EnergeticsModel()
    MODEL.prepare_features(
        criteria=None,
        remove_selective_features=False)
    MODEL.predict_partial_energies(
        target='energy_cell')
    MODEL.meta_analysis(
        target='enthalpy_formation_cell',
        limit=-2.0)
    MODEL.to_csv()
