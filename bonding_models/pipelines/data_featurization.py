import re
import numpy as np

from pandas import concat

from pymatgen.core.composition import Composition
from pymatgen.core.composition import Element

from dataspace.workspaces.local_db import MongoFrame

from matminer.featurizers.composition import Miedema, ElementProperty
from matminer.featurizers.utils.stats import PropertyStats


"""This module implements a workspace for featurizing a collection of
half-Heusler materials. The features are pairwise elemental properties.
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

    def update_features(self, criteria=None, remove_selective_features=True):
        """Updates the "features" document fields in the collection.

        Args:
            criteria: (son) A pymongo query operator used to select entries.
            remove_selective_features: (bool) Whether to remove features that
                can not be computed for all compounds in the dataset.
        """

        criteria = {} or criteria  # empty dict if you want all compounds

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


if __name__ == '__main__':

    workspace = AFLOWFeatures()
    workspace.update_features()
