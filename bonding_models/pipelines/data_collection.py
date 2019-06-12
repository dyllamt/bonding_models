import numpy as np

from dataspace.base import Pipe
from dataspace.workspaces.local_db import MongoFrame
from dataspace.workspaces.materials_api import APIFrame

from matminer.data_retrieval.retrieve_AFLOW import AFLOWDataRetrieval


"""Implements pipeline segments for constructing a half-Heusler dataset from
the AFLOW database. The dataset includes structural, chemical, and
thermodynamic properties.
"""


class AFLOWEntries(Pipe):
    """Collects a set of half-Heusler materials in local storage (mongodb).

    Half-Heuslers in AFLOW are built from prototype structures. The prototypes
    for the calculated, ternary half-Heuslers follow the pattern: T0003*

    Attributes:
        source: (Workspace) Retriever for AFLOW.
        destination: (Workspace) A local collection for the candidates.
    """
    def __init__(self, path='/data/db', database='aflow',
                 collection='halfheuslers'):
        """Initializes the pipe from a pymongo.Collection spec.

        Args:
            path: (str) Path to the local mongodb directory.
            database: (str) Name of the pymongo database.
            collection: (str) Name of the pymongo collection.
        """
        self.source = APIFrame(AFLOWDataRetrieval)
        self.destination = MongoFrame(path=path, database=database,
                                      collection=collection)

    def build_local(self):
        """Populates the half-Heusler collection with materials data.
        """

        # clears destination storage
        self.destination.delete_storage(clear_collection=True)

        # collects entries from AFLOW
        self._get_candidate_entries()

        # cleans incoming entries
        self._remove_incomplete_entries()
        self._remove_conventional_cells()
        self._identify_bcc_atom()  # needed for removing duplicates
        self._remove_duplicate_entries()

        # stores the cleaned entries
        self.transfer(to='destination')
        self.destination.to_storage(identifier=None)

    def _get_candidate_entries(self):
        """Downloads candidate entries from AFLOW.
        """
        self.source.from_storage(
            criteria={'prototype':
                      {'$in':
                       ['T0003.ABC', 'T0003.BCA', 'T0003.CAB']}},
            properties=['auid', 'aurl', 'prototype',
                        'species', 'composition',
                        'geometry', 'positions_fractional',
                        'energy_cell', 'enthalpy_formation_cell',
                        'Egap', 'volume_cell', 'compound'],
            files=[],
            request_size=10000, request_limit=0, index_auid=False)

        # converts array properties into nested lists
        for key in self.source.memory.columns.values:

            value_of_first_record = self.source.memory[key][0]

            if isinstance(value_of_first_record, np.ndarray):
                self.source.memory[key] = [
                    i.tolist() for i in self.source.memory[key]]

    def _remove_incomplete_entries(self):
        """Removes entries that contain nan.
        """
        self.source.memory.dropna(inplace=True)

    def _remove_conventional_cells(self):
        """Removes entries with conventional (as opposed to primitive) cells.
        """
        n_positions = [
            len(i) for i in self.source.memory['positions_fractional']]
        three_total_sites = (np.array(n_positions) == 3).astype(bool)
        self.source.memory = self.source.memory.loc[three_total_sites]

    def _identify_bcc_atom(self):
        """Adds an additional column specifying the atom in the bcc position.
        """

        # parses the x component of the atomic positions
        positions = self.source.memory['positions_fractional']
        x_positions = [[j[0] for j in i] for i in positions]

        # determines the bcc site index for each entry
        bcc_index = []
        for i, coords in enumerate(x_positions):

            # determines the position of the BCC atom
            if set(coords) == set([0.0, 0.25, 0.5]):
                bcc_position = 0.25
            elif set(coords) == set([2.5, 0.5, 0.75]):
                bcc_position = 0.5
            elif set(coords) == set([0.5, 0.75, 1.0]):
                bcc_position = 0.75
            elif set(coords) == set([0.5, 0.75, 0.0]):
                bcc_position = 0.75
            else:
                raise Exception(
                    'unexpected set of fractional positions:\n{}'.format(
                        coords))

            # determines the site index of the BCC atom
            bcc_index.append(coords.index(bcc_position))

        self.source.memory['bcc_atom'] = [
            i[j] for i, j in zip(self.source.memory['species'], bcc_index)]

    def _remove_duplicate_entries(self):
        """Removes entries that are duplicates.
        """
        self.source.memory.drop_duplicates(
            subset=['bcc_atom', 'compound'], keep='first', inplace=True)


if __name__ == '__main__':

    pipe = AFLOWEntries()
    pipe.build_local()
