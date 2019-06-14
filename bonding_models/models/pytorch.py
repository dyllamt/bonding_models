import numpy as np

from sklearn.model_selection import GridSearchCV

from skorch.regressor import NeuralNetRegressor

from torch import cat
from torch.nn import Module, Linear

from model_formation_enthalpy import collect_latent_model_data,\
    load_enthalpy_data

"""Implements a causal model for estimating the latent bonding energy
between pairwise atoms in the half-Heusler structure.
"""


class PairwiseLayer(Module):
    """A model for pairwise (between atomic sites) bonding enthalpies. The
    inputs to the model are pairwise composition features. The output of the
    model is the pairwise bonding enthalpy for that bonding pair.
    """
    def __init__(self, n_input_features=4, n_latent_features=4):
        super(PairwiseLayer, self).__init__()
        self.layer_1 = Linear(n_input_features, n_latent_features)
        self.layer_2 = Linear(n_latent_features, 1)

    def forward(self, x):
        latent_features = self.layer_1(x)  # creates a latent feature matrix
        return self.layer_2(latent_features)  # calculates the bond enthalpy


class CompetitionLayer(Module):
    """A model for competition or cooperation between multiple pairwise bonding
    interactions. The inputs to the model are partial bonding enthalpies. The
    output of the model is the total formation enthalpy.
    """
    def __init__(self, n_input):
        super(CompetitionLayer, self).__init__()

    def forward(self, x):
        pass


class NonInteractingModel(Module):
    """Model that represents the pairwise bonding enthalpies in the
    half-Heusler structure family. There are two tetrehedral sites and one
    bcc cordinated sites for a total of three unique bonding interactions.
    The input to this model is twelve pariwise features. The output of the
    model is the total formation enthalpy.
    """
    def __init__(self, n_input_per_pair=4, n_latent_per_pair=4):
        super(NonInteractingModel, self).__init__()
        self.n_input_per_pair = n_input_per_pair
        self.bcc_tet1 = PairwiseLayer(n_input_per_pair, n_latent_per_pair)
        self.bcc_tet2 = PairwiseLayer(n_input_per_pair, n_latent_per_pair)
        self.tet1_tet2 = PairwiseLayer(n_input_per_pair, n_latent_per_pair)
        self.total = Linear(3, 1)

    def forward(self, x):

        # parses the input into the individual bond pairs (sub-layers)
        input_bcc_tet1 = \
            x[:, 0:self.n_input_per_pair]
        input_bcc_tet2 = \
            x[:, self.n_input_per_pair:2 * self.n_input_per_pair]
        input_tet1_tet2 = \
            x[:, 2 * self.n_input_per_pair:3 * self.n_input_per_pair]

        # computes the individual bond enthalpies as seperate sub-layers
        formation_bcc_tet1 = self.bcc_tet1(input_bcc_tet1)
        formation_bcc_tet2 = self.bcc_tet2(input_bcc_tet2)
        formation_tet1_tet2 = self.tet1_tet2(input_tet1_tet2)

        # combines the individual bond enthalpies in the final layer
        partial_enthalpies = cat(
            [formation_bcc_tet1, formation_bcc_tet2, formation_tet1_tet2],
            dim=1)
        return self.total(partial_enthalpies)


class LinearInteractingModel(Module):
    """Model that represents the pairwise bonding enthalpies in the
    half-Heusler structure family. There are two tetrehedral sites and one
    bcc cordinated sites for a total of three unique bonding interactions.
    The input to this model is twelve pariwise features. The output of the
    model is the total formation enthalpy.
    """
    def __init__(self, n_input_per_pair=4, n_latent_per_pair=4):
        super(LinearInteractingModel, self).__init__()
        self.n_input_per_pair = n_input_per_pair
        self.bcc_tet1 = PairwiseLayer(n_input_per_pair, n_latent_per_pair)
        self.bcc_tet2 = PairwiseLayer(n_input_per_pair, n_latent_per_pair)
        self.tet1_tet2 = PairwiseLayer(n_input_per_pair, n_latent_per_pair)
        self.total = Linear(3, 1)

    def forward(self, x):

        # parses the input into the individual bond pairs (sub-layers)
        input_bcc_tet1 = \
            x[:, 0:self.n_input_per_pair]
        input_bcc_tet2 = \
            x[:, self.n_input_per_pair:2 * self.n_input_per_pair]
        input_tet1_tet2 = \
            x[:, 2 * self.n_input_per_pair:3 * self.n_input_per_pair]

        # computes the individual bond enthalpies as seperate sub-layers
        formation_bcc_tet1 = self.bcc_tet1(input_bcc_tet1)
        formation_bcc_tet2 = self.bcc_tet2(input_bcc_tet2)
        formation_tet1_tet2 = self.tet1_tet2(input_tet1_tet2)

        # combines the individual bond enthalpies in the final layer
        partial_enthalpies = cat(
            [formation_bcc_tet1, formation_bcc_tet2, formation_tet1_tet2],
            dim=1)
        return self.total(partial_enthalpies)


if __name__ == '__main__':

    # defines the model for testing
    net = NeuralNetRegressor(module=NonInteractingModel)
    params = {
        'lr': [0.01],
        'max_epochs': [20],
        'module__n_latent_per_pair': [3]
    }

    # retrieves data from storage
    data = load_enthalpy_data()
    data = collect_latent_model_data(data).dropna()
    X = data.drop('Enthalpy', axis=1).values.astype(np.float32)
    y = data[['Enthalpy']].values.astype(np.float32)

    gs = GridSearchCV(net, params, refit=False, cv=3, scoring='r2')
    gs.fit(X, y)
    print(gs.best_score_, gs.best_params_)