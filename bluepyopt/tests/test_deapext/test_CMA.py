"""bluepyopt.deapext CMA strategy tests"""

import sys
import types

import numpy

import bluepyopt.deapext.CMA_MO
import bluepyopt.deapext.CMA_SO
import bluepyopt.deapext.utils

import pytest


def make_individual(params, fitness_values=None):
    """Build a WSListIndividual, optionally with fitness values"""
    if fitness_values is None:
        obj_size = 1
    else:
        obj_size = len(fitness_values)
    ind = bluepyopt.deapext.utils.WSListIndividual(
        params, obj_size=obj_size)
    if fitness_values is not None:
        ind.fitness.values = tuple(fitness_values)
    return ind


def make_mo_population(fitness_values):
    """Build a population of individuals with the given fitness values"""
    return [
        make_individual([0.0] * len(values), fitness_values=values)
        for values in fitness_values
    ]


def ind_creator():
    """IndCreator function for the CMA strategies"""
    return bluepyopt.deapext.utils.WSListIndividual(obj_size=1)


def rand_ind_creator():
    """RandIndCreator function for the CMA strategies"""
    return make_individual([0.1, 0.2])


@pytest.fixture
def fake_scoop(monkeypatch):
    """Fake the scoop module so its futures.map resolves to builtin map"""
    futures = types.ModuleType("scoop.futures")
    futures.map = map
    scoop = types.ModuleType("scoop")
    scoop.futures = futures
    monkeypatch.setitem(sys.modules, "scoop", scoop)
    monkeypatch.setitem(sys.modules, "scoop.futures", futures)
    return futures


@pytest.mark.unit
def test_get_hyped():
    """deapext.CMA_MO: Testing get_hyped"""

    pop = make_mo_population([
        (10., 100.), (20., 50.), (15., 200.),
        (30., 80.), (5., 150.), (25., 60.),
    ])
    hv = bluepyopt.deapext.CMA_MO.get_hyped(pop)

    assert len(hv) == len(pop)


@pytest.mark.unit
def test_get_hyped_no_improvement_dimension():
    """deapext.CMA_MO: get_hyped drops dimensions without improvement"""

    # second dimension never improves beyond threshold_improvement
    pop = make_mo_population([
        (10., 245.), (20., 250.), (15., 248.),
        (30., 246.), (5., 249.), (25., 247.),
    ])
    hv = bluepyopt.deapext.CMA_MO.get_hyped(pop)

    assert len(hv) == len(pop)


@pytest.mark.unit
def test_get_hyped_no_dimension():
    """deapext.CMA_MO: get_hyped with no usable dimension returns zeros"""

    pop = make_mo_population(
        [(250., 250.), (245., 248.)])
    hv = bluepyopt.deapext.CMA_MO.get_hyped(pop)

    assert hv == [0.] * len(pop)


@pytest.mark.unit
def test_CMA_MO_default_offspring():
    """deapext.CMA_MO: constructor computes offspring size and max ngen"""

    strategy = bluepyopt.deapext.CMA_MO.CMA_MO(
        centroids=None,
        offspring_size=None,
        sigma=0.4,
        max_ngen=0,
        IndCreator=ind_creator,
        RandIndCreator=rand_ind_creator)

    # 4 + 3 * log(2)
    assert strategy.lambda_ == 6
    assert strategy.problem_size == 2
    assert strategy.stopping_conditions[0].max_ngen == pytest.approx(
        100 + 50 * 5 ** 2 / numpy.sqrt(6))


@pytest.mark.unit
def test_CMA_MO_centroids_cycle():
    """deapext.CMA_MO: fewer centroids than offspring are cycled"""

    strategy = bluepyopt.deapext.CMA_MO.CMA_MO(
        centroids=[make_individual([0.1, 0.2])],
        offspring_size=4,
        sigma=0.4,
        max_ngen=10,
        IndCreator=ind_creator,
        RandIndCreator=rand_ind_creator)

    assert strategy.lambda_ == 4
    assert len(strategy.parents) == 4


@pytest.mark.unit
def test_CMA_MO_matching_centroids():
    """deapext.CMA_MO: centroids matching offspring are used directly"""

    centroids = [
        make_individual([0.1, 0.2]),
        make_individual([0.3, 0.4]),
    ]
    strategy = bluepyopt.deapext.CMA_MO.CMA_MO(
        centroids=centroids,
        offspring_size=2,
        sigma=0.4,
        max_ngen=10,
        IndCreator=ind_creator,
        RandIndCreator=rand_ind_creator)

    assert strategy.lambda_ == 2
    assert strategy.parents == centroids


@pytest.mark.unit
def test_CMA_MO_scoop_with_map_function(fake_scoop):
    """deapext.CMA_MO: scoop and map_function are exclusive"""

    with pytest.raises(Exception, match="Impossible to use scoop"):
        bluepyopt.deapext.CMA_MO.CMA_MO(
            centroids=None,
            offspring_size=4,
            sigma=0.4,
            max_ngen=10,
            IndCreator=ind_creator,
            RandIndCreator=rand_ind_creator,
            map_function=map,
            use_scoop=True)


@pytest.mark.unit
def test_CMA_MO_scoop(fake_scoop):
    """deapext.CMA_MO: scoop futures are used as map function"""

    strategy = bluepyopt.deapext.CMA_MO.CMA_MO(
        centroids=None,
        offspring_size=4,
        sigma=0.4,
        max_ngen=10,
        IndCreator=ind_creator,
        RandIndCreator=rand_ind_creator,
        use_scoop=True)

    assert strategy.map_function == fake_scoop.map


@pytest.mark.unit
def test_CMA_MO_select():
    """deapext.CMA_MO: _select with weight_hv of 0 and 1"""

    strategy = bluepyopt.deapext.CMA_MO.CMA_MO(
        centroids=None,
        offspring_size=4,
        sigma=0.4,
        max_ngen=10,
        IndCreator=ind_creator,
        RandIndCreator=rand_ind_creator)

    candidates = make_mo_population(
        [(1.,), (2.,), (3.,), (4.,), (5.,), (6.,)])

    # weight_hv == 0.0: pure fitness ordering, lowest sums are chosen
    strategy.weight_hv = 0.0
    chosen, not_chosen = strategy._select(candidates)
    assert len(chosen) == 2
    assert len(not_chosen) == 4
    assert set(ind.fitness.values[0] for ind in chosen) == {1., 2.}

    # weight_hv == 1.0: pure hypervolume ordering, largest volumes are chosen
    numpy.random.seed(42)
    strategy.weight_hv = 1.0
    chosen, not_chosen = strategy._select(candidates)
    assert len(chosen) == 2
    assert len(not_chosen) == 4
    assert set(ind.fitness.values[0] for ind in chosen) == {1., 2.}

    # weight_hv between 0 and 1: both orderings agree on this population
    numpy.random.seed(42)
    strategy.weight_hv = 0.5
    chosen, not_chosen = strategy._select(candidates)
    assert set(ind.fitness.values[0] for ind in chosen) == {1., 2.}


@pytest.mark.unit
def test_CMA_SO_constructor_defaults():
    """deapext.CMA_SO: constructor computes offspring size and max ngen"""

    strategy = bluepyopt.deapext.CMA_SO.CMA_SO(
        centroids=None,
        offspring_size=5,
        sigma=0.4,
        max_ngen=0,
        IndCreator=ind_creator,
        RandIndCreator=rand_ind_creator)

    assert strategy.lambda_ == 5
    assert strategy.problem_size == 2
    assert strategy.stopping_conditions[0].max_ngen == pytest.approx(
        100 + 50 * 5 ** 2 / numpy.sqrt(5))
