"""bluepyopt.stoppingCriteria tests"""


import numpy

import bluepyopt.stoppingCriteria
import bluepyopt.deapext.stoppingCriteria

import pytest


class FakeFitness(object):
    """Minimal fitness exposing the reduce attribute"""

    def __init__(self, reduce_value):
        self.reduce = reduce_value
        self.values = (reduce_value,)


class FakeIndividual(object):
    """Minimal individual exposing a fitness"""

    def __init__(self, reduce_value):
        self.fitness = FakeFitness(reduce_value)


def make_population(*reduce_values):
    """Build a population of fake individuals"""
    return [FakeIndividual(value) for value in reduce_values]


@pytest.mark.unit
def test_base_StoppingCriteria():
    """stoppingCriteria: Testing base class"""

    criteria = bluepyopt.stoppingCriteria.StoppingCriteria()

    assert criteria.criteria_met is False
    criteria.check({})
    assert criteria.criteria_met is False
    criteria.criteria_met = True
    criteria.reset()
    assert criteria.criteria_met is False


@pytest.mark.unit
def test_MaxNGen():
    """deapext.stoppingCriteria: Testing MaxNGen"""

    max_gen = 3
    criteria = bluepyopt.deapext.stoppingCriteria.MaxNGen(max_gen)

    assert criteria.criteria_met is False
    criteria.check({"gen": max_gen + 1})
    assert criteria.criteria_met is True
    criteria.reset()
    criteria.check({"gen": max_gen})
    assert criteria.criteria_met is False


@pytest.mark.unit
def test_Stagnation():
    """deapext.stoppingCriteria: Testing Stagnation"""

    criteria = bluepyopt.deapext.stoppingCriteria.Stagnation(
        lambda_=4, problem_size=2)
    population = make_population(1.0, 2.0, 3.0, 4.0)

    # history too short -> criterion not met
    criteria.check({"gen": 1, "population": population})
    assert criteria.criteria_met is False
    assert len(criteria.best) == 1
    assert len(criteria.median) == 1

    # long stagnant history -> criterion met
    criteria.best = [1.0] * 200
    criteria.median = [1.0] * 200
    criteria.check({"gen": 1, "population": population})
    assert criteria.criteria_met is True

    criteria.reset()
    assert criteria.criteria_met is False


@pytest.mark.unit
@pytest.mark.parametrize("best, median", [
    (numpy.linspace(2.0, 1.0, 200), numpy.linspace(2.0, 1.0, 200)),
    (numpy.linspace(2.0, 1.0, 200), [1.0] * 200),
    ([1.0] * 200, numpy.linspace(2.0, 1.0, 200)),
])
def test_Stagnation_not_met_while_improving(best, median):
    """deapext.stoppingCriteria: Stagnation needs best and median to stall"""

    criteria = bluepyopt.deapext.stoppingCriteria.Stagnation(
        lambda_=4, problem_size=2)
    criteria.best = list(best)
    criteria.median = list(median)

    criteria.check(
        {"gen": 1, "population": make_population(1.0, 2.0, 3.0, 4.0)})

    assert criteria.criteria_met is False


@pytest.mark.unit
@pytest.mark.parametrize("best", [
    # improving by more than the threshold, slowly enough not to look noisy
    numpy.linspace(1.1, 1.0, 200),
    [1.0] * 180 + [0.5, 1.5] * 10,
])
def test_Stagnationv2_not_met_while_improving_or_noisy(best):
    """deapext.stoppingCriteria: Stagnationv2 needs a flat, quiet history"""

    criteria = bluepyopt.deapext.stoppingCriteria.Stagnationv2(
        lambda_=4, problem_size=2)
    criteria.best = list(best)

    criteria.check(
        {"gen": 1, "population": make_population(1.0, 2.0, 3.0, 4.0)})

    assert criteria.criteria_met is False


@pytest.mark.unit
def test_Stagnationv2():
    """deapext.stoppingCriteria: Testing Stagnationv2"""

    criteria = bluepyopt.deapext.stoppingCriteria.Stagnationv2(
        lambda_=4, problem_size=2)
    population = make_population(1.0, 2.0, 3.0, 4.0)

    criteria.check({"gen": 1, "population": population})
    assert criteria.criteria_met is False
    assert len(criteria.best) == 1

    # long non-improving history -> criterion met
    criteria.best = [1.0] * 200
    criteria.check({"gen": 1, "population": population})
    assert criteria.criteria_met is True

    criteria.reset()
    assert criteria.criteria_met is False


@pytest.mark.unit
def test_TolHistFun():
    """deapext.stoppingCriteria: Testing TolHistFun"""

    criteria = bluepyopt.deapext.stoppingCriteria.TolHistFun(
        lambda_=30, problem_size=1)
    population = make_population(5.0)

    for _ in range(criteria.mins.maxlen - 1):
        criteria.check({"population": population})
    assert criteria.criteria_met is False

    criteria.check({"population": population})
    assert criteria.criteria_met is True

    criteria.reset()
    # varying min fitness -> range above tolerance
    for i in range(criteria.mins.maxlen):
        criteria.check({"population": make_population(float(i))})
    assert criteria.criteria_met is False


@pytest.mark.unit
def test_EqualFunVals():
    """deapext.stoppingCriteria: Testing EqualFunVals"""

    criteria = bluepyopt.deapext.stoppingCriteria.EqualFunVals(
        lambda_=4, problem_size=3)
    # best and k-th best are equal -> every generation counts
    population = make_population(1.0, 1.0, 1.0, 2.0)

    for gen in range(1, criteria.problem_size + 1):
        criteria.check({"gen": gen, "population": population})
    assert criteria.criteria_met is False

    criteria.check(
        {"gen": criteria.problem_size + 1, "population": population})
    assert criteria.criteria_met is True

    criteria.reset()
    criteria.equalvalues = []
    for gen in range(1, criteria.problem_size + 2):
        criteria.check(
            {"gen": gen, "population": make_population(1.0, 2.0, 3.0, 4.0)})
    assert criteria.criteria_met is False


@pytest.mark.unit
def test_TolX():
    """deapext.stoppingCriteria: Testing TolX"""

    criteria = bluepyopt.deapext.stoppingCriteria.TolX()

    criteria.check(
        {"pc": numpy.zeros(2), "C": numpy.zeros((2, 2))})
    assert criteria.criteria_met is True

    criteria.reset()
    criteria.check({"pc": numpy.ones(2), "C": numpy.eye(2)})
    assert criteria.criteria_met is False


@pytest.mark.unit
def test_TolUpSigma():
    """deapext.stoppingCriteria: Testing TolUpSigma"""

    criteria = bluepyopt.deapext.stoppingCriteria.TolUpSigma(sigma0=1.0)

    criteria.check(
        {"sigma": 1e30, "diagD": numpy.array([1.0, 1.0])})
    assert criteria.criteria_met is True

    criteria.reset()
    criteria.check(
        {"sigma": 1.0, "diagD": numpy.array([1.0, 1.0])})
    assert criteria.criteria_met is False


@pytest.mark.unit
def test_ConditionCov():
    """deapext.stoppingCriteria: Testing ConditionCov"""

    criteria = bluepyopt.deapext.stoppingCriteria.ConditionCov()

    criteria.check({"cond": 1e15})
    assert criteria.criteria_met is True

    criteria.reset()
    criteria.check({"cond": 1.0})
    assert criteria.criteria_met is False


@pytest.mark.unit
def test_NoEffectAxis():
    """deapext.stoppingCriteria: Testing NoEffectAxis"""

    criteria = bluepyopt.deapext.stoppingCriteria.NoEffectAxis(
        problem_size=2)
    kwargs = {
        "gen": 0,
        "centroid": numpy.array([1.0, 2.0]),
        "sigma": 0.0,
        "diagD": numpy.ones(2),
        "B": numpy.ones((2, 2)),
    }

    criteria.check(kwargs)
    assert criteria.criteria_met is True

    criteria.reset()
    kwargs["sigma"] = 1.0
    criteria.check(kwargs)
    assert criteria.criteria_met is False


@pytest.mark.unit
def test_NoEffectCoor():
    """deapext.stoppingCriteria: Testing NoEffectCoor"""

    criteria = bluepyopt.deapext.stoppingCriteria.NoEffectCoor()

    criteria.check({
        "centroid": numpy.array([1.0, 2.0]),
        "sigma": 0.0,
        "C": numpy.eye(2),
    })
    assert criteria.criteria_met is True

    criteria.reset()
    criteria.check({
        "centroid": numpy.array([1.0, 2.0]),
        "sigma": 1.0,
        "C": numpy.eye(2),
    })
    assert criteria.criteria_met is False
