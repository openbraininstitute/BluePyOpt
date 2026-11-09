"""bluepyopt.optimisationsCMA tests"""

import os
import sys
import types

import pytest
import bluepyopt
import bluepyopt.ephys.examples.simplecell


def sentinel_map(function, *iterables):
    """Map function that can be told apart from the DEAP toolbox default"""
    return map(function, *iterables)


@pytest.fixture
def fake_scoop(monkeypatch):
    """Fake the scoop module, which is not part of the test dependencies"""
    futures = types.ModuleType("scoop.futures")
    futures.map = sentinel_map
    scoop = types.ModuleType("scoop")
    scoop.futures = futures
    monkeypatch.setitem(sys.modules, "scoop", scoop)
    monkeypatch.setitem(sys.modules, "scoop.futures", futures)
    return futures


class FailingMap(object):

    """Map function failing after a number of calls, like an interrupted run"""

    def __init__(self, allowed_calls):
        self.allowed_calls = allowed_calls

    def __call__(self, function, *iterables):
        if self.allowed_calls == 0:
            raise RuntimeError("interrupted")
        self.allowed_calls -= 1
        return map(function, *iterables)


class CountingMap(object):

    """Map function counting the generations it evaluates"""

    def __init__(self):
        self.calls = 0

    def __call__(self, function, *iterables):
        self.calls += 1
        return map(function, *iterables)


@pytest.mark.unit
def test_optimisationsCMA_normspace():
    """deapext.optimisationsCMA: Testing optimisationsCMA normspace"""

    simplecell = bluepyopt.ephys.examples.simplecell.SimpleCell()
    evaluator = simplecell.cell_evaluator
    optimisation = bluepyopt.deapext.optimisationsCMA.DEAPOptimisationCMA(
        evaluator=evaluator)

    x = [n * 0.1 for n in range(len(evaluator.params))]
    y = [f2(f1(p)) for p, f1, f2 in zip(x, optimisation.to_norm,
                                        optimisation.to_space)]

    for a, b in zip(x, y):
        assert b == pytest.approx(a, abs=1e-5)


@pytest.mark.unit
def test_optimisationsCMA_SO_run():
    """deapext.optimisationsCMA: Testing optimisationsCMA run from centroid"""

    simplecell = bluepyopt.ephys.examples.simplecell.SimpleCell()
    evaluator = simplecell.cell_evaluator
    x = [n * 0.1 for n in range(len(evaluator.params))]

    optimiser = bluepyopt.deapext.optimisationsCMA.DEAPOptimisationCMA
    optimisation = optimiser(evaluator=evaluator, centroids=[x])
    pop, hof, log, hist = optimisation.run(max_ngen=2)

    assert log.select("avg")[-1] == pytest.approx(53.3333, abs=1e-4)
    assert log.select("std")[-1] == pytest.approx(83.7987, abs=1e-4)
    assert pop[0][0] == pytest.approx(0.10525059698894745, abs=1e-6)
    assert pop[0][1] == pytest.approx(0.01000000003249999, abs=1e-6)


@pytest.mark.unit
def test_optimisationsCMA_MO_run():
    """deapext.optimisationsCMA: Testing optimisationsCMA run from centroid"""

    simplecell = bluepyopt.ephys.examples.simplecell.SimpleCell()
    evaluator = simplecell.cell_evaluator

    optimiser = bluepyopt.deapext.optimisationsCMA.DEAPOptimisationCMA
    optimisation = optimiser(
        selector_name="multi_objective",
        offspring_size=3,
        evaluator=evaluator,
        seed=42
    )
    pop, hof, log, hist = optimisation.run(max_ngen=2)

    assert log.select("avg")[-1] == pytest.approx(40., abs=1e-4)
    assert log.select("std")[-1] == pytest.approx(16.32993, abs=1e-4)
    assert pop[0][0] == pytest.approx(0.09601241274168831, abs=1e-6)
    assert pop[0][1] == pytest.approx(0.024646650865379722, abs=1e-6)


@pytest.mark.unit
def test_optimisationsCMA_bad_weight_hv():
    """deapext.optimisationsCMA: weight_hv outside [0, 1] raises"""

    simplecell = bluepyopt.ephys.examples.simplecell.SimpleCell()

    pytest.raises(
        Exception,
        bluepyopt.deapext.optimisationsCMA.DEAPOptimisationCMA,
        evaluator=simplecell.cell_evaluator,
        weight_hv=2.0)


@pytest.mark.unit
def test_optimisationsCMA_bad_selector():
    """deapext.optimisationsCMA: unknown selector_name raises"""

    simplecell = bluepyopt.ephys.examples.simplecell.SimpleCell()

    pytest.raises(
        Exception,
        bluepyopt.deapext.optimisationsCMA.DEAPOptimisationCMA,
        evaluator=simplecell.cell_evaluator,
        selector_name="bad")


@pytest.mark.unit
def test_optimisationsCMA_map_function():
    """deapext.optimisationsCMA: provided map_function is registered"""

    simplecell = bluepyopt.ephys.examples.simplecell.SimpleCell()

    optimisation = bluepyopt.deapext.optimisationsCMA.DEAPOptimisationCMA(
        evaluator=simplecell.cell_evaluator, map_function=sentinel_map)

    pop, hof, log, hist = optimisation.run(max_ngen=1)

    assert optimisation.toolbox.map.func is sentinel_map
    assert log.select("gen") == [1, 2]


@pytest.mark.unit
def test_optimisationsCMA_scoop_with_map_function(fake_scoop):
    """deapext.optimisationsCMA: scoop and map_function are exclusive"""

    simplecell = bluepyopt.ephys.examples.simplecell.SimpleCell()

    with pytest.raises(Exception, match="Impossible to use scoop"):
        bluepyopt.deapext.optimisationsCMA.DEAPOptimisationCMA(
            evaluator=simplecell.cell_evaluator,
            use_scoop=True,
            map_function=map)


@pytest.mark.unit
def test_optimisationsCMA_scoop(fake_scoop):
    """deapext.optimisationsCMA: scoop futures registered as map"""

    simplecell = bluepyopt.ephys.examples.simplecell.SimpleCell()

    optimisation = bluepyopt.deapext.optimisationsCMA.DEAPOptimisationCMA(
        evaluator=simplecell.cell_evaluator, use_scoop=True)

    assert optimisation.toolbox.map.func is sentinel_map


@pytest.mark.unit
def test_optimisationsCMA_checkpoint(tmp_path):
    """deapext.optimisationsCMA: checkpoint write and resume"""

    simplecell = bluepyopt.ephys.examples.simplecell.SimpleCell()
    evaluator = simplecell.cell_evaluator
    cp_filename = str(tmp_path / "checkpoint.pkl")

    optimiser = bluepyopt.deapext.optimisationsCMA.DEAPOptimisationCMA
    optimisation = optimiser(evaluator=evaluator, offspring_size=4)
    pop, hof, log, hist = optimisation.run(
        max_ngen=2, cp_frequency=1, cp_filename=cp_filename)

    assert os.path.isfile(cp_filename)
    assert log.select("gen")[-1] == 3

    optimisation = optimiser(evaluator=evaluator, offspring_size=4)
    pop, hof, log, hist = optimisation.run(
        max_ngen=2, cp_frequency=1,
        cp_filename=cp_filename, continue_cp=True)

    # the checkpoint stored an already finished optimisation state,
    # resuming restores the original logbook without extra generations
    assert log.select("gen") == [1, 2, 3]


@pytest.mark.unit
def test_optimisationsCMA_resumes_an_interrupted_run(tmp_path):
    """deapext.optimisationsCMA: an interrupted run continues from its data"""

    simplecell = bluepyopt.ephys.examples.simplecell.SimpleCell()
    evaluator = simplecell.cell_evaluator
    cp_filename = str(tmp_path / "checkpoint.pkl")
    optimiser = bluepyopt.deapext.optimisationsCMA.DEAPOptimisationCMA

    interrupted = optimiser(
        evaluator=evaluator, offspring_size=4, map_function=FailingMap(1))
    with pytest.raises(RuntimeError, match="interrupted"):
        interrupted.run(max_ngen=3, cp_frequency=1, cp_filename=cp_filename)

    counting_map = CountingMap()
    resumed = optimiser(
        evaluator=evaluator, offspring_size=4, map_function=counting_map)
    pop, hof, log, hist = resumed.run(
        max_ngen=3, cp_frequency=1,
        cp_filename=cp_filename, continue_cp=True)

    # the generation stored before the interruption is kept in the logbook
    assert counting_map.calls > 0
    assert len(log) == counting_map.calls + 1
