"""Test ephys model objects"""

# pylint: disable=R0914

import os


from unittest import mock

import pytest
import numpy

from bluepyopt import ephys

TESTDATA_DIR = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)),
    'testdata')

simple_morphology_path = os.path.join(TESTDATA_DIR, 'simple.swc')


@pytest.mark.unit
def test_CellEvaluator_init():
    """ephys.evaluators: Test CellEvaluator init"""
    sim = ephys.simulators.NrnSimulator()

    model = ephys.models.CellModel('test_model', params=[])

    fitness_calc = ephys.objectivescalculators.ObjectivesCalculator()

    evaluator = ephys.evaluators.CellEvaluator(
        cell_model=model,
        param_names=[],
        fitness_calculator=fitness_calc,
        sim=sim)

    assert isinstance(evaluator, ephys.evaluators.CellEvaluator)
    assert (
        str(evaluator) ==
        'cell evaluator:\n  cell model:\n    test_model:\n  morphology:\n  '
        'mechanisms:\n  params:\n\n  fitness protocols:\n  '
        'fitness calculator:\n    objectives:\n\n')


@pytest.mark.unit
def test_CellEvaluator_evaluate():
    """ephys.evaluators: Test CellEvaluator evaluate"""
    sim = ephys.simulators.NrnSimulator()

    simple_morph = ephys.morphologies.NrnFileMorphology(
        simple_morphology_path,
        do_replace_axon=True)

    all_loc = ephys.locations.NrnSeclistLocation('all', 'all')

    cm = ephys.parameters.NrnRangeParameter(
        name='cm',
        param_name='cm',
        bounds=[.5, 1.5],
        locations=[all_loc])

    cell_model = ephys.models.CellModel('CellModel',
                                        morph=simple_morph,
                                        mechs=[],
                                        params=[cm])

    soma_loc = ephys.locations.NrnSeclistCompLocation(
        name='soma_loc',
        seclist_name='somatic',
        sec_index=0,
        comp_x=.5)

    rec_soma = ephys.recordings.CompRecording(
        name='soma.v',
        location=soma_loc,
        variable='v')

    stim = ephys.stimuli.NrnSquarePulse(
        step_amplitude=0.1,
        step_delay=100.0,
        step_duration=100,
        total_duration=200,
        location=soma_loc)

    protocol = ephys.protocols.SweepProtocol(
        name='prot',
        stimuli=[stim],
        recordings=[rec_soma])

    mean = -65

    efeature = ephys.efeatures.eFELFeature(name='test_eFELFeature',
                                           efel_feature_name='voltage_base',
                                           recording_names={'': 'soma.v'},
                                           stim_start=100.0,
                                           stim_end=200.0,
                                           exp_mean=-65,
                                           exp_std=1)

    s_obj = ephys.objectives.SingletonObjective(
        'singleton',
        feature=efeature)

    fitness_calc = ephys.objectivescalculators.ObjectivesCalculator(
        objectives=[s_obj])

    evaluator = ephys.evaluators.CellEvaluator(
        cell_model=cell_model,
        param_names=['cm'],
        fitness_calculator=fitness_calc,
        fitness_protocols={'sweep': protocol},
        sim=sim)

    responses = protocol.run(cell_model, {'cm': 1.0}, sim=sim)

    feature_value = efeature.calculate_feature(responses)
    feature_value_eva = evaluator.evaluate_with_dicts(
        {'cm': 1.0}, target='values'
    )

    score = evaluator.evaluate([1.0])
    expected_score = abs(mean - feature_value)

    numpy.testing.assert_almost_equal(score, expected_score)

    score_dict = evaluator.objective_dict(score)

    numpy.testing.assert_almost_equal(score_dict['singleton'], expected_score)
    numpy.testing.assert_almost_equal(
        feature_value, feature_value_eva['singleton']
    )


def bare_evaluator(**kwargs):
    """Evaluator around a mock cell model"""
    arguments = dict(
        cell_model=mock.MagicMock(),
        param_names=[],
        fitness_protocols={},
        fitness_calculator=mock.MagicMock(objectives=[]),
        sim=mock.MagicMock(),
    )
    arguments.update(kwargs)
    return ephys.evaluators.CellEvaluator(**arguments)


@pytest.mark.unit
def test_CellEvaluator_requires_a_simulator():
    """ephys.evaluators: CellEvaluator refuses a missing simulator"""
    with pytest.raises(ValueError, match="provide a Simulator"):
        bare_evaluator(sim=None)


@pytest.mark.unit
def test_CellEvaluator_checks_the_number_of_objectives():
    """ephys.evaluators: objective_dict needs one value per objective"""
    with pytest.raises(Exception, match="wrong number of objectives"):
        bare_evaluator().objective_dict([1.0])


@pytest.mark.unit
def test_CellEvaluator_rejects_unknown_targets():
    """ephys.evaluators: evaluate_with_dicts validates the target"""
    with pytest.raises(Exception, match='has to be "scores" or "values"'):
        bare_evaluator().evaluate_with_dicts({}, target='other')


@pytest.mark.unit
def test_CellEvaluator_requires_a_fitness_calculator():
    """ephys.evaluators: evaluate_with_dicts needs a fitness calculator"""
    evaluator = bare_evaluator()
    evaluator.fitness_calculator = None

    with pytest.raises(Exception, match="need fitness_calculator"):
        evaluator.evaluate_with_dicts({})


class LegacyProtocol(object):

    """Protocol whose run method predates the timeout argument"""

    def run(self, cell_model, param_values, sim=None, isolate=None):
        return {'cell': cell_model, 'isolate': isolate}


@pytest.mark.unit
def test_CellEvaluator_supports_protocols_without_timeout():
    """ephys.evaluators: run_protocol retries without timeout"""
    evaluator = bare_evaluator()

    response = evaluator.run_protocol(LegacyProtocol(), {}, isolate=False)

    assert response == {'cell': evaluator.cell_model, 'isolate': False}


@pytest.mark.unit
def test_CellEvaluator_reraises_other_type_errors():
    """ephys.evaluators: run_protocol keeps unrelated TypeErrors"""
    protocol = mock.MagicMock()
    protocol.run.side_effect = TypeError("something else went wrong")

    with pytest.raises(TypeError, match="something else"):
        bare_evaluator().run_protocol(protocol, {})


@pytest.mark.unit
def test_CellEvaluator_str_lists_the_protocols():
    """ephys.evaluators: the string representation shows each protocol"""
    protocol = mock.MagicMock()
    protocol.__str__.return_value = 'the protocol'

    text = str(bare_evaluator(fitness_protocols={'p': protocol}))

    assert 'the protocol' in text
