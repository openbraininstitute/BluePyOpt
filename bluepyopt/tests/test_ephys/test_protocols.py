"""bluepyopt.ephys.simulators tests"""

"""
Copyright (c) 2016-2020, EPFL/Blue Brain Project

 This file is part of BluePyOpt <https://github.com/BlueBrain/BluePyOpt>

 This library is free software; you can redistribute it and/or modify it under
 the terms of the GNU Lesser General Public License version 3.0 as published
 by the Free Software Foundation.

 This library is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 details.

 You should have received a copy of the GNU Lesser General Public License
 along with this library; if not, write to the Free Software Foundation, Inc.,
 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

# pylint:disable=W0612


import time
from types import SimpleNamespace
from unittest import mock

import pytest

import bluepyopt.ephys as ephys
from .testmodels import dummycells


@pytest.mark.unit
def test_distloc_exception():
    """ephys.protocols: test if protocol raise dist loc exception"""

    nrn_sim = ephys.simulators.NrnSimulator()
    dummy_cell = dummycells.DummyCellModel1()
    # icell = dummy_cell.instantiate(sim=nrn_sim)
    soma_loc = ephys.locations.NrnSeclistCompLocation(
        name='soma_loc',
        seclist_name='somatic',
        sec_index=0,
        comp_x=.5)
    dend_loc = ephys.locations.NrnSomaDistanceCompLocation(
        name='dend_loc',
        soma_distance=800,
        seclist_name='apical')

    rec_soma = ephys.recordings.CompRecording(
        name='soma.v',
        location=soma_loc,
        variable='v')
    rec_dend = ephys.recordings.CompRecording(
        name='dend.v',
        location=dend_loc,
        variable='v')

    stim = ephys.stimuli.NrnSquarePulse(
        step_amplitude=0.0,
        step_delay=0.0,
        step_duration=50,
        total_duration=50,
        location=soma_loc)

    protocol = ephys.protocols.SweepProtocol(
        name='prot',
        stimuli=[stim],
        recordings=[
            rec_soma,
            rec_dend])

    responses = protocol.run(
        cell_model=dummy_cell,
        param_values={},
        sim=nrn_sim)

    assert responses['soma.v'] is not None
    assert responses['dend.v'] is None

    protocol.destroy(sim=nrn_sim)
    dummy_cell.destroy(sim=nrn_sim)


def run_RuntimeError(
        self,
        tstop=None,
        dt=None,
        cvode_active=None,
        random123_globalindex=None):
    """Mock version of run that throws runtimeerror"""
    raise RuntimeError()


def run_NrnSimulatorException(
        self,
        tstop=None,
        dt=None,
        cvode_active=None,
        random123_globalindex=None):
    """Mock version of run that throws runtimeerror"""
    raise ephys.simulators.NrnSimulatorException('mock', None)


@pytest.mark.unit
def test_sweepprotocol_init():
    """ephys.protocols: Test SweepProtocol init"""

    nrn_sim = ephys.simulators.NrnSimulator()
    dummy_cell = dummycells.DummyCellModel1()
    # icell = dummy_cell.instantiate(sim=nrn_sim)
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
        step_amplitude=0.0,
        step_delay=0.0,
        step_duration=50,
        total_duration=50,
        location=soma_loc)

    protocol = ephys.protocols.SweepProtocol(
        name='prot',
        stimuli=[stim],
        recordings=[rec_soma])

    assert isinstance(protocol, ephys.protocols.SweepProtocol)
    assert protocol.total_duration == 50
    assert (
        protocol.subprotocols() == {'prot': protocol})

    assert 'somatic[0](0.5)' in str(protocol)

    protocol.destroy(sim=nrn_sim)
    dummy_cell.destroy(sim=nrn_sim)


@pytest.mark.unit
def test_sequenceprotocol_init():
    """ephys.protocols: Test SequenceProtocol init"""

    nrn_sim = ephys.simulators.NrnSimulator()
    dummy_cell = dummycells.DummyCellModel1()
    # icell = dummy_cell.instantiate(sim=nrn_sim)
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
        step_amplitude=0.0,
        step_delay=0.0,
        step_duration=50,
        total_duration=50,
        location=soma_loc)

    sweep_protocol = ephys.protocols.SweepProtocol(
        name='sweep_prot',
        stimuli=[stim],
        recordings=[rec_soma])

    seq_protocol = ephys.protocols.SequenceProtocol(
        name='seq_prot',
        protocols=[sweep_protocol])

    assert isinstance(seq_protocol, ephys.protocols.SequenceProtocol)
    assert (
        seq_protocol.subprotocols() == {
            'seq_prot': seq_protocol, 'sweep_prot': sweep_protocol})

    sweep_protocol.destroy(sim=nrn_sim)
    dummy_cell.destroy(sim=nrn_sim)


@pytest.mark.unit
def test_sequenceprotocol_run():
    """ephys.protocols: Test SequenceProtocol run"""

    nrn_sim = ephys.simulators.NrnSimulator()
    dummy_cell = dummycells.DummyCellModel1()
    # icell = dummy_cell.instantiate(sim=nrn_sim)
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
        step_amplitude=0.0,
        step_delay=0.0,
        step_duration=50,
        total_duration=50,
        location=soma_loc)

    sweep_protocol = ephys.protocols.SweepProtocol(
        name='sweep_prot',
        stimuli=[stim],
        recordings=[rec_soma])

    seq_protocol = ephys.protocols.SequenceProtocol(
        name='seq_prot',
        protocols=[sweep_protocol])

    responses = seq_protocol.run(
        cell_model=dummy_cell,
        param_values={},
        sim=nrn_sim)

    assert responses is not None

    sweep_protocol.destroy(sim=nrn_sim)
    dummy_cell.destroy(sim=nrn_sim)


@pytest.mark.unit
def test_sequenceprotocol_overwrite():
    """ephys.protocols: Test SequenceProtocol overwriting keys"""

    nrn_sim = ephys.simulators.NrnSimulator()
    dummy_cell = dummycells.DummyCellModel1()

    sweep_protocols = []
    for x in [.2, .5]:
        soma_loc = ephys.locations.NrnSeclistCompLocation(
            name='soma_loc',
            seclist_name='somatic',
            sec_index=0,
            comp_x=x)

        rec_soma = ephys.recordings.CompRecording(
            name='soma.v',
            location=soma_loc,
            variable='v')

        stim = ephys.stimuli.NrnSquarePulse(
            step_amplitude=0.0,
            step_delay=0.0,
            step_duration=50,
            total_duration=50,
            location=soma_loc)

        sweep_protocols.append(ephys.protocols.SweepProtocol(
            name='sweep_prot',
            stimuli=[stim],
            recordings=[rec_soma]))

    seq_protocol = ephys.protocols.SequenceProtocol(
        name='seq_prot',
        protocols=sweep_protocols)

    pytest.raises(Exception, seq_protocol.run,
                  cell_model=dummy_cell,
                  param_values={},
                  sim=nrn_sim)

    for sweep_protocol in sweep_protocols:
        sweep_protocol.destroy(sim=nrn_sim)
    dummy_cell.destroy(sim=nrn_sim)


@pytest.mark.unit
def test_stepprotocol_init():
    """ephys.protocols: Test StepProtocol init"""

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
        step_amplitude=0.0,
        step_delay=5.0,
        step_duration=50,
        total_duration=50,
        location=soma_loc)
    hold_stim = ephys.stimuli.NrnSquarePulse(
        step_amplitude=0.0,
        step_delay=0.0,
        step_duration=50,
        total_duration=50,
        location=soma_loc)

    step_protocol = ephys.protocols.StepProtocol(
        name='step_prot',
        step_stimulus=stim,
        holding_stimulus=hold_stim,
        recordings=[rec_soma])

    assert step_protocol.step_delay == 5.0
    assert step_protocol.step_duration == 50


@pytest.mark.unit
def test_sweepprotocol_run_unisolated():
    """ephys.protocols: Test SweepProtocol unisolated run"""

    nrn_sim = ephys.simulators.NrnSimulator()
    dummy_cell = dummycells.DummyCellModel1()
    # icell = dummy_cell.instantiate(sim=nrn_sim)
    soma_loc = ephys.locations.NrnSeclistCompLocation(
        name='soma_loc',
        seclist_name='somatic',
        sec_index=0,
        comp_x=.5)
    unknown_loc = ephys.locations.NrnSomaDistanceCompLocation(
        name='unknown_loc',
        seclist_name='somatic',
        soma_distance=100)

    rec_soma = ephys.recordings.CompRecording(
        name='soma.v',
        location=soma_loc,
        variable='v')
    rec_unknown = ephys.recordings.CompRecording(
        name='unknown.v',
        location=unknown_loc,
        variable='v')

    stim = ephys.stimuli.NrnSquarePulse(
        step_amplitude=0.0,
        step_delay=0.0,
        step_duration=50,
        total_duration=50,
        location=soma_loc)

    protocol = ephys.protocols.SweepProtocol(
        name='prot',
        stimuli=[stim],
        recordings=[rec_soma, rec_unknown])

    responses = protocol.run(
        cell_model=dummy_cell,
        param_values={},
        sim=nrn_sim,
        isolate=False)

    assert 'soma.v' in responses
    assert 'unknown.v' in responses
    assert responses['unknown.v'] is None

    protocol.destroy(sim=nrn_sim)
    dummy_cell.destroy(sim=nrn_sim)


@pytest.mark.unit
def test_sweepprotocol_run_isolated():
    """ephys.protocols: Test SweepProtocol isolated run"""

    nrn_sim = ephys.simulators.NrnSimulator(dt=0.1)
    dummy_cell = dummycells.DummyCellModel1()
    soma_loc = ephys.locations.NrnSeclistCompLocation(
        name='soma_loc',
        seclist_name='somatic',
        sec_index=0,
        comp_x=.5)
    unknown_loc = ephys.locations.NrnSomaDistanceCompLocation(
        name='unknown_loc',
        seclist_name='somatic',
        soma_distance=100)

    rec_soma = ephys.recordings.CompRecording(
        name='soma.v',
        location=soma_loc,
        variable='v')
    rec_unknown = ephys.recordings.CompRecording(
        name='unknown.v',
        location=unknown_loc,
        variable='v')

    stim = ephys.stimuli.NrnSquarePulse(
        step_amplitude=0.0,
        step_delay=0.0,
        step_duration=50,
        total_duration=50,
        location=soma_loc)

    protocol = ephys.protocols.SweepProtocol(
        name='prot',
        stimuli=[stim],
        recordings=[rec_soma, rec_unknown])

    responses = protocol.run(
        cell_model=dummy_cell,
        param_values={},
        sim=nrn_sim,
        isolate=True)

    assert 'soma.v' in responses
    assert 'unknown.v' in responses
    assert responses['unknown.v'] is None

    protocol.destroy(sim=nrn_sim)
    dummy_cell.destroy(sim=nrn_sim)


@pytest.mark.unit
def test_nrnsimulator_exception():
    """ephys.protocols: test if protocol raise nrn sim exception"""

    nrn_sim = ephys.simulators.NrnSimulator()
    dummy_cell = dummycells.DummyCellModel1()
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
        step_amplitude=0.0,
        step_delay=0.0,
        step_duration=50,
        total_duration=50,
        location=soma_loc)

    protocol = ephys.protocols.SweepProtocol(
        name='prot',
        stimuli=[stim],
        recordings=[rec_soma])

    nrn_sim.run = run_RuntimeError

    responses = protocol.run(
        cell_model=dummy_cell,
        param_values={},
        sim=nrn_sim,
        isolate=False)

    assert responses['soma.v'] is None

    nrn_sim.run = run_NrnSimulatorException

    responses = protocol.run(
        cell_model=dummy_cell,
        param_values={},
        sim=nrn_sim,
        isolate=False)

    assert responses['soma.v'] is None

    protocol.destroy(sim=nrn_sim)
    dummy_cell.destroy(sim=nrn_sim)


@pytest.mark.unit
def test_sweepprotocol_instantiate_with_LFPyCellModel():
    """ephys.protocols: Test SweepProtocol instantiate with LFPyCellModel"""
    import LFPy

    dummy_cell = dummycells.DummyLFPyCellModel1()
    nrn_sim = ephys.simulators.LFPySimulator()

    soma_loc = ephys.locations.NrnSeclistCompLocation(
        name='soma_loc',
        seclist_name='somatic',
        sec_index=0,
        comp_x=.5
    )

    rec = ephys.recordings.LFPRecording(
        name='lfpy_rec'
    )

    stim = ephys.stimuli.LFPySquarePulse(
        step_amplitude=0.1,
        step_delay=70,
        step_duration=100,
        location=soma_loc,
        total_duration=300
    )

    protocol = ephys.protocols.SweepProtocol(
        name='prot',
        stimuli=[stim],
        recordings=[rec])

    dummy_cell.instantiate(sim=nrn_sim)
    protocol.instantiate(sim=nrn_sim, cell_model=dummy_cell)

    # check that recording and stimuli have been instantiated with LFPy
    assert rec.instantiated
    assert isinstance(rec.cell, LFPy.Cell)
    assert isinstance(stim.iclamp, LFPy.StimIntElectrode)

    protocol.destroy(sim=nrn_sim)
    dummy_cell.destroy(sim=nrn_sim)


class LegacyProtocol(ephys.protocols.Protocol):

    """Protocol whose run method predates the timeout argument"""

    def __init__(self, name, response):
        super(LegacyProtocol, self).__init__(name)
        self.response = response

    def run(self, cell_model, param_values, sim=None, isolate=None):
        return self.response

    def subprotocols(self):
        return {self.name: self}


class BrokenProtocol(LegacyProtocol):

    """Protocol that fails with an unrelated TypeError"""

    def run(self, cell_model, param_values, sim=None, isolate=None,
            timeout=None):
        raise TypeError("something else went wrong")


@pytest.mark.unit
def test_sequenceprotocol_supports_protocols_without_timeout():
    """ephys.protocols: SequenceProtocol retries without timeout"""
    sequence = ephys.protocols.SequenceProtocol(
        'seq', protocols=[LegacyProtocol('legacy', {'a': 1})])

    responses = sequence.run(None, {}, sim=None, isolate=False, timeout=5)

    assert responses == {'a': 1}


@pytest.mark.unit
def test_sequenceprotocol_reraises_other_type_errors():
    """ephys.protocols: SequenceProtocol keeps unrelated TypeErrors"""
    sequence = ephys.protocols.SequenceProtocol(
        'seq', protocols=[BrokenProtocol('broken', {})])

    with pytest.raises(TypeError, match="something else"):
        sequence.run(None, {}, sim=None, isolate=False, timeout=5)


@pytest.mark.unit
def test_sequenceprotocol_str():
    """ephys.protocols: SequenceProtocol string representation"""
    legacy = LegacyProtocol('legacy', {})
    sequence = ephys.protocols.SequenceProtocol('seq', protocols=[legacy])

    assert str(sequence) == (
        'Sequence protocol seq:\n1 subprotocols:\n%s\n' % str(legacy))


def make_sweep(name='prot', **kwargs):
    """Sweep protocol recording the soma of a dummy cell"""
    soma_loc = ephys.locations.NrnSeclistCompLocation(
        name='soma_loc', seclist_name='somatic', sec_index=0, comp_x=.5)
    stim = ephys.stimuli.NrnSquarePulse(
        step_amplitude=0.0, step_delay=0.0, step_duration=5,
        total_duration=5, location=soma_loc)
    rec = ephys.recordings.CompRecording(
        name='soma.v', location=soma_loc, variable='v')
    return ephys.protocols.SweepProtocol(
        name, stimuli=[stim], recordings=[rec], **kwargs)


class StochasticCell(dummycells.DummyCellModel1):

    """Dummy cell recording the mechanism state seen during the run"""

    def __init__(self):
        super(StochasticCell, self).__init__('stochastic')
        self.mechanisms = [SimpleNamespace(deterministic=False)]
        self.seen = None

    def instantiate(self, sim=None):
        self.seen = self.mechanisms[0].deterministic
        return super(StochasticCell, self).instantiate(sim=sim)


class SlowCell(dummycells.DummyCellModel1):

    """Dummy cell that never finishes freezing"""

    def freeze(self, param_values):
        time.sleep(30)


@pytest.mark.unit
def test_sweepprotocol_deterministic_restores_mechanism_state():
    """ephys.protocols: deterministic only applies during the run"""
    nrn_sim = ephys.simulators.NrnSimulator()
    cell = StochasticCell()

    make_sweep(deterministic=True).run(cell, {}, sim=nrn_sim, isolate=False)

    assert cell.seen is True
    assert cell.mechanisms[0].deterministic is False


@pytest.mark.unit
def test_sweepprotocol_passes_lfpy_objects_to_the_lfpy_simulator():
    """ephys.protocols: SweepProtocol runs LFPySimulator with its cell"""
    protocol = make_sweep()
    protocol.instantiate = mock.MagicMock()
    protocol.destroy = mock.MagicMock()
    sim = mock.MagicMock(spec=ephys.simulators.LFPySimulator)
    cell = mock.MagicMock()

    protocol._run_func(cell, {}, sim=sim)

    sim.run.assert_called_once_with(
        lfpy_cell=cell.lfpy_cell,
        lfpy_electrode=cell.lfpy_electrode,
        tstop=protocol.total_duration,
        cvode_active=protocol.cvode_active)


@pytest.mark.unit
def test_sweepprotocol_wraps_failures():
    """ephys.protocols: SweepProtocol reports its own exception"""
    cell = mock.MagicMock()
    cell.freeze.side_effect = KeyError('missing')

    with pytest.raises(ephys.protocols.SweepProtocolException):
        make_sweep()._run_func(cell, {}, sim=mock.MagicMock())


@pytest.mark.unit
def test_sweepprotocol_rejects_negative_timeout():
    """ephys.protocols: SweepProtocol refuses a negative timeout"""
    nrn_sim = ephys.simulators.NrnSimulator()

    with pytest.raises(ValueError, match="timeout should be > 0"):
        make_sweep().run(dummycells.DummyCellModel1(), {}, sim=nrn_sim,
                         isolate=True, timeout=-1)


@pytest.mark.unit
def test_sweepprotocol_returns_none_on_timeout():
    """ephys.protocols: a timed out SweepProtocol returns None responses"""
    nrn_sim = ephys.simulators.NrnSimulator()

    responses = make_sweep().run(SlowCell(), {}, sim=nrn_sim, isolate=True,
                                 timeout=0.2)

    assert responses == {'soma.v': None}
