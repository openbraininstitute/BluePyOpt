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

import os
import pathlib
import types
from unittest import mock

import numpy
import pytest

import bluepyopt.ephys as ephys
import bluepyopt.ephys.examples as examples


@pytest.mark.unit
def test_nrnsimulator_init():
    """ephys.simulators: test if NrnSimulator constructor works"""

    neuron_sim = ephys.simulators.NrnSimulator()
    assert isinstance(neuron_sim, ephys.simulators.NrnSimulator)


@pytest.mark.unit
def test_nrnsimulator_init_windows():
    """ephys.simulators: test if NrnSimulator constructor works on Windows"""

    with mock.patch("platform.system", mock.MagicMock(return_value="Windows")):
        neuron_sim = ephys.simulators.NrnSimulator()
        assert isinstance(neuron_sim, ephys.simulators.NrnSimulator)
        assert not neuron_sim.disable_banner
        assert not neuron_sim.banner_disabled

        neuron_sim.neuron.h.celsius = 34

        assert not neuron_sim.disable_banner
        assert not neuron_sim.banner_disabled


@pytest.mark.unit
def test_nrnsimulator_cvode_minstep():
    """ephys.simulators: test if NrnSimulator constructor works"""

    # Check with minstep specified
    neuron_sim = ephys.simulators.NrnSimulator()
    assert neuron_sim.cvode.minstep() == 0.0
    assert neuron_sim.cvode_minstep == 0.0

    # Check default minstep before and after run
    neuron_sim = ephys.simulators.NrnSimulator(cvode_minstep=0.01)
    assert neuron_sim.cvode.minstep() == 0.0
    neuron_sim.run(tstop=10)
    assert neuron_sim.cvode.minstep() == 0.0

    # Check with that minstep is set back to the original value after run
    neuron_sim = ephys.simulators.NrnSimulator(cvode_minstep=0.0)
    neuron_sim.cvode_minstep = 0.05
    assert neuron_sim.cvode.minstep() == 0.05
    neuron_sim.run(tstop=10)
    assert neuron_sim.cvode.minstep() == 0.05

    # Check that the minstep is effective
    cvode_minstep = 0.012
    params = {
        "gnabar_hh": 0.10299326453483033,
        "gkbar_hh": 0.027124836082684685,
    }
    simplecell = examples.simplecell.SimpleCell()
    evaluator = simplecell.cell_evaluator
    evaluator.cell_model.unfreeze(params.keys())
    evaluator.sim = ephys.simulators.NrnSimulator(cvode_minstep=cvode_minstep)
    responses = evaluator.run_protocols(
        protocols=evaluator.fitness_protocols.values(), param_values=params
    )
    ton = list(evaluator.fitness_protocols.values())[0].stimuli[0].step_delay
    toff = (
        ton
        + list(evaluator.fitness_protocols.values())[0]
        .stimuli[0]
        .step_duration
    )
    t_series = numpy.array(responses["Step1.soma.v"]["time"])
    t_series = t_series[((ton + 1.0) < t_series) & (t_series < (toff - 1.0))]
    min_dt = numpy.min(numpy.ediff1d(t_series))
    assert (min_dt >= cvode_minstep) == 1
    evaluator.cell_model.freeze(params)


@pytest.mark.unit
def test_neuron_import():
    """ephys.simulators: test if bluepyopt.neuron import was successful"""
    from bluepyopt import ephys  # NOQA

    neuron_sim = ephys.simulators.NrnSimulator()
    assert isinstance(neuron_sim.neuron, types.ModuleType)


@pytest.mark.unit
def test_nrnsim_run_dt_exception():
    """ephys.simulators: test if run return exception when dt was changed"""

    from bluepyopt import ephys  # NOQA

    neuron_sim = ephys.simulators.NrnSimulator()
    neuron_sim.neuron.h.dt = 1.0
    pytest.raises(Exception, neuron_sim.run, 10, cvode_active=False)


@pytest.mark.unit
def test_nrnsim_run_cvodeactive_dt_exception():
    """ephys.simulators: test if run return exception cvode and dt both used"""

    from bluepyopt import ephys  # NOQA

    neuron_sim = ephys.simulators.NrnSimulator()
    neuron_sim.neuron.h.dt = 1.0
    pytest.raises(ValueError, neuron_sim.run, 10, dt=0.1, cvode_active=True)


@pytest.mark.unit
@mock.patch.object(pathlib.Path, "glob")
def test_disable_banner_exception(mock_glob):
    """ephys.simulators: test if disable_banner raises exception"""
    mock_glob.return_value = []

    import warnings

    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("always")
        ephys.simulators.NrnSimulator._nrn_disable_banner()
        assert len(warnings_record) == 1


@pytest.mark.unit
def test_lfpysimulator_init():
    """ephys.simulators: test if LFPySimulator constructor works"""

    empty_cell = ephys.models.LFPyCellModel(name="empty_cell")
    neuron_sim = ephys.simulators.LFPySimulator()
    assert isinstance(neuron_sim, ephys.simulators.LFPySimulator)


@pytest.mark.unit
def test_lfpyimulator_init_windows():
    """ephys.simulators: test if LFPySimulator constructor works on Windows"""

    with mock.patch("platform.system", mock.MagicMock(return_value="Windows")):
        empty_cell = ephys.models.LFPyCellModel(name="empty_cell")
        neuron_sim = ephys.simulators.LFPySimulator()
        assert isinstance(neuron_sim, ephys.simulators.LFPySimulator)
        assert not neuron_sim.disable_banner
        assert not neuron_sim.banner_disabled

        neuron_sim.neuron.h.celsius = 34

        assert not neuron_sim.disable_banner
        assert not neuron_sim.banner_disabled


@pytest.mark.unit
def test__lfpysimulator_neuron_import():
    """ephys.simulators: test neuron import from LFPySimulator"""
    from bluepyopt import ephys  # NOQA

    empty_cell = ephys.models.LFPyCellModel(name="empty_cell")
    neuron_sim = ephys.simulators.LFPySimulator()
    assert isinstance(neuron_sim.neuron, types.ModuleType)


@pytest.mark.unit
def test_lfpysim_run_cvodeactive_dt_exception():
    """ephys.simulators: test if LFPySimulator run returns exception"""

    from bluepyopt import ephys  # NOQA

    TESTDATA_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "testdata"
    )
    simple_morphology_path = os.path.join(TESTDATA_DIR, "simple.swc")
    test_morph = ephys.morphologies.NrnFileMorphology(simple_morphology_path)

    lfpy_cell = ephys.models.LFPyCellModel(
        name="lfpy_cell", morph=test_morph, mechs=[]
    )
    neuron_sim = ephys.simulators.LFPySimulator()
    lfpy_cell.instantiate(sim=neuron_sim)

    with pytest.raises(
        ValueError,
        match=(
            "NrnSimulator: "
            "Impossible to combine the dt argument when "
            "cvode_active is True in the NrnSimulator run method"
        ),
    ):
        neuron_sim.run(
            tstop=10,
            dt=0.1,
            cvode_active=True,
            lfpy_cell=lfpy_cell.lfpy_cell,
            lfpy_electrode=lfpy_cell.lfpy_electrode,
        )

    lfpy_cell.destroy(sim=neuron_sim)


@pytest.mark.unit
@mock.patch.object(pathlib.Path, "glob")
def test_lfpysimulator_disable_banner_exception(mock_glob):
    """ephys.simulators: test LFPySimulator disable_banner raises exception"""
    mock_glob.return_value = []

    import warnings

    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("always")
        ephys.simulators.LFPySimulator._nrn_disable_banner()
        assert len(warnings_record) == 1


@pytest.mark.unit
def test_disable_banner_hoc_so():
    """ephys.simulators: test if disable_banner loads the hoc library"""

    fake_ctypes = mock.MagicMock()

    with mock.patch("platform.system", mock.MagicMock(return_value="Linux")), \
            mock.patch.object(
                pathlib.Path, "glob",
                return_value=[pathlib.Path("/fake/hoc.so")]), \
            mock.patch.object(ephys.simulators, "ctypes", fake_ctypes):
        neuron_sim = ephys.simulators.NrnSimulator()

        assert neuron_sim.disable_banner
        assert neuron_sim.banner_disabled
        fake_ctypes.cdll.__getitem__.assert_called_once_with("/fake/hoc.so")
        fake_ctypes.c_int.in_dll.assert_called_once()


@pytest.mark.unit
def test_nrnsim_run_fixed_dt():
    """ephys.simulators: test run with cvode disabled uses fixed dt"""

    neuron_sim = ephys.simulators.NrnSimulator(dt=0.025)
    neuron_sim.run(tstop=1.0, cvode_active=False)

    assert neuron_sim.neuron.h.dt == 0.025


@pytest.mark.unit
def test_nrnsim_run_explicit_dt():
    """ephys.simulators: test run applies the given fixed time step"""

    neuron_sim = ephys.simulators.NrnSimulator()
    fake_neuron = mock.MagicMock()

    with mock.patch.object(
        ephys.simulators.NrnSimulator, "neuron",
        new_callable=mock.PropertyMock, return_value=fake_neuron
    ):
        neuron_sim.run(tstop=1.0, cvode_active=False, dt=0.05)

    assert fake_neuron.h.dt == 0.05
    assert fake_neuron.h.steps_per_ms == 20.0
    fake_neuron.h.run.assert_called_once_with()


@pytest.mark.unit
def test_nrnsim_run_random123():
    """ephys.simulators: test run sets the Random123 global index"""

    neuron_sim = ephys.simulators.NrnSimulator()
    fake_neuron = mock.MagicMock()

    with mock.patch.object(
        ephys.simulators.NrnSimulator, "neuron",
        new_callable=mock.PropertyMock, return_value=fake_neuron
    ):
        neuron_sim.run(tstop=1.0, cvode_active=True, random123_globalindex=5)

    fake_neuron.h.Random.return_value.Random123_globalindex \
        .assert_called_once_with(5)


@pytest.mark.unit
def test_nrnsim_run_exception():
    """ephys.simulators: test run wraps hoc errors in NrnSimulatorException"""

    neuron_sim = ephys.simulators.NrnSimulator()

    fake_neuron = mock.MagicMock()
    fake_neuron.h.dt = neuron_sim.dt
    fake_neuron.h.run.side_effect = RuntimeError("hoc failure")

    with mock.patch.object(
        ephys.simulators.NrnSimulator, "neuron",
        new_callable=mock.PropertyMock, return_value=fake_neuron
    ):
        pytest.raises(
            ephys.simulators.NrnSimulatorException,
            neuron_sim.run,
            tstop=1.0)


@pytest.mark.unit
def test_lfpysim_run_random123():
    """ephys.simulators: test LFPy run sets the Random123 global index"""

    neuron_sim = ephys.simulators.LFPySimulator()
    lfpy_cell = mock.MagicMock()
    fake_neuron = mock.MagicMock()

    with mock.patch.object(
        ephys.simulators.LFPySimulator, "neuron",
        new_callable=mock.PropertyMock, return_value=fake_neuron
    ):
        neuron_sim.run(
            lfpy_cell=lfpy_cell,
            lfpy_electrode=None,
            tstop=1.0,
            random123_globalindex=5)

    assert lfpy_cell.simulate.called
    fake_neuron.h.Random.return_value.Random123_globalindex \
        .assert_called_once_with(5)


@pytest.mark.unit
def test_lfpysim_run_exception():
    """ephys.simulators: test LFPy run raises LFPySimulatorException"""

    neuron_sim = ephys.simulators.LFPySimulator()
    lfpy_cell = mock.MagicMock()
    lfpy_cell.simulate.side_effect = RuntimeError("simulate failure")

    with pytest.raises(ephys.simulators.LFPySimulatorException):
        neuron_sim.run(
            lfpy_cell=lfpy_cell,
            lfpy_electrode=None,
            tstop=1.0)


def make_fake_arbor():
    """Build a fake arbor namespace for ArbSimulator tests"""

    catalogue = mock.MagicMock()

    cell_model = mock.MagicMock()
    cell_model.properties.catalogue = catalogue

    fake_arbor = types.SimpleNamespace(
        cable_cell=mock.MagicMock(return_value=mock.MagicMock()),
        single_cell_model=mock.MagicMock(return_value=cell_model),
        catalogue=mock.MagicMock(return_value=catalogue),
        load_catalogue=mock.MagicMock(return_value=mock.MagicMock()),
        default_catalogue=mock.MagicMock(return_value=mock.MagicMock()),
        bbp_catalogue=mock.MagicMock(return_value=mock.MagicMock()),
        allen_catalogue=mock.MagicMock(return_value=mock.MagicMock()),
        units=types.SimpleNamespace(ms=1.0),
    )
    return fake_arbor, cell_model


@pytest.mark.unit
def test_arbsimulator_init():
    """ephys.simulators: test if ArbSimulator constructor works"""

    arb_sim = ephys.simulators.ArbSimulator()
    assert isinstance(arb_sim, ephys.simulators.ArbSimulator)
    arb_sim.initialize()


@pytest.mark.unit
def test_arbsimulator_missing_catalogue():
    """ephys.simulators: test missing catalogue raises ArbSimulatorException"""

    pytest.raises(
        ephys.simulators.ArbSimulatorException,
        ephys.simulators.ArbSimulator,
        ext_catalogues={"mycat": "/nonexistent/path"})


@pytest.mark.unit
def test_arbsimulator_instantiate(tmp_path, monkeypatch):
    """ephys.simulators: test ArbSimulator instantiate with catalogues"""

    fake_arbor, cell_model = make_fake_arbor()
    monkeypatch.setattr(ephys.simulators, "arbor", fake_arbor)

    (tmp_path / "mycat-catalogue.so").touch()
    arb_sim = ephys.simulators.ArbSimulator(
        ext_catalogues={"mycat": str(tmp_path)})
    model = arb_sim.instantiate(morph=None, decor=None, labels=None)

    assert model is cell_model
    fake_arbor.load_catalogue.assert_called_once()
    # mycat + default + BBP + allen
    assert cell_model.properties.catalogue.extend.call_count == 4


@pytest.mark.unit
def test_arbsimulator_instantiate_no_ext(monkeypatch):
    """ephys.simulators: test ArbSimulator instantiate without catalogues"""

    fake_arbor, cell_model = make_fake_arbor()
    monkeypatch.setattr(ephys.simulators, "arbor", fake_arbor)

    arb_sim = ephys.simulators.ArbSimulator()
    model = arb_sim.instantiate(morph=None, decor=None, labels=None)

    assert model is cell_model
    fake_arbor.load_catalogue.assert_not_called()
    # default + BBP + allen
    assert cell_model.properties.catalogue.extend.call_count == 3


@pytest.mark.unit
def test_arbsimulator_run(monkeypatch):
    """ephys.simulators: test ArbSimulator run with and without dt"""

    fake_arbor, cell_model = make_fake_arbor()
    monkeypatch.setattr(ephys.simulators, "arbor", fake_arbor)

    arb_cell_model = mock.MagicMock()

    arb_sim = ephys.simulators.ArbSimulator(dt=0.1)
    arb_sim.run(arb_cell_model, tstop=10)
    arb_cell_model.run.assert_called_once_with(tfinal=10.0, dt=0.1)

    arb_cell_model = mock.MagicMock()
    arb_sim = ephys.simulators.ArbSimulator()
    arb_sim.run(arb_cell_model, tstop=10)
    arb_cell_model.run.assert_called_once_with(tfinal=10.0)
