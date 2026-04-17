"""Tests for NrnSimulator and LFPySimulator mechanisms_directory.

Verifies that simulators correctly store mechanisms_directory and that
evaluators set it, ensuring pebble subprocesses can reload mechanisms.

**Validates: Requirements 1.1, 1.2, 2.1, 2.2, 3.1, 3.2, 3.5, 3.6**
"""

import os
import sys

import pytest

import bluepyopt.ephys as ephys


# -------------------------------------------------------------------
# Bug condition tests: mechanisms_directory handling
# -------------------------------------------------------------------

@pytest.mark.unit
def test_nrnsimulator_mechanisms_directory_set():
    """NrnSimulator(mechanisms_directory=path) stores the attribute."""
    for path in ["/tmp/mechs", "/some/other/path", "relative/path"]:
        sim = ephys.simulators.NrnSimulator(
            mechanisms_directory=path)
        assert sim.mechanisms_directory == path


@pytest.mark.unit
def test_lfpysimulator_mechanisms_directory_set():
    """LFPySimulator(mechanisms_directory=path) stores the attribute."""
    for path in ["/tmp/mechs", "/some/other/path"]:
        sim = ephys.simulators.LFPySimulator(
            mechanisms_directory=path)
        assert sim.mechanisms_directory == path


@pytest.mark.unit
def test_l5pc_evaluator_has_mechanisms_directory():
    """l5pc_evaluator.create() sets mechanisms_directory on sim."""
    l5pc_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../../examples/l5pc"))
    sys.path.insert(0, l5pc_path)

    import l5pc_evaluator  # noqa: E402

    evaluator = l5pc_evaluator.create()
    assert evaluator.sim.mechanisms_directory is not None


@pytest.mark.unit
def test_l5pc_lfpy_evaluator_has_mechanisms_directory():
    """l5pc_lfpy_evaluator.create() sets mechanisms_directory."""
    lfpy_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../../examples/l5pc_lfpy"))
    sys.path.insert(0, lfpy_path)

    import l5pc_lfpy_evaluator  # noqa: E402

    feature_file = os.path.join(
        lfpy_path, "extra_features.json")
    evaluator = l5pc_lfpy_evaluator.create(
        feature_file=feature_file)
    assert evaluator.sim.mechanisms_directory is not None


# -------------------------------------------------------------------
# Preservation tests: NrnSimulator core behavior unchanged
# -------------------------------------------------------------------

@pytest.mark.unit
def test_nrnsimulator_defaults_preserved():
    """NrnSimulator() defaults are all correct, and custom args are stored."""
    sim = ephys.simulators.NrnSimulator()
    assert sim.cvode_active is True
    assert isinstance(sim.dt, float)
    assert sim.dt > 0
    assert sim.mechanisms_directory is None
    assert sim.cvode_minstep_value is None
    assert sim.random123_globalindex is None


@pytest.mark.unit
def test_nrnsimulator_attributes_preserved():
    """NrnSimulator preserves dt, cvode_active, mechanisms_directory."""
    test_cases = [
        (0.025, False, "/tmp/mechs"),
        (0.001, True, None),
        (0.1, False, "/another/path"),
    ]
    for dt_val, cv_val, md_val in test_cases:
        sim = ephys.simulators.NrnSimulator(
            dt=dt_val,
            cvode_active=cv_val,
            mechanisms_directory=md_val)
        assert sim.dt == dt_val
        assert sim.cvode_active == cv_val
        assert sim.mechanisms_directory == md_val
