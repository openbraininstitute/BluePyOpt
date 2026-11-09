"""Tests of ephys.protocols.ArbSweepProtocol"""

import os
import time
from types import SimpleNamespace
from unittest import mock

import numpy
import pytest

from bluepyopt import ephys
from bluepyopt.ephys import protocols

arbor = pytest.importorskip("arbor")

MORPHOLOGY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "testdata", "simple.swc"
)


def locset(name, expression="(root)"):
    return ephys.locations.ArbLocsetLocation(name=name, locset=expression)


def square_pulse(location, amplitude=0.5):
    return ephys.stimuli.NrnSquarePulse(
        step_amplitude=amplitude,
        step_delay=5.0,
        step_duration=10.0,
        total_duration=20.0,
        location=location,
    )


def recording(location, name="soma.v"):
    return ephys.recordings.CompRecording(
        name=name, location=location, variable="v"
    )


def passive_cell():
    somatic = ephys.locations.NrnSeclistLocation(
        "somatic", seclist_name="somatic"
    )
    return ephys.models.CellModel(
        "passive_cell",
        morph=ephys.morphologies.NrnFileMorphology(MORPHOLOGY),
        mechs=[
            ephys.mechanisms.NrnMODMechanism(
                name="pas", suffix="pas", locations=[somatic]
            )
        ],
        params=[],
    )


def make_protocol(use_labels=False):
    location = locset("soma_center", "(location 0 0.5)")
    return protocols.ArbSweepProtocol(
        "step",
        stimuli=[square_pulse(location)],
        recordings=[recording(location)],
        use_labels=use_labels,
    )


@pytest.mark.unit
def test_arbsweepprotocol_description():
    protocol = make_protocol()

    assert protocol.total_duration == 20.0
    assert list(protocol.subprotocols()) == ["step"]
    assert "stimuli" in str(protocol)
    assert "soma.v" in str(protocol)


@pytest.mark.unit
@pytest.mark.parametrize(
    "isolate, use_labels", [(False, False), (False, True), (True, False)]
)
def test_arbsweepprotocol_depolarises_during_the_step(isolate, use_labels):
    protocol = make_protocol(use_labels=use_labels)

    sim = ephys.simulators.ArbSimulator()

    responses = protocol.run(passive_cell(), {}, sim=sim, isolate=isolate)

    response = responses["soma.v"]
    assert response["time"].iloc[-1] == pytest.approx(20.0, abs=0.2)
    before = response["voltage"][response["time"] < 5].iloc[-1]
    during = response["voltage"][response["time"] < 15].iloc[-1]
    assert during > before


def synaptic_stimulus():
    stimulus = mock.MagicMock(
        spec=ephys.stimuli.NrnNetStimStimulus, total_duration=20.0
    )
    stimulus.acc_events.return_value = []
    return stimulus


@pytest.fixture
def fake_acc(monkeypatch):
    values = (
        "cell.json",
        mock.sentinel.morph,
        mock.MagicMock(name="decor"),
        arbor.label_dict(),
    )
    monkeypatch.setattr(
        protocols.create_acc, "read_acc", lambda path: values
    )


@pytest.mark.unit
def test_arbsweepprotocol_returns_none_when_the_simulation_fails(fake_acc):
    sim = mock.MagicMock()
    sim.run.side_effect = RuntimeError("diverged")
    protocol = protocols.ArbSweepProtocol(
        "step",
        stimuli=[synaptic_stimulus()],
        recordings=[SimpleNamespace(name="soma.v")],
    )
    protocol.instantiate_recordings = lambda model, use_labels: model

    assert protocol._run_func("cell.json", {}, sim=sim) == {"soma.v": None}


@pytest.mark.unit
def test_arbsweepprotocol_checks_the_number_of_traces(fake_acc):
    sim = mock.MagicMock()
    sim.instantiate.return_value.traces = []
    protocol = protocols.ArbSweepProtocol(
        "step",
        stimuli=[synaptic_stimulus()],
        recordings=[SimpleNamespace(name="soma.v")],
    )
    protocol.instantiate_recordings = lambda model, use_labels: model

    with pytest.raises(protocols.ArbSweepProtocolException) as error:
        protocol._run_func("cell.json", {}, sim=sim)

    assert "Number of Arbor voltage traces" in str(error.value.__cause__)


class SlowArbSweepProtocol(protocols.ArbSweepProtocol):
    def _run_func(self, cell_json, param_values, sim=None):
        time.sleep(30)


def fake_cell():
    return SimpleNamespace(name="cell", write_acc=lambda *a, **k: None)


@pytest.mark.unit
def test_arbsweepprotocol_rejects_negative_timeout():
    protocol = protocols.ArbSweepProtocol("step", stimuli=[], recordings=[])
    sim = SimpleNamespace(ext_catalogues=None)

    with pytest.raises(ValueError, match="timeout should be > 0"):
        protocol.run(fake_cell(), {}, sim=sim, isolate=True, timeout=-1)


@pytest.mark.unit
def test_arbsweepprotocol_returns_none_on_timeout():
    protocol = SlowArbSweepProtocol(
        "step", stimuli=[], recordings=[SimpleNamespace(name="soma.v")]
    )
    sim = SimpleNamespace(ext_catalogues=None)

    responses = protocol.run(
        fake_cell(), {}, sim=sim, isolate=True, timeout=0.2
    )

    assert responses == {"soma.v": None}


@pytest.mark.unit
def test_instantiate_locations_adds_stimulus_and_recording_labels():
    stimulus = square_pulse(locset("stim"))
    several = SimpleNamespace(
        locations=[
            SimpleNamespace(
                acc_label=lambda: [
                    locset("syn1").acc_label(),
                    locset("syn2").acc_label(),
                ]
            )
        ]
    )
    protocol = protocols.ArbSweepProtocol(
        "step",
        stimuli=[stimulus, several],
        recordings=[recording(locset("rec"))],
    )

    labels = protocol.instantiate_locations(arbor.label_dict())

    assert {"stim", "syn1", "syn2", "rec"} <= set(labels.keys())


@pytest.mark.unit
def test_instantiate_locations_rejects_a_conflict_with_existing_labels():
    protocol = protocols.ArbSweepProtocol(
        "step",
        stimuli=[square_pulse(locset("soma", "(root)"))],
        recordings=[],
    )
    labels = arbor.label_dict({"soma": "(location 0 0.5)"})

    with pytest.raises(ValueError, match="already exists"):
        protocol.instantiate_locations(labels)


@pytest.mark.unit
def test_instantiate_locations_rejects_a_conflict_within_the_protocol():
    protocol = protocols.ArbSweepProtocol(
        "step",
        stimuli=[square_pulse(locset("soma", "(root)"))],
        recordings=[recording(locset("soma", "(location 0 0.5)"))],
    )

    with pytest.raises(ValueError, match="defined multiple times"):
        protocol.instantiate_locations(arbor.label_dict())


@pytest.mark.unit
def test_instantiate_locations_rejects_a_recording_on_several_labels():
    location = SimpleNamespace(
        acc_label=lambda: [locset("a").acc_label(), locset("b").acc_label()]
    )
    protocol = protocols.ArbSweepProtocol(
        "step", stimuli=[], recordings=[SimpleNamespace(location=location)]
    )

    with pytest.raises(ValueError, match="length != 1"):
        protocol.instantiate_locations(arbor.label_dict())


@pytest.mark.unit
def test_instantiate_iclamp_requires_an_envelope():
    protocol = protocols.ArbSweepProtocol(
        "step", stimuli=[SimpleNamespace()], recordings=[]
    )

    with pytest.raises(ValueError, match="must provide envelope"):
        protocol.instantiate_iclamp_stimuli(mock.MagicMock())


@pytest.mark.unit
def test_instantiate_iclamp_places_the_stimulus_on_every_label():
    stimulus = square_pulse(
        SimpleNamespace(
            acc_label=lambda: [
                locset("a").acc_label(),
                locset("b").acc_label(),
            ]
        )
    )
    protocol = protocols.ArbSweepProtocol(
        "step", stimuli=[stimulus], recordings=[]
    )
    decor = mock.MagicMock()

    protocol.instantiate_iclamp_stimuli(decor, use_labels=True)

    assert [call.args[2] for call in decor.place.call_args_list] == [
        "step.iclamp.0.a",
        "step.iclamp.0.b",
    ]


@pytest.mark.unit
def test_instantiate_synaptic_stimuli_adds_every_event_generator():
    stimulus = mock.MagicMock(spec=ephys.stimuli.NrnNetStimStimulus)
    stimulus.acc_events.return_value = ["first", "second"]
    protocol = protocols.ArbSweepProtocol(
        "step", stimuli=[stimulus], recordings=[]
    )
    cell_model = mock.MagicMock()

    protocol.instantiate_synaptic_stimuli(cell_model)

    assert [c.args[0] for c in cell_model.event_generator.call_args_list] == [
        "first",
        "second",
    ]


@pytest.mark.unit
def test_instantiate_recordings_probes_each_location():
    protocol = protocols.ArbSweepProtocol(
        "step", stimuli=[], recordings=[recording(locset("rec"))]
    )
    cell_model = mock.MagicMock()
    cell_model.cable_cell.locations.return_value = ["only one"]

    protocol.instantiate_recordings(cell_model, use_labels=True)

    args = cell_model.probe.call_args.args
    assert args[0] == "voltage"
    assert args[1] == '(locset "rec")'
    assert args[2] == "probe-0"


@pytest.mark.unit
def test_instantiate_recordings_rejects_a_recording_on_several_labels():
    location = SimpleNamespace(
        acc_label=lambda: [locset("a").acc_label(), locset("b").acc_label()]
    )
    protocol = protocols.ArbSweepProtocol(
        "step", stimuli=[], recordings=[SimpleNamespace(location=location)]
    )

    with pytest.raises(ValueError, match="length != 1"):
        protocol.instantiate_recordings(mock.MagicMock())


@pytest.mark.unit
def test_instantiate_recordings_rejects_a_non_unique_location():
    protocol = protocols.ArbSweepProtocol(
        "step", stimuli=[], recordings=[recording(locset("rec"))]
    )
    cell_model = mock.MagicMock()
    cell_model.cable_cell.locations.return_value = ["one", "two"]

    with pytest.raises(ValueError, match="non-unique"):
        protocol.instantiate_recordings(cell_model)


@pytest.mark.unit
def test_arbsweepprotocol_response_is_a_time_series():
    protocol = make_protocol()

    response = protocol.run(
        passive_cell(), {}, sim=ephys.simulators.ArbSimulator(), isolate=False
    )["soma.v"]

    assert numpy.all(numpy.diff(response["time"]) > 0)


ONE_LABEL_BUG = (
    "a recording location with a one-item label list is used as a single label"
)


def one_label_recording():
    location = SimpleNamespace(acc_label=lambda: [locset("a").acc_label()])
    return SimpleNamespace(location=location, name="soma.v")


@pytest.mark.unit
@pytest.mark.xfail(raises=AttributeError, strict=True, reason=ONE_LABEL_BUG)
def test_instantiate_locations_accepts_a_recording_on_one_label():
    protocol = protocols.ArbSweepProtocol(
        "step", stimuli=[], recordings=[one_label_recording()]
    )

    labels = protocol.instantiate_locations(arbor.label_dict())

    assert "a" in labels.keys()


@pytest.mark.unit
@pytest.mark.xfail(raises=AttributeError, strict=True, reason=ONE_LABEL_BUG)
def test_instantiate_recordings_accepts_a_recording_on_one_label():
    protocol = protocols.ArbSweepProtocol(
        "step", stimuli=[], recordings=[one_label_recording()]
    )
    cell_model = mock.MagicMock()
    cell_model.cable_cell.locations.return_value = ["only one"]

    protocol.instantiate_recordings(cell_model)

    cell_model.probe.assert_called_once()
