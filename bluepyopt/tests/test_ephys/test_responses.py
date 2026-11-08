"""Tests of ephys.responses"""

from unittest import mock

import pandas
import pytest

from bluepyopt.ephys import responses


def voltage_response():
    return responses.TimeVoltageResponse(
        "soma.v", time=[0.0, 1.0], voltage=[-70.0, -60.0]
    )


@pytest.mark.unit
def test_response_str():
    assert str(voltage_response()) == "TimeVoltageResponse: soma.v"


@pytest.mark.unit
def test_response_csv_round_trip(tmp_path):
    path = tmp_path / "response.csv"
    voltage_response().to_csv(path)

    assert pandas.read_csv(path)["voltage"].tolist() == [-70.0, -60.0]


@pytest.mark.unit
def test_response_plot_draws_voltage_against_time():
    axes = mock.MagicMock()

    voltage_response().plot(axes)

    time, voltage = axes.plot.call_args.args[:2]
    assert time.tolist() == [0.0, 1.0]
    assert voltage.tolist() == [-70.0, -60.0]


@pytest.mark.unit
def test_lfp_response_cannot_be_plotted():
    response = responses.TimeLFPResponse("lfp", time=[0.0, 1.0], lfp=[1, 2])

    with pytest.raises(NotImplementedError):
        response.plot(mock.MagicMock())
