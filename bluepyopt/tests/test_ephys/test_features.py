"""Tests for ephys.efeatures"""

import os
from os.path import join as joinp
from unittest import mock

import pytest
import numpy

from bluepyopt.ephys import efeatures
from bluepyopt.ephys.responses import TimeVoltageResponse, TimeLFPResponse
from bluepyopt.ephys.serializer import instantiator


@pytest.mark.unit
def test_EFeature():
    """ephys.efeatures: Testing EFeature creation"""
    efeature = efeatures.EFeature('name')
    assert efeature.name == 'name'


@pytest.mark.unit
def test_eFELFeature():
    """ephys.efeatures: Testing eFELFeature creation"""
    recording_names = {'': 'square_pulse_step1.soma.v'}
    efeature = efeatures.eFELFeature(name='test_eFELFeature',
                                     efel_feature_name='voltage_base',
                                     recording_names=recording_names,
                                     stim_start=700,
                                     stim_end=2700,
                                     exp_mean=1,
                                     exp_std=1)

    response = TimeVoltageResponse('mock_response')
    testdata_dir = joinp(
        os.path.dirname(
            os.path.abspath(__file__)),
        'testdata')
    response.read_csv(joinp(testdata_dir, 'TimeVoltageResponse.csv'))
    responses = {'square_pulse_step1.soma.v': response, }

    ret = efeature.calculate_feature(responses, raise_warnings=True)
    numpy.testing.assert_almost_equal(ret, -72.05761247316858)

    score = efeature.calculate_score(responses)
    numpy.testing.assert_almost_equal(score, 73.05761247316858)

    assert efeature.name == 'test_eFELFeature'
    assert 'voltage_base' in str(efeature)


@pytest.mark.unit
def test_eFELFeature_max_score():
    """ephys.efeatures: Testing eFELFeature max_score option"""

    recording_names = {'': 'square_pulse_step1.soma.v'}

    response = TimeVoltageResponse('mock_response')
    testdata_dir = joinp(
        os.path.dirname(
            os.path.abspath(__file__)),
        'testdata')
    response.read_csv(joinp(testdata_dir, 'TimeVoltageResponse.csv'))
    responses = {'square_pulse_step1.soma.v': response, }

    efeature_normal = efeatures.eFELFeature(name='test_eFELFeature',
                                            efel_feature_name='AP_amplitude',
                                            recording_names=recording_names,
                                            stim_start=600,
                                            stim_end=700,
                                            exp_mean=1,
                                            exp_std=1)
    score_normal = efeature_normal.calculate_score(responses)
    numpy.testing.assert_almost_equal(score_normal, 250)

    efeature_150 = efeatures.eFELFeature(name='test_eFELFeature',
                                         efel_feature_name='AP_amplitude',
                                         recording_names=recording_names,
                                         stim_start=600,
                                         stim_end=700,
                                         exp_mean=1,
                                         exp_std=1,
                                         max_score=150)

    score_150 = efeature_150.calculate_score(responses)
    numpy.testing.assert_almost_equal(score_150, 150)


@pytest.mark.unit
def test_eFELFeature_force_max_score():
    """ephys.efeatures: Testing eFELFeature force_max_score option"""

    recording_names = {'': 'square_pulse_step1.soma.v'}

    response = TimeVoltageResponse('mock_response')
    testdata_dir = joinp(
        os.path.dirname(
            os.path.abspath(__file__)),
        'testdata')
    response.read_csv(joinp(testdata_dir, 'TimeVoltageResponse.csv'))
    responses = {'square_pulse_step1.soma.v': response, }

    efeature_normal = efeatures.eFELFeature(name='test_eFELFeature',
                                            efel_feature_name='voltage_base',
                                            recording_names=recording_names,
                                            stim_start=700,
                                            stim_end=2700,
                                            exp_mean=1,
                                            exp_std=.001)
    score_normal = efeature_normal.calculate_score(responses)
    assert score_normal > 250

    efeature_force = efeatures.eFELFeature(name='test_eFELFeature',
                                           efel_feature_name='voltage_base',
                                           recording_names=recording_names,
                                           stim_start=700,
                                           stim_end=2700,
                                           exp_mean=1,
                                           exp_std=.001,
                                           force_max_score=True)

    score_force = efeature_force.calculate_score(responses)
    numpy.testing.assert_almost_equal(score_force, 250)


@pytest.mark.unit
def test_eFELFeature_double_settings():
    """ephys.efeatures: Testing eFELFeature double_settings"""
    recording_names = {'': 'square_pulse_step1.soma.v'}
    efeature = efeatures.eFELFeature(name='test_eFELFeature',
                                     efel_feature_name='voltage_base',
                                     recording_names=recording_names,
                                     stim_start=700,
                                     stim_end=2700,
                                     exp_mean=1,
                                     exp_std=1)
    efeature_ds = efeatures.eFELFeature(
        name='test_eFELFeature_other_perc',
        efel_feature_name='voltage_base',
        recording_names=recording_names,
        stim_start=700,
        stim_end=2700,
        exp_mean=1,
        exp_std=1,
        double_settings={
            'voltage_base_start_perc': 0.01})

    response = TimeVoltageResponse('mock_response')
    testdata_dir = joinp(
        os.path.dirname(
            os.path.abspath(__file__)),
        'testdata')
    response.read_csv(joinp(testdata_dir, 'TimeVoltageResponse.csv'))
    responses = {'square_pulse_step1.soma.v': response, }

    vb_other_perc = efeature_ds.calculate_feature(
        responses,
        raise_warnings=True)
    vb = efeature.calculate_feature(responses, raise_warnings=True)

    assert vb_other_perc != vb


@pytest.mark.unit
def test_eFELFeature_int_settings():
    """ephys.efeatures: Testing eFELFeature int_settings"""
    recording_names = {'': 'square_pulse_step1.soma.v'}
    efeature = efeatures.eFELFeature(name='test_eFELFeature',
                                     efel_feature_name='Spikecount',
                                     recording_names=recording_names,
                                     stim_start=1200,
                                     stim_end=2000,
                                     exp_mean=1,
                                     exp_std=1)
    efeature_strict = efeatures.eFELFeature(
        name='test_eFELFeature_strict',
        efel_feature_name='Spikecount',
        recording_names=recording_names,
        stim_start=1200,
        stim_end=2000,
        exp_mean=1,
        exp_std=1,
        int_settings={
            'strict_stiminterval': True})

    response = TimeVoltageResponse('mock_response')
    testdata_dir = joinp(
        os.path.dirname(
            os.path.abspath(__file__)),
        'testdata')
    response.read_csv(joinp(testdata_dir, 'TimeVoltageResponse.csv'))
    responses = {'square_pulse_step1.soma.v': response, }

    spikecount = efeature.calculate_feature(responses)
    spikecount_strict = efeature_strict.calculate_feature(responses)

    assert spikecount_strict != spikecount


@pytest.mark.unit
def test_eFELFeature_string_settings():
    """ephys.efeatures: Testing eFELFeature string_settings"""
    recording_names = {'': 'square_pulse_step1.soma.v'}
    efeature = efeatures.eFELFeature(name='test_eFELFeature_vb_default',
                                     efel_feature_name='voltage_base',
                                     recording_names=recording_names,
                                     stim_start=700,
                                     stim_end=2700)
    efeature_median = efeatures.eFELFeature(
        name='test_eFELFeature_vb_median',
        efel_feature_name='voltage_base',
        recording_names=recording_names,
        stim_start=700,
        stim_end=2700,
        string_settings={
            'voltage_base_mode': "median"})

    response = TimeVoltageResponse('mock_response')
    testdata_dir = joinp(
        os.path.dirname(
            os.path.abspath(__file__)),
        'testdata')
    response.read_csv(joinp(testdata_dir, 'TimeVoltageResponse.csv'))
    responses = {'square_pulse_step1.soma.v': response, }

    vb_median = efeature_median.calculate_feature(
        responses,
        raise_warnings=True)
    vb_default = efeature.calculate_feature(responses, raise_warnings=True)

    assert vb_median != vb_default


@pytest.mark.unit
def test_eFELFeature_serialize():
    """ephys.efeatures: Testing eFELFeature serialization"""
    recording_names = {'': 'square_pulse_step1.soma.v'}
    efeature = efeatures.eFELFeature(name='test_eFELFeature',
                                     efel_feature_name='voltage_base',
                                     recording_names=recording_names,
                                     stim_start=700,
                                     stim_end=2700,
                                     exp_mean=1,
                                     exp_std=1)
    serialized = efeature.to_dict()
    deserialized = instantiator(serialized)
    assert isinstance(deserialized, efeatures.eFELFeature)
    assert deserialized.stim_start == 700
    assert deserialized.recording_names == recording_names


@pytest.mark.unit
def test_extraFELFeature():
    """ephys.efeatures: Testing extraFELFeature calculation"""
    import pandas as pd

    somatic_recording_name = 'soma_response'
    recording_names = {'': 'lfp_response'}
    channel_ids = 0
    extrafel_feature_name = 'halfwidth'
    name = 'test_extraFELFeature'
    stim_start = 400
    stim_end = 1750
    fs = 10
    ms_cut = [10, 25]

    # load responses from file
    testdata_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'testdata'
    )
    soma_time = numpy.load(os.path.join(testdata_dir, 'lfpy_soma_time.npy'))
    soma_voltage = numpy.load(
        os.path.join(testdata_dir, 'lfpy_soma_voltage.npy')
    )
    lfpy_time = numpy.load(os.path.join(testdata_dir, 'lfpy_time.npy'))
    lfpy_voltage = numpy.load(os.path.join(testdata_dir, 'lfpy_voltage.npy'))

    soma_response = TimeVoltageResponse(
        name='soma_response', time=soma_time, voltage=soma_voltage
    )
    lfpy_response = TimeLFPResponse(
        name="lfpy_response", time=lfpy_time, lfp=lfpy_voltage
    )
    responses = {
        somatic_recording_name: soma_response,
        recording_names['']: lfpy_response,
    }

    # compute for all electrodes
    efeature = efeatures.extraFELFeature(
        name=name,
        extrafel_feature_name=extrafel_feature_name,
        somatic_recording_name=somatic_recording_name,
        recording_names=recording_names,
        channel_ids=None,
        exp_mean=0.001,
        exp_std=0.001,
        stim_start=stim_start,
        stim_end=stim_end,
        fs=fs,
        ms_cut=ms_cut
    )

    ret = efeature.calculate_feature(responses, raise_warnings=True)
    assert len(ret) == 209

    # compute for 1 electrode
    efeature = efeatures.extraFELFeature(
        name=name,
        extrafel_feature_name=extrafel_feature_name,
        somatic_recording_name=somatic_recording_name,
        recording_names=recording_names,
        channel_ids=channel_ids,
        exp_mean=0.001,
        exp_std=0.001,
        stim_start=stim_start,
        stim_end=stim_end,
        fs=fs,
        ms_cut=ms_cut
    )

    ret = efeature.calculate_feature(responses, raise_warnings=True)
    numpy.testing.assert_almost_equal(ret, 0.0015)

    score = efeature.calculate_score(responses)
    numpy.testing.assert_almost_equal(score, 0.5)

    assert efeature.name == name
    assert extrafel_feature_name in str(efeature)


@pytest.mark.unit
def test_masked_cosine_distance():
    """ephys.efeatures: Testing masked_cosine_distance"""
    from scipy.spatial import distance

    exp = numpy.array([0.5, 0.5, 0.5])
    model = numpy.array([0.7, 0.8, 0.4])

    score = efeatures.masked_cosine_distance(exp, model)
    assert score == distance.cosine(exp, model)

    # test nan in model feature values
    model = numpy.array([0.7, numpy.nan, 0.4])
    score = efeatures.masked_cosine_distance(exp, model)
    assert score == distance.cosine([0.5, 0.5], [0.7, 0.4])

    # test nan in experimental feature values
    exp = numpy.array([0.5, 0.5, numpy.nan])
    model = numpy.array([0.7, 0.8, 0.4])

    score = efeatures.masked_cosine_distance(exp, model)
    assert score == distance.cosine([0.5, 0.5], [0.7, 0.8]) * 2. / 3.

    # test na in both exp and model feature values
    exp = numpy.array([0.5, 0.5, numpy.nan])
    model = numpy.array([0.7, numpy.nan, 0.4])

    score = efeatures.masked_cosine_distance(exp, model)
    assert score == distance.cosine([0.5], [0.7]) * 2. / 3.


def load_response():
    testdata_dir = joinp(
        os.path.dirname(os.path.abspath(__file__)), 'testdata')
    response = TimeVoltageResponse('mock_response')
    response.read_csv(joinp(testdata_dir, 'TimeVoltageResponse.csv'))
    return response


def efel_feature(recording_names=None, **kwargs):
    return efeatures.eFELFeature(
        name='feature',
        efel_feature_name='voltage_base',
        recording_names=recording_names or {'': 'soma.v'},
        stim_start=700,
        stim_end=2700,
        exp_mean=1,
        exp_std=1,
        **kwargs)


@pytest.mark.unit
def test_eFELFeature_needs_an_unnamed_recording():
    """ephys.efeatures: eFELFeature requires the '' recording"""
    feature = efel_feature(recording_names={'soma': 'soma.v'})

    with pytest.raises(Exception, match="needs to be in recording_names"):
        feature.calculate_feature({})


@pytest.mark.unit
@pytest.mark.parametrize("responses", [{}, {'soma.v': None}])
def test_eFELFeature_without_a_response(responses):
    """ephys.efeatures: a missing response gives no value and max score"""
    feature = efel_feature()

    assert feature.calculate_feature(responses) is None
    assert feature.calculate_score(responses) == feature.max_score


@pytest.mark.unit
def test_eFELFeature_names_traces_of_other_locations():
    """ephys.efeatures: extra recordings get a location postfix"""
    feature = efel_feature(
        recording_names={'': 'soma.v', 'location_AIS': 'ais.v'})
    response = load_response()

    trace = feature._construct_efel_trace(
        {'soma.v': response, 'ais.v': response})

    assert {'T', 'V', 'T;location_AIS', 'V;location_AIS'} <= set(trace)


@pytest.mark.unit
def test_eFELFeature_applies_interp_step():
    """ephys.efeatures: interp_step is passed on to eFEL"""
    feature = efel_feature(interp_step=0.1)

    with mock.patch('efel.setDoubleSetting') as set_double:
        feature._setup_efel()

    set_double.assert_any_call('interp_step', 0.1)


SOMA = 'soma_response'
LFP = 'lfp_response'


def extra_responses():
    testdata_dir = joinp(
        os.path.dirname(os.path.abspath(__file__)), 'testdata')
    soma = TimeVoltageResponse(
        name=SOMA,
        time=numpy.load(joinp(testdata_dir, 'lfpy_soma_time.npy')),
        voltage=numpy.load(joinp(testdata_dir, 'lfpy_soma_voltage.npy')))
    lfp = TimeLFPResponse(
        name=LFP,
        time=numpy.load(joinp(testdata_dir, 'lfpy_time.npy')),
        lfp=numpy.load(joinp(testdata_dir, 'lfpy_voltage.npy')))
    return {SOMA: soma, LFP: lfp}


def extra_feature(**kwargs):
    arguments = dict(
        name='extra',
        extrafel_feature_name='halfwidth',
        somatic_recording_name=SOMA,
        recording_names={'': LFP},
        channel_ids=0,
        exp_mean=0.001,
        exp_std=0.001,
        stim_start=400,
        stim_end=1750,
        fs=10,
        ms_cut=[10, 25])
    arguments.update(kwargs)
    return efeatures.extraFELFeature(**arguments)


@pytest.mark.unit
@pytest.mark.parametrize("responses", [{}, {SOMA: None, LFP: None}])
def test_extraFELFeature_without_a_somatic_response(responses):
    """ephys.efeatures: no somatic response means no spikes to look at"""
    feature = extra_feature()

    assert feature.calculate_feature(responses) is None
    assert feature.calculate_feature(
        responses, return_waveforms=True) == (None, None)


@pytest.mark.unit
def test_extraFELFeature_without_an_extracellular_response():
    """ephys.efeatures: a missing LFP gives no value and max score"""
    feature = extra_feature()
    responses = extra_responses()
    responses[LFP] = None

    assert feature.calculate_feature(responses) is None
    assert feature.calculate_feature(
        responses, return_waveforms=True) == (None, None)
    assert feature.calculate_score(responses) == feature.max_score


@pytest.mark.unit
def test_extraFELFeature_can_return_waveforms():
    """ephys.efeatures: the mean waveform is available on request"""
    value, waveforms = extra_feature().calculate_feature(
        extra_responses(), return_waveforms=True)

    numpy.testing.assert_almost_equal(value, 0.0015)
    assert waveforms.ndim == 2


@pytest.mark.unit
def test_extraFELFeature_applies_efel_settings():
    """ephys.efeatures: every eFEL setting is passed on"""
    feature = extra_feature(
        interp_step=0.1,
        double_settings={'a_double': 1.5},
        int_settings={'an_int': 3})

    with mock.patch('efel.setDoubleSetting') as set_double, \
            mock.patch('efel.setIntSetting') as set_int:
        feature._setup_efel()

    set_double.assert_any_call('interp_step', 0.1)
    set_double.assert_any_call('a_double', 1.5)
    set_int.assert_any_call('an_int', 3)


@pytest.mark.unit
@pytest.mark.parametrize("value, feature_kwargs", [
    (numpy.inf, {}),
    (numpy.float64(1.0), {'exp_std': 0.0}),
    (numpy.nan, {}),
])
def test_extraFELFeature_unusable_scores_become_max_score(
        value, feature_kwargs):
    """ephys.efeatures: non-finite values score max_score"""
    feature = extra_feature(**feature_kwargs)
    feature.calculate_feature = lambda *args, **kwargs: value

    assert feature.calculate_score(extra_responses()) == feature.max_score


@pytest.mark.unit
def test_extraFELFeature_scores_all_channels_by_cosine_distance():
    """ephys.efeatures: several channels are compared as a vector"""
    feature = extra_feature(channel_ids=None, exp_mean=[1.0, 0.0])
    feature.calculate_feature = lambda *args, **kwargs: numpy.array([0., 1.])

    assert feature.calculate_score(extra_responses()) == pytest.approx(1.0)


@pytest.mark.unit
def test_extraFELFeature_force_max_score_caps_the_score():
    """ephys.efeatures: force_max_score limits large scores"""
    feature = extra_feature(force_max_score=True, max_score=2.0)
    feature.calculate_feature = lambda *args, **kwargs: numpy.float64(1.0)

    assert feature.calculate_score(extra_responses()) == 2.0


def sine_response(channels=None):
    time = numpy.arange(0.0, 200.0, 0.1)
    voltage = 5.0 + numpy.sin(2 * numpy.pi * time / 2.0)
    if channels:
        voltage = numpy.tile(voltage, (channels, 1))
    return {'time': time, 'voltage': voltage}


@pytest.mark.unit
@pytest.mark.parametrize("channels", [None, 2])
@pytest.mark.parametrize("filt_type", ["lfilter", "filtfilt"])
def test_filter_response_highpass_removes_the_offset(channels, filt_type):
    """ephys.efeatures: a scalar cut-off is a high-pass filter"""
    filtered = efeatures._filter_response(
        sine_response(channels), fcut=100.0, filt_type=filt_type)

    assert filtered['voltage'].shape == sine_response(channels)[
        'voltage'].shape
    assert abs(numpy.mean(filtered['voltage'][..., 1000:])) < 0.1


@pytest.mark.unit
def test_get_waveforms_supports_a_single_channel():
    """ephys.efeatures: a 1D trace is treated as one channel"""
    waveforms = efeatures._get_waveforms(
        sine_response(), numpy.array([50.0]), [1.0, 2.0])

    assert waveforms.shape == (1, 1, 30)


@pytest.mark.unit
def test_get_waveforms_pads_snippets_outside_the_recording():
    """ephys.efeatures: out-of-bounds snippets are padded with zeros"""
    response = sine_response()
    near_start, near_end = efeatures._get_waveforms(
        response, numpy.array([0.5, 199.5]), [1.0, 2.0])[:, 0]

    assert numpy.all(near_start[:5] == 0)
    assert numpy.all(near_end[-15:] == 0)


@pytest.mark.unit
@pytest.mark.xfail(
    raises=ValueError,
    strict=True,
    reason="a scalar snippet length mixes samples and milliseconds",
)
def test_get_waveforms_accepts_a_scalar_snippet_length():
    """ephys.efeatures: a scalar snippet length is split around the peak"""
    waveforms = efeatures._get_waveforms(
        sine_response(), numpy.array([50.0]), 3.0)

    assert waveforms.shape == (1, 1, 30)
