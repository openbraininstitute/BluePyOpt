"""bluepyopt.deapext.hype tests"""

import numpy

import bluepyopt.deapext.hype

import pytest


@pytest.mark.unit
def test_hypeIndicatorExact():
    """deapext.hype: Testing hypeIndicatorExact"""

    points = numpy.asarray([[250., 250.], [0., 0.], [240., 240.]])
    bounds = numpy.asarray([250., 250.])

    hv = bluepyopt.deapext.hype.hypeIndicatorExact(points, bounds, k=5)

    assert hv[0] == 0
    assert hv[1] == 62500
    assert hv[2] == 100


@pytest.mark.unit
def test_hypeIndicatorSampled():
    """deapext.hype: Testing hypeIndicatorSampled"""

    points = numpy.asarray([[250., 250.], [0., 0.], [240., 240.]])
    bounds = numpy.asarray([250., 250.])

    numpy.random.seed(42)
    hv = bluepyopt.deapext.hype.hypeIndicatorSampled(
        points, bounds, nrOfSamples=1000000, k=5
    )

    assert hv[0] == 0
    assert hv[1] == 62500
    assert numpy.abs((hv[2] / 100) - 1) < 0.05


@pytest.mark.unit
@pytest.mark.xfail(
    raises=AssertionError,
    strict=True,
    reason="hype divides by i inside the product of the alpha coefficients",
)
def test_hypeIndicatorExact_negative_k():
    """deapext.hype: hypeIndicatorExact defaults k to the population size"""

    points = numpy.asarray([[0., 0.], [1., 1.], [2., 2.]])
    bounds = numpy.asarray([5., 5.])

    hv = bluepyopt.deapext.hype.hypeIndicatorExact(points, bounds, k=-1)

    # with k equal to the population size, values sum to the dominated volume
    numpy.testing.assert_almost_equal(hv, [15.5, 6.5, 3.])


@pytest.mark.unit
def test_hypeIndicatorExact_small_k():
    """deapext.hype: hypeIndicatorExact stops once i exceeds k"""

    points = numpy.asarray([[0., 0.], [1., 1.], [2., 2.]])
    bounds = numpy.asarray([5., 5.])

    hv = bluepyopt.deapext.hype.hypeIndicatorExact(points, bounds, k=2)

    numpy.testing.assert_almost_equal(hv, [10.75, 1.75, 0.])
