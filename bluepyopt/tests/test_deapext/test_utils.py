"""bluepyopt.utils tests"""

import multiprocessing
import time

import bluepyopt.deapext.utils as utils
import pytest


def flag(event):
    """Send a multiprocessing event."""
    time.sleep(1)
    event.set()


def catch_event(event):
    """Verify that run_next_gen changes when event is caught."""
    # None case
    assert utils.run_next_gen(True, None)

    # event is not set case
    assert utils.run_next_gen(True, event)

    # event is set by another process case
    time.sleep(2)
    assert not (utils.run_next_gen(True, event))


@pytest.mark.unit
def test_run_next_gen_condition():
    """deapext.utils: Testing run_next_gen."""
    event = multiprocessing.Event()
    p1 = multiprocessing.Process(target=catch_event, args=(event,))
    p2 = multiprocessing.Process(target=flag, args=(event,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()


@pytest.mark.unit
def test_uniform_scalar_bounds():
    """deapext.utils: Testing uniform with scalar bounds"""
    individual = utils.uniform(0.0, 1.0, 5)

    assert len(individual) == 5
    assert all(0.0 <= value <= 1.0 for value in individual)


@pytest.mark.unit
def test_reduce_method():
    """deapext.utils: Testing reduce_method"""
    class _Foo(object):
        def meth(self):
            pass

    foo = _Foo()
    getter, (obj, name) = utils.reduce_method(foo.meth)

    assert getter is getattr
    assert obj is foo
    assert name == "meth"


@pytest.mark.unit
def test_run_next_gen_terminator():
    """deapext.utils: Testing run_next_gen with a terminator"""
    class Terminator(object):
        def __init__(self, is_set):
            self._is_set = is_set

        def is_set(self):
            return self._is_set

    assert utils.run_next_gen(True, Terminator(False))
    assert not utils.run_next_gen(True, Terminator(True))
    assert not utils.run_next_gen(False, Terminator(False))
