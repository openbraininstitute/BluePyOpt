"""bluepyopt.ipyp.bpopt_tasksdb tests"""

import sqlite3
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import pytest

import bluepyopt.ipyp.bpopt_tasksdb as bpopt_tasksdb


@pytest.fixture
def tasksdb_filename(tmp_path):
    """Create a temporary ipyparallel tasks.db sqlite file"""
    db_filename = str(tmp_path / "tasks.db")
    conn = sqlite3.connect(db_filename)
    conn.execute(
        'CREATE TABLE "ipython-tasks" '
        '(started TEXT, completed TEXT, engine_uuid TEXT)'
    )
    conn.executemany(
        'INSERT INTO "ipython-tasks" VALUES (?, ?, ?)',
        [
            ("2020-01-01T00:00:00", "2020-01-01T00:00:10", "engine-0"),
            ("2020-01-01T00:00:05", "2020-01-01T00:00:20", "engine-1"),
            ("2020-01-01T00:00:10", "2020-01-01T00:00:30", "engine-0"),
            # incomplete task, should be skipped
            ("2020-01-01T00:00:15", None, "engine-1"),
        ],
    )
    conn.commit()
    conn.close()
    return db_filename


@pytest.fixture
def tasks(tasksdb_filename):
    """Tasks dict parsed from the fixture db"""
    tasks, _ = bpopt_tasksdb.get_engine_data(tasksdb_filename)
    return tasks


@pytest.mark.unit
def test_get_engine_data(tasksdb_filename):
    """ipyp.bpopt_tasksdb: Testing get_engine_data"""

    tasks, engine_number_map = bpopt_tasksdb.get_engine_data(
        tasksdb_filename)

    assert len(tasks) == 2
    assert len(tasks["engine-0"]) == 2
    assert len(tasks["engine-1"]) == 1
    assert sorted(engine_number_map.values()) == [0, 1]

    task = tasks["engine-0"][0]
    assert task["duration"] == pytest.approx(10.0)
    assert task["engine_uuid"] == "engine-0"


@pytest.mark.unit
def test_get_engine_data_no_completed(tmp_path):
    """ipyp.bpopt_tasksdb: get_engine_data without completed tasks"""

    db_filename = str(tmp_path / "empty.db")
    conn = sqlite3.connect(db_filename)
    conn.execute(
        'CREATE TABLE "ipython-tasks" '
        '(started TEXT, completed TEXT, engine_uuid TEXT)'
    )
    conn.execute(
        'INSERT INTO "ipython-tasks" VALUES (?, ?, ?)',
        ("2020-01-01T00:00:00", None, "engine-0"),
    )
    conn.commit()
    conn.close()

    pytest.raises(
        Exception,
        bpopt_tasksdb.get_engine_data,
        db_filename)


@pytest.mark.unit
def test_calculate_unused_compute(tasks):
    """ipyp.bpopt_tasksdb: Testing calculate_unused_compute"""

    idle_time, idle_perc = bpopt_tasksdb.calculate_unused_compute(tasks)

    # window is [00:00:00, 00:00:30] on 2 engines -> 60s of compute,
    # tasks take 10 + 15 + 20 = 45s -> 15s idle
    assert idle_time.total_seconds() == pytest.approx(15.0)
    assert idle_perc == pytest.approx(100.0 * 15.0 / 45.0)


@pytest.mark.unit
def test_plot_usage(tasks):
    """ipyp.bpopt_tasksdb: Testing plot_usage"""

    engine_number_map = dict(zip(tasks.keys(), range(len(tasks))))
    bpopt_tasksdb.plot_usage(tasks, engine_number_map)
    plt.close("all")


@pytest.mark.unit
def test_plot_duration_histogram(tasks):
    """ipyp.bpopt_tasksdb: Testing plot_duration_histogram"""

    bpopt_tasksdb.plot_duration_histogram(tasks)
    plt.close("all")


@pytest.mark.unit
def test_run(tasksdb_filename, monkeypatch):
    """ipyp.bpopt_tasksdb: Testing run"""

    monkeypatch.setattr(plt, "show", lambda: None)
    bpopt_tasksdb.run([tasksdb_filename])
    plt.close("all")


@pytest.mark.unit
def test_run_missing_file():
    """ipyp.bpopt_tasksdb: run raises IOError on missing db"""

    pytest.raises(
        IOError,
        bpopt_tasksdb.run,
        ["/nonexistent/tasks.db"])


@pytest.mark.unit
def test_main(tasksdb_filename, monkeypatch):
    """ipyp.bpopt_tasksdb: Testing main"""

    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setattr(sys, "argv", ["bpopt_tasksdb", tasksdb_filename])
    bpopt_tasksdb.main()
    plt.close("all")
