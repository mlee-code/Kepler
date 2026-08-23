import numpy as np

from src.orbit import accelerations, conservation_summary, get_preset, simulate, to_dataframe


def test_accelerations_obey_newtons_third_law():
    a1, a2 = accelerations(np.array([-1.0, 0.0]), np.array([1.0, 0.0]), 2.0, 3.0, 1.0)
    np.testing.assert_allclose(2.0 * a1, -3.0 * a2)


def test_circular_binary_remains_bound_and_conserves_energy():
    config = get_preset("Stable circular binary")
    result = simulate(config)
    summary = conservation_summary(result)
    assert result["classification"] == "Bound orbit"
    assert summary["max_relative_energy_error"] < 1e-6
    assert abs(result["separation"][-1] - result["separation"][0]) < 1e-4


def test_collision_preset_stops_early():
    result = simulate(get_preset("Head-on collision"))
    assert result["event"] == "collision"
    assert result["classification"] == "Collision"
    assert result["time"][-1] < result["config"]["duration"]


def test_escape_preset_is_unbound():
    result = simulate(get_preset("Hyperbolic escape"))
    assert result["classification"] == "Escape / unbound"
    assert result["specific_orbital_energy"][0] > 0


def test_dataframe_has_one_row_per_sample():
    result = simulate(get_preset("Stable circular binary"))
    frame = to_dataframe(result)
    assert len(frame) == len(result["time"])
    assert {"time", "body1_x", "body2_speed", "total_energy"} <= set(frame.columns)
