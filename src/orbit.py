"""Structured, function-based Newtonian two-body simulation utilities."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PRESETS = {
    "Stable circular binary": {
        "description": "Equal masses on a circular orbit around their barycenter.",
        "m1": 1.0, "m2": 1.0,
        "r1": [-0.5, 0.0], "r2": [0.5, 0.0],
        "v1": [0.0, -0.70710678], "v2": [0.0, 0.70710678],
        "radius1": 0.035, "radius2": 0.035, "G": 1.0,
        "dt": 0.005, "duration": 12.0,
    },
    "Elliptical orbit": {
        "description": "A bound eccentric orbit with unequal masses.",
        "m1": 2.0, "m2": 0.5,
        "r1": [-0.2, 0.0], "r2": [0.8, 0.0],
        "v1": [0.0, -0.20], "v2": [0.0, 0.80],
        "radius1": 0.045, "radius2": 0.025, "G": 1.0,
        "dt": 0.003, "duration": 12.0,
    },
    "Near escape (marginal)": {
        "description": "A nearly parabolic flyby close to the escape threshold.",
        "m1": 10.0, "m2": 0.01,
        "r1": [-0.001, 0.0], "r2": [0.999, 0.0],
        "v1": [0.0, -0.00446], "v2": [0.0, 4.46],
        "radius1": 0.055, "radius2": 0.012, "G": 1.0,
        "dt": 0.001, "duration": 4.0,
    },
    "Hyperbolic escape": {
        "description": "Positive orbital energy produces an unbound trajectory.",
        "m1": 10.0, "m2": 0.01,
        "r1": [-0.001, 0.0], "r2": [0.999, 0.0],
        "v1": [0.0, -0.006], "v2": [0.0, 6.0],
        "radius1": 0.055, "radius2": 0.012, "G": 1.0,
        "dt": 0.001, "duration": 4.0,
    },
    "Head-on collision": {
        "description": "Two equal bodies start at rest and fall directly together.",
        "m1": 1.0, "m2": 1.0,
        "r1": [-0.75, 0.0], "r2": [0.75, 0.0],
        "v1": [0.0, 0.0], "v2": [0.0, 0.0],
        "radius1": 0.08, "radius2": 0.08, "G": 1.0,
        "dt": 0.001, "duration": 3.0,
    },
}


def get_preset(name: str) -> dict:
    """Return an independent copy of a named scenario."""
    if name not in PRESETS:
        raise KeyError(f"Unknown preset: {name}")
    return deepcopy(PRESETS[name])


def _validate(config: dict) -> None:
    for key in ("m1", "m2", "G", "dt", "duration"):
        if float(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("radius1", "radius2"):
        if float(config[key]) < 0:
            raise ValueError(f"{key} cannot be negative")
    if int(np.ceil(config["duration"] / config["dt"])) > 2_000_000:
        raise ValueError("Simulation exceeds the safety limit of 2,000,000 steps")


def accelerations(r1: np.ndarray, r2: np.ndarray, m1: float, m2: float, G: float):
    """Calculate both Newtonian accelerations."""
    displacement = r2 - r1
    distance_squared = float(displacement @ displacement)
    if distance_squared == 0:
        raise ZeroDivisionError("Bodies occupy the same position")
    inverse_distance_cubed = distance_squared ** -1.5
    return (G * m2 * displacement * inverse_distance_cubed,
            -G * m1 * displacement * inverse_distance_cubed)


def simulate(config: dict) -> dict:
    """Integrate a two-body system with velocity Verlet and collision events."""
    _validate(config)
    m1, m2, G = float(config["m1"]), float(config["m2"]), float(config["G"])
    dt, duration = float(config["dt"]), float(config["duration"])
    collision_distance = float(config["radius1"]) + float(config["radius2"])
    capacity = int(np.ceil(duration / dt)) + 1
    time = np.empty(capacity)
    r1 = np.empty((capacity, 2)); r2 = np.empty((capacity, 2))
    v1 = np.empty((capacity, 2)); v2 = np.empty((capacity, 2))
    time[0] = 0.0
    r1[0] = config["r1"]; r2[0] = config["r2"]
    v1[0] = config["v1"]; v2[0] = config["v2"]
    a1, a2 = accelerations(r1[0], r2[0], m1, m2, G)
    event = "duration_reached"
    last = 0

    if np.linalg.norm(r2[0] - r1[0]) <= collision_distance:
        event = "collision"
    else:
        for index in range(1, capacity):
            step = min(dt, duration - time[index - 1])
            if step <= 0:
                break
            r1[index] = r1[index - 1] + v1[index - 1] * step + 0.5 * a1 * step * step
            r2[index] = r2[index - 1] + v2[index - 1] * step + 0.5 * a2 * step * step
            separation = np.linalg.norm(r2[index] - r1[index])
            time[index] = time[index - 1] + step
            if separation <= collision_distance:
                v1[index] = v1[index - 1] + a1 * step
                v2[index] = v2[index - 1] + a2 * step
                last = index
                event = "collision"
                break
            next_a1, next_a2 = accelerations(r1[index], r2[index], m1, m2, G)
            v1[index] = v1[index - 1] + 0.5 * (a1 + next_a1) * step
            v2[index] = v2[index - 1] + 0.5 * (a2 + next_a2) * step
            a1, a2 = next_a1, next_a2
            last = index

    result = {"time": time[:last + 1], "r1": r1[:last + 1], "r2": r2[:last + 1],
              "v1": v1[:last + 1], "v2": v2[:last + 1], "event": event,
              "config": deepcopy(config)}
    result.update(diagnostics(result))
    result["classification"] = classify(result)
    return result


def diagnostics(result: dict) -> dict:
    """Calculate scalar observables and conservation quantities."""
    config = result["config"]
    m1, m2, G = config["m1"], config["m2"], config["G"]
    relative_r = result["r2"] - result["r1"]
    relative_v = result["v2"] - result["v1"]
    separation = np.linalg.norm(relative_r, axis=1)
    speed1 = np.linalg.norm(result["v1"], axis=1)
    speed2 = np.linalg.norm(result["v2"], axis=1)
    kinetic = 0.5 * m1 * speed1**2 + 0.5 * m2 * speed2**2
    potential = -G * m1 * m2 / separation
    total_energy = kinetic + potential
    angular1 = result["r1"][:, 0] * result["v1"][:, 1] - result["r1"][:, 1] * result["v1"][:, 0]
    angular2 = result["r2"][:, 0] * result["v2"][:, 1] - result["r2"][:, 1] * result["v2"][:, 0]
    angular_momentum = m1 * angular1 + m2 * angular2
    center_of_mass = (m1 * result["r1"] + m2 * result["r2"]) / (m1 + m2)
    specific_energy = 0.5 * np.sum(relative_v**2, axis=1) - G * (m1 + m2) / separation
    radial_velocity = np.sum(relative_r * relative_v, axis=1) / separation
    return {"separation": separation, "speed1": speed1, "speed2": speed2,
            "kinetic_energy": kinetic, "potential_energy": potential,
            "total_energy": total_energy, "angular_momentum": angular_momentum,
            "center_of_mass": center_of_mass, "specific_orbital_energy": specific_energy,
            "radial_velocity": radial_velocity}


def classify(result: dict) -> str:
    """Classify the terminal state using collision and orbital energy criteria."""
    if result["event"] == "collision":
        return "Collision"
    energy = result["specific_orbital_energy"][-1]
    radial_velocity = result["radial_velocity"][-1]
    if energy > 1e-8 and radial_velocity > 0:
        return "Escape / unbound"
    if abs(energy) <= 1e-8:
        return "Marginal escape"
    return "Bound orbit"


def conservation_summary(result: dict) -> dict:
    """Return relative drift in conserved quantities."""
    energy = result["total_energy"]
    angular = result["angular_momentum"]
    energy_scale = max(abs(energy[0]), np.finfo(float).eps)
    angular_scale = max(abs(angular[0]), np.finfo(float).eps)
    return {
        "classification": result["classification"],
        "event": result["event"],
        "samples": len(result["time"]),
        "simulated_time": float(result["time"][-1]),
        "max_relative_energy_error": float(np.max(np.abs(energy - energy[0])) / energy_scale),
        "max_relative_angular_momentum_error": float(np.max(np.abs(angular - angular[0])) / angular_scale),
        "minimum_separation": float(np.min(result["separation"])),
        "maximum_separation": float(np.max(result["separation"])),
    }


def to_dataframe(result: dict) -> pd.DataFrame:
    """Convert every simulation sample to a tidy tabular representation."""
    return pd.DataFrame({
        "time": result["time"],
        "body1_x": result["r1"][:, 0], "body1_y": result["r1"][:, 1],
        "body2_x": result["r2"][:, 0], "body2_y": result["r2"][:, 1],
        "body1_vx": result["v1"][:, 0], "body1_vy": result["v1"][:, 1],
        "body2_vx": result["v2"][:, 0], "body2_vy": result["v2"][:, 1],
        "body1_speed": result["speed1"], "body2_speed": result["speed2"],
        "separation": result["separation"], "kinetic_energy": result["kinetic_energy"],
        "potential_energy": result["potential_energy"], "total_energy": result["total_energy"],
        "angular_momentum": result["angular_momentum"],
    })


def _sample_indices(length: int, maximum: int = 350) -> np.ndarray:
    return np.unique(np.linspace(0, length - 1, min(length, maximum), dtype=int))


def trajectory_figure(result: dict, animated: bool = True) -> go.Figure:
    """Build an equal-axis trajectory plot, optionally with animation controls."""
    r1, r2 = result["r1"], result["r2"]
    all_positions = np.vstack((r1, r2))
    span = max(float(np.ptp(all_positions[:, 0])), float(np.ptp(all_positions[:, 1])), 0.25)
    center = np.mean([np.min(all_positions, axis=0), np.max(all_positions, axis=0)], axis=0)
    margin = 0.12 * span
    axis_range_x = [center[0] - span / 2 - margin, center[0] + span / 2 + margin]
    axis_range_y = [center[1] - span / 2 - margin, center[1] + span / 2 + margin]
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=r1[:, 0], y=r1[:, 1], mode="lines", name="Body 1 path",
                                line={"color": "#38bdf8", "width": 2}))
    figure.add_trace(go.Scatter(x=r2[:, 0], y=r2[:, 1], mode="lines", name="Body 2 path",
                                line={"color": "#fb7185", "width": 2}))
    figure.add_trace(go.Scatter(x=[r1[0, 0]], y=[r1[0, 1]], mode="markers", name="Body 1",
                                marker={"color": "#38bdf8", "size": 15}))
    figure.add_trace(go.Scatter(x=[r2[0, 0]], y=[r2[0, 1]], mode="markers", name="Body 2",
                                marker={"color": "#fb7185", "size": 15}))
    if animated:
        indices = _sample_indices(len(result["time"]))
        figure.frames = [go.Frame(name=str(index), data=[
            go.Scatter(x=r1[:index + 1, 0], y=r1[:index + 1, 1]),
            go.Scatter(x=r2[:index + 1, 0], y=r2[:index + 1, 1]),
            go.Scatter(x=[r1[index, 0]], y=[r1[index, 1]]),
            go.Scatter(x=[r2[index, 0]], y=[r2[index, 1]]),
        ], traces=[0, 1, 2, 3]) for index in indices]
        steps = [{"method": "animate", "label": f"{result['time'][i]:.2f}",
                  "args": [[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": False},
                                      "transition": {"duration": 0}}]} for i in indices]
        figure.update_layout(
            updatemenus=[{"type": "buttons", "direction": "left", "x": 0.0, "y": 1.13,
                          "buttons": [
                              {"label": "Play", "method": "animate",
                               "args": [None, {"fromcurrent": True, "frame": {"duration": 35, "redraw": False},
                                               "transition": {"duration": 0}}]},
                              {"label": "Pause", "method": "animate",
                               "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]},
                          ]}],
            sliders=[{"active": 0, "currentvalue": {"prefix": "Time: "}, "steps": steps}],
        )
    figure.update_layout(title=f"Two-body trajectory — {result['classification']}",
                         template="plotly_dark", height=650, hovermode="closest",
                         xaxis={"title": "x", "range": axis_range_x, "scaleanchor": "y", "scaleratio": 1},
                         yaxis={"title": "y", "range": axis_range_y}, legend={"orientation": "h"})
    return figure


def diagnostics_figure(result: dict) -> go.Figure:
    """Build position, speed, separation, and conservation plots."""
    t = result["time"]
    figure = make_subplots(rows=2, cols=2,
        subplot_titles=("Position components", "Speed", "Separation", "Conserved quantities"))
    for values, name, color, dash in (
        (result["r1"][:, 0], "Body 1 x", "#38bdf8", "solid"),
        (result["r1"][:, 1], "Body 1 y", "#38bdf8", "dot"),
        (result["r2"][:, 0], "Body 2 x", "#fb7185", "solid"),
        (result["r2"][:, 1], "Body 2 y", "#fb7185", "dot")):
        figure.add_trace(go.Scatter(x=t, y=values, name=name, line={"color": color, "dash": dash}), row=1, col=1)
    figure.add_trace(go.Scatter(x=t, y=result["speed1"], name="Body 1 speed", line={"color": "#38bdf8"}), row=1, col=2)
    figure.add_trace(go.Scatter(x=t, y=result["speed2"], name="Body 2 speed", line={"color": "#fb7185"}), row=1, col=2)
    figure.add_trace(go.Scatter(x=t, y=result["separation"], name="Separation", line={"color": "#a78bfa"}), row=2, col=1)
    energy0 = result["total_energy"][0]
    energy_drift = (result["total_energy"] - energy0) / max(abs(energy0), np.finfo(float).eps)
    angular0 = result["angular_momentum"][0]
    angular_drift = (result["angular_momentum"] - angular0) / max(abs(angular0), np.finfo(float).eps)
    figure.add_trace(go.Scatter(x=t, y=energy_drift, name="Relative energy drift", line={"color": "#fbbf24"}), row=2, col=2)
    figure.add_trace(go.Scatter(x=t, y=angular_drift, name="Relative angular momentum drift", line={"color": "#34d399"}), row=2, col=2)
    figure.update_xaxes(title_text="Time")
    figure.update_yaxes(title_text="Position", row=1, col=1)
    figure.update_yaxes(title_text="Speed", row=1, col=2)
    figure.update_yaxes(title_text="Distance", row=2, col=1)
    figure.update_yaxes(title_text="Relative drift", row=2, col=2)
    figure.update_layout(template="plotly_dark", height=750, title="Simulation diagnostics", hovermode="x unified")
    return figure


def export_results(result: dict, output_directory="exports", prefix="kepler_run") -> dict:
    """Save data, metadata, figures, and a self-contained interactive report."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = output / f"{prefix}_{timestamp}"
    dataframe = to_dataframe(result)
    summary = conservation_summary(result)
    trajectory = trajectory_figure(result, animated=True)
    diagnostics_plot = diagnostics_figure(result)
    csv_path = stem.with_suffix(".csv")
    json_path = stem.with_suffix(".json")
    html_path = stem.with_suffix(".html")
    trajectory_html_path = stem.parent / f"{stem.name}_trajectory.html"
    dataframe.to_csv(csv_path, index=False)
    metadata = {"created_utc": timestamp, "config": result["config"], "summary": summary}
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    trajectory.write_html(trajectory_html_path, include_plotlyjs=True)
    trajectory_fragment = trajectory.to_html(full_html=False, include_plotlyjs=True)
    diagnostics_fragment = diagnostics_plot.to_html(full_html=False, include_plotlyjs=False)
    summary_rows = "".join(f"<tr><th>{key}</th><td>{value}</td></tr>" for key, value in summary.items())
    report = f"""<!doctype html><html><head><meta charset='utf-8'><title>Kepler report</title>
    <style>body{{font-family:system-ui;background:#111827;color:#e5e7eb;margin:2rem}}th,td{{padding:.4rem 1rem;text-align:left}}table{{border-collapse:collapse;background:#1f2937}}</style>
    </head><body><h1>Kepler simulation report</h1><p>Generated {timestamp}</p>
    <table>{summary_rows}</table>{trajectory_fragment}{diagnostics_fragment}</body></html>"""
    html_path.write_text(report, encoding="utf-8")
    files = {"csv": str(csv_path), "metadata_json": str(json_path),
             "interactive_trajectory": str(trajectory_html_path), "html_report": str(html_path)}
    try:
        image_path = stem.parent / f"{stem.name}_trajectory.png"
        trajectory_figure(result, animated=False).write_image(image_path, width=1200, height=900, scale=2)
        files["png"] = str(image_path)
    except Exception:
        # Kaleido needs a system browser in recent releases. Matplotlib provides
        # a browser-independent fallback while Plotly remains the interactive UI.
        import matplotlib.pyplot as plt

        image_path = stem.parent / f"{stem.name}_trajectory.png"
        figure, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
        axis.plot(result["r1"][:, 0], result["r1"][:, 1], color="#0284c7", label="Body 1")
        axis.plot(result["r2"][:, 0], result["r2"][:, 1], color="#e11d48", label="Body 2")
        axis.scatter(*result["r1"][-1], color="#0284c7", s=70)
        axis.scatter(*result["r2"][-1], color="#e11d48", s=70)
        axis.set(title=f"Two-body trajectory — {result['classification']}", xlabel="x", ylabel="y")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.savefig(image_path, dpi=180)
        plt.close(figure)
        files["png"] = str(image_path)
    return files
