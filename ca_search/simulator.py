from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any


def _load_numpy():
    import numpy as np

    return np


@dataclass(frozen=True)
class MetricSeries:
    density: list[float]
    entropy: list[float]
    activity: list[float]


@dataclass(frozen=True)
class BenchmarkResult:
    backend: str
    batch: int
    width: int
    height: int
    steps: int
    elapsed_seconds: float
    cells_updated_per_second: float
    rule_steps_per_second: float


class NumpyBackend:
    name = "numpy"

    def __init__(self) -> None:
        self.np = _load_numpy()

    def asarray(self, value: Any, dtype: str | None = None):
        return self.np.asarray(value, dtype=dtype)

    def to_numpy(self, value):
        return self.np.asarray(value)

    def step_pairwise(self, states, rules):
        np = self.np
        states = np.asarray(states, dtype=np.uint8)
        rules = np.asarray(rules, dtype=np.uint8)
        x = np.roll(np.roll(states, 1, axis=1), 1, axis=2)
        y = np.roll(states, 1, axis=1)
        z = np.roll(np.roll(states, 1, axis=1), -1, axis=2)
        t = np.roll(states, 1, axis=2)
        u = states
        w = np.roll(states, -1, axis=2)
        a = np.roll(np.roll(states, -1, axis=1), 1, axis=2)
        b = np.roll(states, -1, axis=1)
        c = np.roll(np.roll(states, -1, axis=1), -1, axis=2)
        indices = (
            (x.astype(np.uint16) << 8)
            | (y.astype(np.uint16) << 7)
            | (z.astype(np.uint16) << 6)
            | (t.astype(np.uint16) << 5)
            | (u.astype(np.uint16) << 4)
            | (w.astype(np.uint16) << 3)
            | (a.astype(np.uint16) << 2)
            | (b.astype(np.uint16) << 1)
            | c.astype(np.uint16)
        )
        gathered = np.take_along_axis(rules[:, None, None, :], indices[..., None], axis=3)
        return gathered[..., 0].astype(np.uint8, copy=False)


class MLXBackend:
    name = "mlx"

    def __init__(self) -> None:
        import mlx.core as mx

        self.mx = mx
        self._step_eager = self._build_step_impl()
        self._step_impl = self._compile_step()

    def _build_step_impl(self):
        mx = self.mx

        def step(states, rules):
            x = mx.roll(mx.roll(states, 1, axis=1), 1, axis=2)
            y = mx.roll(states, 1, axis=1)
            z = mx.roll(mx.roll(states, 1, axis=1), -1, axis=2)
            t = mx.roll(states, 1, axis=2)
            u = states
            w = mx.roll(states, -1, axis=2)
            a = mx.roll(mx.roll(states, -1, axis=1), 1, axis=2)
            b = mx.roll(states, -1, axis=1)
            c = mx.roll(mx.roll(states, -1, axis=1), -1, axis=2)
            indices = (
                (x.astype(mx.uint16) << 8)
                | (y.astype(mx.uint16) << 7)
                | (z.astype(mx.uint16) << 6)
                | (t.astype(mx.uint16) << 5)
                | (u.astype(mx.uint16) << 4)
                | (w.astype(mx.uint16) << 3)
                | (a.astype(mx.uint16) << 2)
                | (b.astype(mx.uint16) << 1)
                | c.astype(mx.uint16)
            )
            gathered = mx.take_along_axis(rules[:, None, None, :], indices[..., None], axis=3)
            return mx.squeeze(gathered, axis=3).astype(mx.uint8)

        return step

    def _compile_step(self):
        mx = self.mx
        step = self._step_eager
        return mx.compile(step, shapeless=True)

    def asarray(self, value: Any, dtype: str | None = None):
        if dtype is None:
            return self.mx.array(value)
        return self.mx.array(value, dtype=getattr(self.mx, dtype))

    def to_numpy(self, value):
        np = _load_numpy()
        return np.asarray(value)

    def step_pairwise(self, states, rules):
        try:
            return self._step_impl(states, rules)
        except ValueError as exc:
            # Some MLX builds reject the compiled gather graph for this step
            # shape pattern. Fall back to the eager kernel instead of failing.
            if "output_shapes" not in str(exc):
                raise
            return self._step_eager(states, rules)


def create_backend(name: str = "auto"):
    if name == "numpy":
        return NumpyBackend()
    if name == "mlx":
        return MLXBackend()
    if name != "auto":
        raise ValueError(f"Unknown backend: {name}")
    # MLX can abort the Python process during Metal device initialization in
    # restricted or headless environments, so "auto" stays on the verified
    # NumPy path unless the user explicitly requests MLX.
    return NumpyBackend()


def _binary_entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -(probability * math.log2(probability) + (1.0 - probability) * math.log2(1.0 - probability))


def collect_metric_series(backend_name: str, rules, initial_states, steps: int) -> MetricSeries:
    backend = create_backend(backend_name)
    np = _load_numpy()
    states = backend.asarray(initial_states, dtype="uint8")
    rules_array = backend.asarray(rules, dtype="uint8")

    density: list[float] = []
    entropy: list[float] = []
    activity: list[float] = []
    previous = backend.to_numpy(states)
    for _ in range(steps):
        states = backend.step_pairwise(states, rules_array)
        current = backend.to_numpy(states)
        p = float(current.mean())
        density.append(p)
        entropy.append(_binary_entropy(p))
        activity.append(float(np.not_equal(current, previous).mean()))
        previous = current
    return MetricSeries(density=density, entropy=entropy, activity=activity)


def benchmark_pairwise(backend_name: str, rules, initial_states, steps: int) -> BenchmarkResult:
    backend = create_backend(backend_name)
    rules_array = backend.asarray(rules, dtype="uint8")
    states = backend.asarray(initial_states, dtype="uint8")
    batch, height, width = states.shape

    start = time.perf_counter()
    for _ in range(steps):
        states = backend.step_pairwise(states, rules_array)
    if backend.name == "mlx":
        backend.mx.eval(states)
        backend.mx.synchronize()
    elapsed = time.perf_counter() - start
    total_cell_updates = batch * width * height * steps
    return BenchmarkResult(
        backend=backend.name,
        batch=batch,
        width=width,
        height=height,
        steps=steps,
        elapsed_seconds=elapsed,
        cells_updated_per_second=total_cell_updates / elapsed,
        rule_steps_per_second=(batch * steps) / elapsed,
    )
