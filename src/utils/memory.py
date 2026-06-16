"""
RiskLens AI — Memory Profiler
===============================
Tracks memory usage at each pipeline stage.

Why memory profiling matters:
    On a 16 GB laptop, loading 2.2M rows naively uses 3-5 GB.
    This profiler catches memory spikes before they crash the pipeline.
"""

import psutil
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class StageMetrics:
    """Memory metrics for a single pipeline stage."""
    name: str
    start_mb: float
    end_mb: float
    delta_mb: float
    duration_s: float


class MemoryProfiler:
    """Track memory usage across pipeline stages."""

    def __init__(self):
        self.stages: list[StageMetrics] = []
        self.peak_mb: float = 0.0

    @staticmethod
    def get_current_mb() -> float:
        """Get current process memory in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 2)

    @contextmanager
    def track(self, stage_name: str):
        """Context manager to track a pipeline stage."""
        start_mb = self.get_current_mb()
        start_time = time.time()

        yield

        end_mb = self.get_current_mb()
        duration = time.time() - start_time
        delta = end_mb - start_mb

        if end_mb > self.peak_mb:
            self.peak_mb = end_mb

        metrics = StageMetrics(
            name=stage_name,
            start_mb=round(start_mb, 1),
            end_mb=round(end_mb, 1),
            delta_mb=round(delta, 1),
            duration_s=round(duration, 3),
        )
        self.stages.append(metrics)

    def print_report(self) -> None:
        """Print memory usage report."""
        print("\n" + "=" * 60)
        print("  MEMORY PROFILING REPORT")
        print("=" * 60)
        print(f"  {'Stage':<30} {'Start':>8} {'End':>8} {'Delta':>8} {'Time':>8}")
        print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        for s in self.stages:
            sign = "+" if s.delta_mb >= 0 else ""
            print(
                f"  {s.name:<30} {s.start_mb:>7.1f}M {s.end_mb:>7.1f}M "
                f"{sign}{s.delta_mb:>6.1f}M {s.duration_s:>7.1f}s"
            )

        print(f"\n  Peak Memory: {self.peak_mb:.1f} MB")
        print("=" * 60 + "\n")


def get_system_memory_info() -> dict:
    """Get system memory information."""
    mem = psutil.virtual_memory()
    return {
        "total_gb": mem.total / (1024 ** 3),
        "available_gb": mem.available / (1024 ** 3),
        "used_pct": mem.percent,
    }
