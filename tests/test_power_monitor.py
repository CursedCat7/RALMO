"""Tests for Power Monitor and RAPL backend."""

from __future__ import annotations

from unittest.mock import patch

from ralmo_core.utils.power_monitor import PowerMonitor
from ralmo_core.utils.rapl_backend import RAPLBackend


class TestRAPLBackend:
    """Tests for RAPL backend."""

    def test_not_available_on_non_linux(self) -> None:
        """RAPL should not be available on macOS/Windows."""
        with patch("ralmo_core.utils.rapl_backend.platform.system", return_value="Darwin"):
            rapl = RAPLBackend()
            assert rapl.available is False

    def test_compute_delta_normal(self) -> None:
        """Normal energy delta computation."""
        rapl = RAPLBackend.__new__(RAPLBackend)
        rapl._max_energy_uj = 1_000_000_000
        result = rapl.compute_delta_joules(100_000, 500_000)
        assert abs(result - 0.4) < 1e-6

    def test_compute_delta_overflow(self) -> None:
        """Overflow handling in energy delta."""
        rapl = RAPLBackend.__new__(RAPLBackend)
        rapl._max_energy_uj = 1_000_000  # 1 Joule max
        # start=900000, end=100000 → overflow
        result = rapl.compute_delta_joules(900_000, 100_000)
        # delta = (1000000 - 900000) + 100000 = 200000 uJ = 0.2 J
        assert abs(result - 0.2) < 1e-6


class TestPowerMonitor:
    """Tests for PowerMonitor."""

    def test_stub_returns_zero(self) -> None:
        """Stub backend returns 0.0 energy."""
        pm = PowerMonitor(enabled=True, backend="stub")
        pm.start_measurement()
        energy = pm.stop_measurement()
        assert energy == 0.0

    def test_disabled_returns_zero(self) -> None:
        """Disabled monitor returns 0.0."""
        pm = PowerMonitor(enabled=False)
        pm.start_measurement()
        assert pm.stop_measurement() == 0.0

    def test_rapl_fallback_on_non_linux(self) -> None:
        """RAPL backend falls back to stub on non-Linux."""
        pm = PowerMonitor(enabled=True, backend="rapl")
        # On macOS this should fall back to stub
        pm.start_measurement()
        energy = pm.stop_measurement()
        assert energy == 0.0
