"""Power monitoring stub for RALMO.

Provides a measurement interface for energy consumption tracking.
The stub implementation returns zeros; real backends (RAPL, powermetrics)
will be added in Phase 2.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class PowerMonitor:
    """Energy measurement interface for RALMO inference.

    Provides start/stop measurement pairs with configurable backends.
    The stub backend always returns 0.0J — designed for interface stability
    while real energy measurement is implemented later.

    Supported backends:
        - 'stub': No-op, returns 0.0 (default)
        - 'rapl': Intel RAPL via /sys/class/powercap (Linux, future)
        - 'powermetrics': macOS powermetrics (future)

    Attributes:
        enabled: Whether power monitoring is active.
        backend: The measurement backend identifier.
    """

    def __init__(self, enabled: bool = False, backend: str = "stub") -> None:
        """Initialize power monitor.

        Args:
            enabled: Whether to enable power monitoring.
            backend: Measurement backend ('stub', 'rapl').
        """
        self.enabled = enabled
        self.backend = backend
        self._start_time: float | None = None
        self._is_measuring = False
        self._rapl: Any = None
        self._rapl_start_uj: int = 0

        if enabled and backend == "rapl":
            self._init_rapl()

    def _init_rapl(self) -> None:
        """Initialize RAPL backend if available."""
        try:
            from ralmo_core.utils.rapl_backend import RAPLBackend

            rapl = RAPLBackend()
            if rapl.available:
                self._rapl = rapl
                logger.info("RAPL power monitoring enabled.")
            else:
                logger.warning(
                    "RAPL not available on this system. "
                    "Falling back to stub."
                )
                self.backend = "stub"
        except Exception as e:
            logger.warning("Failed to init RAPL: %s. Using stub.", e)
            self.backend = "stub"

    def start_measurement(self) -> None:
        """Begin an energy measurement interval."""
        if not self.enabled:
            return

        self._start_time = time.perf_counter()
        self._is_measuring = True

        if self.backend == "rapl" and self._rapl is not None:
            self._rapl_start_uj = self._rapl.read_energy_uj()

        logger.debug("Power measurement started (backend=%s)", self.backend)

    def stop_measurement(self) -> float:
        """End the measurement interval and return estimated energy.

        Returns:
            Estimated energy consumption in Joules.
            Returns 0.0 for the stub backend.
        """
        if not self.enabled or not self._is_measuring:
            return 0.0

        self._is_measuring = False
        elapsed = time.perf_counter() - (self._start_time or 0.0)
        self._start_time = None

        if self.backend == "rapl" and self._rapl is not None:
            end_uj = self._rapl.read_energy_uj()
            energy_j = self._rapl.compute_delta_joules(
                self._rapl_start_uj, end_uj
            )
            logger.debug(
                "Power measurement stopped (RAPL): "
                "elapsed=%.3fs, energy=%.6fJ",
                elapsed,
                energy_j,
            )
            return float(energy_j)

        # Stub: return 0.0 Joules
        logger.debug(
            "Power measurement stopped (stub): "
            "elapsed=%.3fs, energy=0.0J",
            elapsed,
        )
        return 0.0

    def get_energy_joules(self) -> float:
        """Get the energy consumed since the last start_measurement call.

        Returns:
            Estimated energy in Joules. Always 0.0 for stub backend.
        """
        if self.backend == "rapl" and self._rapl is not None:
            current_uj = self._rapl.read_energy_uj()
            return float(
                self._rapl.compute_delta_joules(
                    self._rapl_start_uj, current_uj
                )
            )
        return 0.0

    @property
    def is_available(self) -> bool:
        """Check if a real power monitoring backend is available."""
        if self.backend == "rapl":
            return self._rapl is not None
        return self.enabled and self.backend != "stub"
