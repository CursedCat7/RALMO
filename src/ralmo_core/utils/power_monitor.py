"""Power monitoring stub for RALMO.

Provides a measurement interface for energy consumption tracking.
The stub implementation returns zeros; real backends (RAPL, powermetrics)
will be added in Phase 2.
"""

from __future__ import annotations

import logging
import time

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
            backend: Measurement backend ('stub', 'rapl', 'powermetrics').
        """
        self.enabled = enabled
        self.backend = backend
        self._start_time: float | None = None
        self._is_measuring = False

        if enabled and backend != "stub":
            logger.warning(
                "Power monitoring backend '%s' not yet implemented. Using stub.",
                backend,
            )

    def start_measurement(self) -> None:
        """Begin an energy measurement interval.

        Can be called even when disabled (no-op).
        """
        if not self.enabled:
            return

        self._start_time = time.perf_counter()
        self._is_measuring = True
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

        if self.backend == "stub":
            # Stub: return 0.0 Joules
            logger.debug("Power measurement stopped (stub): elapsed=%.3fs, energy=0.0J", elapsed)
            return 0.0

        # Future backends will compute actual energy here
        logger.debug("Power measurement stopped: elapsed=%.3fs", elapsed)
        return 0.0

    def get_energy_joules(self) -> float:
        """Get the energy consumed since the last start_measurement call.

        Returns:
            Estimated energy in Joules. Always 0.0 for stub backend.
        """
        if self.backend == "stub":
            return 0.0
        return 0.0

    @property
    def is_available(self) -> bool:
        """Check if a real power monitoring backend is available."""
        return self.enabled and self.backend != "stub"
