"""RAPL (Running Average Power Limit) energy measurement backend.

Reads energy counters from the Linux sysfs powercap interface at
/sys/class/powercap/intel-rapl/ to measure real CPU package energy
consumption in Joules.
"""

from __future__ import annotations

import logging
import platform
from pathlib import Path

logger = logging.getLogger(__name__)

# Default RAPL sysfs path (can be overridden for testing)
DEFAULT_RAPL_PATH = "/sys/class/powercap/intel-rapl"


class RAPLBackend:
    """Linux RAPL energy measurement backend.

    Reads microjoule counters from the powercap sysfs interface
    to compute actual CPU energy consumption.

    Attributes:
        available: Whether RAPL counters are readable.
    """

    def __init__(self, rapl_path: str = DEFAULT_RAPL_PATH) -> None:
        """Initialize RAPL backend.

        Args:
            rapl_path: Path to the RAPL sysfs directory.
        """
        self._rapl_path = Path(rapl_path)
        self._energy_file: Path | None = None
        self._max_energy_uj: int = 0
        self._available = False

        self._detect()

    def _detect(self) -> None:
        """Detect RAPL availability and locate energy counter."""
        if platform.system() != "Linux":
            logger.info("RAPL not available: not running on Linux.")
            return

        # Search for the first package energy counter
        # Typical path: /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
        for subdir in sorted(self._rapl_path.glob("intel-rapl:*")):
            energy_file = subdir / "energy_uj"
            max_file = subdir / "max_energy_range_uj"

            if energy_file.exists() and energy_file.is_file():
                try:
                    energy_file.read_text()
                    self._energy_file = energy_file

                    if max_file.exists():
                        self._max_energy_uj = int(
                            max_file.read_text().strip()
                        )
                    else:
                        # Default to a large value if max not available
                        self._max_energy_uj = 2**63

                    self._available = True
                    logger.info(
                        "RAPL detected: %s (max=%d uJ)",
                        energy_file,
                        self._max_energy_uj,
                    )
                    return
                except (PermissionError, OSError) as e:
                    logger.warning(
                        "RAPL file exists but cannot be read: %s (%s)",
                        energy_file,
                        e,
                    )

        logger.info("RAPL not available: no readable energy counters found.")

    @property
    def available(self) -> bool:
        """Whether RAPL energy measurement is available."""
        return self._available

    def read_energy_uj(self) -> int:
        """Read the current energy counter in microjoules.

        Returns:
            Current energy counter value in microjoules.

        Raises:
            RuntimeError: If RAPL is not available.
        """
        if not self._available or self._energy_file is None:
            raise RuntimeError("RAPL is not available.")

        return int(self._energy_file.read_text().strip())

    def compute_delta_joules(self, start_uj: int, end_uj: int) -> float:
        """Compute energy delta in Joules, handling counter overflow.

        The RAPL energy counter wraps around at max_energy_range_uj.
        This method correctly handles that overflow.

        Args:
            start_uj: Energy counter at start (microjoules).
            end_uj: Energy counter at end (microjoules).

        Returns:
            Energy consumed in Joules.
        """
        if end_uj >= start_uj:
            delta_uj = end_uj - start_uj
        else:
            # Counter overflow
            delta_uj = (self._max_energy_uj - start_uj) + end_uj

        return delta_uj / 1_000_000.0
