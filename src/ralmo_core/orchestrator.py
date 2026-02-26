"""Main orchestrator for RALMO inference pipeline.

Ties together the draft model, target model, speculative engine,
policy module, KV manager, and logging into a unified interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig

from ralmo_core.kv_manager import KVManager
from ralmo_core.models.draft_model import DraftModel
from ralmo_core.models.target_model import TargetModel
from ralmo_core.policy import AdaptivePolicy, BasePolicy, StaticPolicy
from ralmo_core.speculative import SpeculativeEngine, SpeculativeResult
from ralmo_core.utils.logger import RALMOLogger
from ralmo_core.utils.power_monitor import PowerMonitor

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationResult:
    """Result of an orchestration run.

    Attributes:
        text: The generated text output.
        tokens: Generated token IDs.
        stats: Speculative decoding statistics.
        energy_joules: Estimated energy consumption (0.0 if not measured).
        finish_reason: Why generation stopped.
    """

    text: str
    tokens: list[int]
    stats: Any
    energy_joules: float
    finish_reason: str


class Orchestrator:
    """High-level orchestration engine for RALMO.

    Manages the full inference lifecycle:
    1. Initialize models from configuration
    2. Set up the speculative engine with policy and KV manager
    3. Process inference requests
    4. Log results and metrics

    Usage:
        orch = Orchestrator(cfg)
        orch.initialize()
        result = orch.generate("What is the meaning of life?")
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize orchestrator with Hydra configuration.

        Args:
            cfg: Hydra DictConfig with all hyperparameters.
        """
        self.cfg = cfg
        self.draft_model: DraftModel | None = None
        self.target_model: TargetModel | None = None
        self.engine: SpeculativeEngine | None = None
        self.policy: BasePolicy | None = None
        self.kv_manager: KVManager | None = None
        self.ralmo_logger: RALMOLogger | None = None
        self.power_monitor: PowerMonitor | None = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize all components from configuration.

        Loads models, creates the speculative engine, and sets up logging.
        Must be called before generate().
        """
        logger.info("Initializing RALMO Orchestrator...")

        # Initialize policy
        policy_type = self.cfg.speculative.get("policy_type", "static")
        k = self.cfg.speculative.k

        if policy_type == "adaptive":
            cfg_adapt = self.cfg.speculative.get("adaptive", {})
            self.policy = AdaptivePolicy(
                tau_0=cfg_adapt.get("tau_0", -0.7),
                alpha=cfg_adapt.get("alpha", 0.1),
                h_0=cfg_adapt.get("h_0", 1.0),
                k=k,
            )
            logger.info(
                "Policy: AdaptivePolicy(tau_0=%.2f, alpha=%.2f, h_0=%.2f, k=%d)",
                cfg_adapt.get("tau_0", -0.7),
                cfg_adapt.get("alpha", 0.1),
                cfg_adapt.get("h_0", 1.0),
                k,
            )
        else:
            self.policy = StaticPolicy(
                tau=self.cfg.speculative.get("tau", -0.7),
                k=k,
            )
            logger.info(
                "Policy: StaticPolicy(tau=%.2f, k=%d)",
                self.policy.get_tau(),
                k,
            )

        # Initialize KV manager
        self.kv_manager = KVManager(
            use_native=self.cfg.kv_cache.get("use_native", True),
        )

        # Initialize models
        self.draft_model = DraftModel()
        self.draft_model.load(
            model_path=self.cfg.draft.model_path,
            n_ctx=self.cfg.draft.n_ctx,
            n_gpu_layers=self.cfg.draft.n_gpu_layers,
            seed=self.cfg.draft.seed,
        )

        self.target_model = TargetModel()
        self.target_model.load(
            model_path=self.cfg.target.model_path,
            n_ctx=self.cfg.target.n_ctx,
            n_gpu_layers=self.cfg.target.n_gpu_layers,
            seed=self.cfg.target.seed,
        )

        # Initialize speculative engine
        self.engine = SpeculativeEngine(
            draft_model=self.draft_model,
            target_model=self.target_model,
            policy=self.policy,
            kv_manager=self.kv_manager,
        )

        # Initialize logging
        self.ralmo_logger = RALMOLogger(
            output_dir=self.cfg.logging.output_dir,
            log_format=self.cfg.logging.format,
            verbose=self.cfg.logging.get("verbose", False),
        )

        # Initialize power monitoring
        self.power_monitor = PowerMonitor(
            enabled=self.cfg.power.get("enabled", False),
            backend=self.cfg.power.get("backend", "stub"),
        )

        self._initialized = True
        logger.info("RALMO Orchestrator initialized successfully.")

    def generate(self, prompt: str, max_tokens: int | None = None) -> OrchestrationResult:
        """Generate text using speculative decoding.

        Args:
            prompt: Input text prompt.
            max_tokens: Override for maximum output tokens. If None, uses config value.

        Returns:
            OrchestrationResult with generated text, stats, and metrics.

        Raises:
            RuntimeError: If orchestrator is not initialized.
        """
        if not self._initialized or self.engine is None or self.draft_model is None:
            raise RuntimeError("Orchestrator not initialized. Call initialize() first.")

        if max_tokens is None:
            max_tokens = self.cfg.speculative.max_tokens

        # Tokenize prompt
        prompt_tokens = self.draft_model.tokenize(prompt)
        logger.info("Prompt: %d tokens", len(prompt_tokens))

        # Start power measurement
        assert self.power_monitor is not None
        self.power_monitor.start_measurement()

        # Run speculative decoding
        result: SpeculativeResult = self.engine.run(
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            draft_temperature=self.cfg.draft.temperature,
        )

        # Stop power measurement
        energy = self.power_monitor.stop_measurement()

        # Log the result
        if self.ralmo_logger is not None:
            self.ralmo_logger.log_request(
                prompt=prompt,
                result=result,
                energy_joules=energy,
            )

        return OrchestrationResult(
            text=result.text,
            tokens=result.tokens,
            stats=result.stats,
            energy_joules=energy,
            finish_reason=result.finish_reason,
        )

    def shutdown(self) -> None:
        """Clean up resources."""
        logger.info("Shutting down RALMO Orchestrator...")
        if self.draft_model is not None:
            self.draft_model.reset()
        if self.target_model is not None:
            self.target_model.reset()
        if self.kv_manager is not None:
            self.kv_manager.clear()
        if self.ralmo_logger is not None:
            self.ralmo_logger.close()
        self._initialized = False
        logger.info("Orchestrator shut down.")
