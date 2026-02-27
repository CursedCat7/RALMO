"""Main orchestrator for RALMO inference pipeline.

Ties together the draft model, target model, speculative engine,
policy module, KV manager, and logging into a unified interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig

from ralmo_core.draft_selector import (
    BestOfNSelector,
    DraftSelector,
    MajorityVoteSelector,
)
from ralmo_core.external.verifier_adapter import (
    ExternalVerifier,
    VerificationResult,
    create_verifier,
)
from ralmo_core.kv_manager import KVManager
from ralmo_core.models.draft_model import DraftModel
from ralmo_core.models.target_model import TargetModel
from ralmo_core.multi_draft_engine import MultiDraftEngine
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
    escalated: bool = False
    verifier_result: Any | None = None


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
        self.draft_models: list[DraftModel] = []
        self.target_model: TargetModel | None = None
        self.engine: SpeculativeEngine | MultiDraftEngine | None = None
        self.policy: BasePolicy | None = None
        self.kv_manager: KVManager | None = None
        self.ralmo_logger: RALMOLogger | None = None
        self.power_monitor: PowerMonitor | None = None
        self.verifier: ExternalVerifier | None = None
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
        multi_draft_cfg = self.cfg.get("multi_draft", {})
        use_multi_draft = multi_draft_cfg.get("enabled", False)

        if use_multi_draft:
            model_cfgs = multi_draft_cfg.get("models", [])
            if not model_cfgs:
                logger.warning("multi_draft enabled but no models. Using single draft.")
                use_multi_draft = False

        if use_multi_draft:
            for mcfg in model_cfgs:
                dm = DraftModel()
                dm.load(
                    model_path=mcfg.get("path", self.cfg.draft.model_path),
                    n_ctx=mcfg.get("n_ctx", self.cfg.draft.n_ctx),
                    n_gpu_layers=mcfg.get("n_gpu_layers", self.cfg.draft.n_gpu_layers),
                    seed=self.cfg.draft.get("seed", 42),
                )
                self.draft_models.append(dm)
            self.draft_model = self.draft_models[0]
            logger.info("Loaded %d draft models for multi-draft.", len(self.draft_models))

            strategy = multi_draft_cfg.get("strategy", "best_of_n")
            selector: DraftSelector = (
                MajorityVoteSelector()
                if strategy == "majority_vote"
                else BestOfNSelector()
            )

            self.target_model = TargetModel()
            self.target_model.load(
                model_path=self.cfg.target.model_path,
                n_ctx=self.cfg.target.n_ctx,
                n_gpu_layers=self.cfg.target.n_gpu_layers,
                seed=self.cfg.target.seed,
            )

            self.engine = MultiDraftEngine(
                draft_models=self.draft_models,  # type: ignore[arg-type]
                target_model=self.target_model,  # type: ignore[arg-type]
                policy=self.policy,
                kv_manager=self.kv_manager,
                selector=selector,
            )
            logger.info("MultiDraftEngine initialized (strategy=%s).", strategy)
        else:
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

        # Initialize external verifier
        ext_cfg = self.cfg.get("external", {})
        self.verifier = create_verifier(
            provider=ext_cfg.get("provider", "stub"),
            api_key=ext_cfg.get("api_key", ""),
            model=ext_cfg.get("model", ""),
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

        # Check if escalation is needed
        escalated = False
        verifier_result: VerificationResult | None = None
        escalation_threshold = self.cfg.get("external", {}).get(
            "escalation_threshold", 0.3
        )
        if (
            result.stats.acceptance_rate < escalation_threshold
            and self.verifier is not None
            and self.verifier.is_available()
        ):
            logger.info(
                "Low acceptance rate (%.1f%%). Escalating to external verifier.",
                result.stats.acceptance_rate * 100,
            )
            verifier_result = self.verifier.verify(prompt, result.text)
            escalated = True

        # Log the result
        if self.ralmo_logger is not None:
            self.ralmo_logger.log_request(
                prompt=prompt,
                result=result,
                energy_joules=energy,
            )

        final_text = result.text
        if (
            escalated
            and verifier_result
            and not verifier_result.verified
            and verifier_result.alternative_text
        ):
            final_text = verifier_result.alternative_text

        return OrchestrationResult(
            text=final_text,
            tokens=result.tokens,
            stats=result.stats,
            energy_joules=energy,
            finish_reason=result.finish_reason,
            escalated=escalated,
            verifier_result=verifier_result,
        )

    def shutdown(self) -> None:
        """Clean up resources."""
        logger.info("Shutting down RALMO Orchestrator...")
        # Clean up all draft models (multi-draft mode)
        if self.draft_models:
            for dm in self.draft_models:
                dm.reset()
        elif self.draft_model is not None:
            self.draft_model.reset()
        if self.target_model is not None:
            self.target_model.reset()
        if self.kv_manager is not None:
            self.kv_manager.clear()
        if self.ralmo_logger is not None:
            self.ralmo_logger.close()
        self._initialized = False
        logger.info("Orchestrator shut down.")
