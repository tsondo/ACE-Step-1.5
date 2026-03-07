"""V2 (corrected) training start route -- uses FixedLoRATrainer.

Replaces the V1 training routes with the corrected training loop that uses:
- Continuous logit-normal timestep sampling (matching model.forward())
- CFG dropout during training (cfg_ratio=0.15)
- Lightning Fabric for mixed precision and gradient scaling

Supports both LoRA (PEFT) and LoKR (LyCORIS) via the adapter_type field.
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from loguru import logger

from acestep.api.train_api_models import (
    StartLoKRTrainingRequest,
    StartTrainingRequest,
    initialize_training_state,
)
from acestep.api.train_api_runtime import RuntimeComponentManager, unwrap_module
from acestep.handler import AceStepHandler


# ---------------------------------------------------------------------------
# Shared runner thread -- identical logic for LoRA and LoKR
# ---------------------------------------------------------------------------

def _make_runner(
    trainer: Any,
    training_state: Dict[str, Any],
    run_id: str,
    handler: AceStepHandler,
) -> Callable[[], None]:
    """Build the background thread target for a V2 training run."""

    def _runner() -> None:
        local_run_id = run_id
        try:
            for step, loss, status in trainer.train(training_state):
                if training_state.get("run_id") != local_run_id:
                    break
                training_state["current_step"] = step
                training_state["current_loss"] = loss
                training_state["status"] = status
                text = str(status)
                match = re.search(r"Epoch (\d+)/(\d+)", text)
                if match:
                    training_state["current_epoch"] = int(match.group(1))
                if loss is not None and loss == loss and step > 0:
                    history = training_state.get("loss_history", [])
                    history.append({"step": step, "loss": float(loss)})
                    training_state["loss_history"] = history[-1000:]
                if training_state.get("should_stop", False):
                    break
        except Exception as exc:
            logger.exception("V2 training runner failed")
            training_state["error"] = str(exc)
        finally:
            training_state["is_training"] = False
            try:
                if handler.model is not None and getattr(handler.model, "decoder", None) is not None:
                    handler.model.decoder = unwrap_module(handler.model.decoder)
                    handler.model.decoder.set_to_inference_mode()
            except Exception:
                try:
                    handler.model.decoder.eval()
                except Exception:
                    logger.exception("Failed to restore decoder state after V2 training")
            cm = training_state.pop("_component_manager", None)
            if cm is not None:
                cm.restore()

    return _runner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_model_timestep_params(model: Any) -> Dict[str, float]:
    """Read timestep sampling params from model.config, with safe defaults."""
    config = getattr(model, "config", None)
    return {
        "timestep_mu": getattr(config, "timestep_mu", -0.4),
        "timestep_sigma": getattr(config, "timestep_sigma", 1.0),
        "data_proportion": getattr(config, "data_proportion", 0.5),
    }


def _check_handler(handler: Optional[AceStepHandler]) -> AceStepHandler:
    """Common handler validation for training routes."""
    if handler is None or handler.model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    if not hasattr(handler.model, "decoder") or handler.model.decoder is None:
        raise HTTPException(
            status_code=500,
            detail="Decoder not found. Please reload the model via /v1/reinitialize before training.",
        )
    return handler


def _prepare_for_training(handler: AceStepHandler, app: FastAPI) -> RuntimeComponentManager:
    """Unwrap decoder, move to GPU, offload everything else."""
    handler.model.decoder = unwrap_module(handler.model.decoder)
    mgr = RuntimeComponentManager(handler=handler, llm=app.state.llm_handler, app_state=app.state)
    mgr.move_decoder_to(str(handler.device))
    mgr.offload_vae_to_cpu()
    mgr.offload_text_encoder_to_cpu()
    mgr.offload_model_encoder_to_cpu()
    mgr.unload_llm()
    return mgr


# ---------------------------------------------------------------------------
# Route registration -- LoRA (replaces V1 /v1/training/start)
# ---------------------------------------------------------------------------

def register_v2_lora_training_start_route(
    app: FastAPI,
    verify_api_key: Callable[..., Any],
    wrap_response: Callable[[Any, int, Optional[str]], Dict[str, Any]],
    start_tensorboard: Callable[[FastAPI, str], Optional[str]],
) -> None:
    """Register the V2 ``/v1/training/start`` route (LoRA via FixedLoRATrainer)."""

    @app.post("/v1/training/start")
    async def start_training_v2(request: StartTrainingRequest, _: None = Depends(verify_api_key)):
        """Start V2 corrected LoRA training from preprocessed tensors."""

        initialize_training_state(app)
        training_state = app.state.training_state
        if training_state.get("is_training", False):
            raise HTTPException(status_code=400, detail="Training already in progress")

        handler: AceStepHandler = _check_handler(app.state.handler)
        mgr = _prepare_for_training(handler, app)

        try:
            from acestep.training_v2.configs import LoRAConfigV2, TrainingConfigV2
            from acestep.training_v2.trainer_fixed import FixedLoRATrainer

            ts_params = _read_model_timestep_params(handler.model)

            adapter_cfg = LoRAConfigV2(
                r=request.lora_rank,
                alpha=request.lora_alpha,
                dropout=request.lora_dropout,
            )
            training_cfg = TrainingConfigV2(
                shift=request.training_shift,
                learning_rate=request.learning_rate,
                batch_size=request.train_batch_size,
                gradient_accumulation_steps=request.gradient_accumulation,
                max_epochs=request.train_epochs,
                save_every_n_epochs=request.save_every_n_epochs,
                seed=request.training_seed,
                output_dir=request.lora_output_dir,
                gradient_checkpointing=request.gradient_checkpointing,
                # V2 corrected training params
                adapter_type="lora",
                dataset_dir=request.tensor_dir,
                device=str(handler.device),
                cfg_ratio=0.15,
                timestep_mu=ts_params["timestep_mu"],
                timestep_sigma=ts_params["timestep_sigma"],
                data_proportion=ts_params["data_proportion"],
                offload_encoder=False,  # already offloaded by RuntimeComponentManager
            )
            trainer = FixedLoRATrainer(
                model=handler.model,
                adapter_config=adapter_cfg,
                training_config=training_cfg,
            )
        except Exception as exc:
            training_state["is_training"] = False
            mgr.restore()
            return wrap_response(None, code=500, error=f"Failed to start V2 training: {exc}")

        tensorboard_logdir = os.path.join(request.lora_output_dir, "logs")
        os.makedirs(tensorboard_logdir, exist_ok=True)

        run_id = str(uuid4())
        training_state.update(
            {
                "is_training": True,
                "should_stop": False,
                "run_id": run_id,
                "trainer": trainer,
                "tensor_dir": request.tensor_dir,
                "tensorboard_logdir": tensorboard_logdir,
                "current_step": 0,
                "current_loss": None,
                "status": "Starting (V2 corrected)...",
                "loss_history": [],
                "training_log": "Starting (V2 corrected)...",
                "start_time": time.time(),
                "current_epoch": 0,
                "last_step_time": time.time(),
                "steps_per_second": 0.0,
                "estimated_time_remaining": 0.0,
                "error": None,
                "config": {
                    "adapter_type": "lora",
                    "lora_rank": request.lora_rank,
                    "lora_alpha": request.lora_alpha,
                    "learning_rate": request.learning_rate,
                    "epochs": request.train_epochs,
                    "trainer_version": "v2",
                },
                "_component_manager": mgr,
            }
        )
        training_state["tensorboard_url"] = start_tensorboard(app, tensorboard_logdir)

        threading.Thread(
            target=_make_runner(trainer, training_state, run_id, handler),
            daemon=True,
        ).start()

        return wrap_response(
            {
                "message": "V2 corrected LoRA training started",
                "tensor_dir": request.tensor_dir,
                "output_dir": request.lora_output_dir,
                "config": training_state["config"],
            }
        )


# ---------------------------------------------------------------------------
# Route registration -- LoKR (replaces V1 /v1/training/start_lokr)
# ---------------------------------------------------------------------------

def register_v2_lokr_training_start_route(
    app: FastAPI,
    verify_api_key: Callable[..., Any],
    wrap_response: Callable[[Any, int, Optional[str]], Dict[str, Any]],
    start_tensorboard: Callable[[FastAPI, str], Optional[str]],
) -> None:
    """Register the V2 ``/v1/training/start_lokr`` route (LoKR via FixedLoRATrainer)."""

    @app.post("/v1/training/start_lokr")
    async def start_lokr_training_v2(request: StartLoKRTrainingRequest, _: None = Depends(verify_api_key)):
        """Start V2 corrected LoKR training from preprocessed tensors."""

        initialize_training_state(app)
        training_state = app.state.training_state
        if training_state.get("is_training", False):
            raise HTTPException(status_code=400, detail="Training already in progress")

        handler: AceStepHandler = _check_handler(app.state.handler)
        mgr = _prepare_for_training(handler, app)

        try:
            from acestep.training_v2.configs import LoKRConfigV2, TrainingConfigV2
            from acestep.training_v2.trainer_fixed import FixedLoRATrainer

            ts_params = _read_model_timestep_params(handler.model)

            factor = request.lokr_factor
            if factor != -1:
                factor = int(factor)
                if factor == 0:
                    factor = 1
                factor = min(factor, 8)

            adapter_cfg = LoKRConfigV2(
                linear_dim=request.lokr_linear_dim,
                linear_alpha=request.lokr_linear_alpha,
                factor=factor,
                decompose_both=request.lokr_decompose_both,
                use_tucker=request.lokr_use_tucker,
                use_scalar=request.lokr_use_scalar,
                weight_decompose=request.lokr_weight_decompose,
            )
            training_cfg = TrainingConfigV2(
                shift=request.training_shift,
                learning_rate=request.learning_rate,
                batch_size=request.train_batch_size,
                gradient_accumulation_steps=request.gradient_accumulation,
                max_epochs=request.train_epochs,
                save_every_n_epochs=request.save_every_n_epochs,
                seed=request.training_seed,
                output_dir=request.output_dir,
                gradient_checkpointing=request.gradient_checkpointing,
                # V2 corrected training params
                adapter_type="lokr",
                dataset_dir=request.tensor_dir,
                device=str(handler.device),
                cfg_ratio=0.15,
                timestep_mu=ts_params["timestep_mu"],
                timestep_sigma=ts_params["timestep_sigma"],
                data_proportion=ts_params["data_proportion"],
                offload_encoder=False,
            )
            trainer = FixedLoRATrainer(
                model=handler.model,
                adapter_config=adapter_cfg,
                training_config=training_cfg,
            )
        except Exception as exc:
            training_state["is_training"] = False
            mgr.restore()
            return wrap_response(None, code=500, error=f"Failed to start V2 LoKR training: {exc}")

        tensorboard_logdir = os.path.join(request.output_dir, "logs")
        os.makedirs(tensorboard_logdir, exist_ok=True)

        run_id = str(uuid4())
        training_state.update(
            {
                "is_training": True,
                "should_stop": False,
                "run_id": run_id,
                "trainer": trainer,
                "tensor_dir": request.tensor_dir,
                "tensorboard_logdir": tensorboard_logdir,
                "current_step": 0,
                "current_loss": None,
                "status": "Starting (V2 corrected)...",
                "loss_history": [],
                "training_log": "Starting (V2 corrected)...",
                "start_time": time.time(),
                "current_epoch": 0,
                "last_step_time": time.time(),
                "steps_per_second": 0.0,
                "estimated_time_remaining": 0.0,
                "error": None,
                "config": {
                    "adapter_type": "lokr",
                    "lokr_linear_dim": request.lokr_linear_dim,
                    "lokr_linear_alpha": request.lokr_linear_alpha,
                    "lokr_factor": request.lokr_factor,
                    "learning_rate": request.learning_rate,
                    "epochs": request.train_epochs,
                    "trainer_version": "v2",
                },
                "_component_manager": mgr,
            }
        )
        training_state["tensorboard_url"] = start_tensorboard(app, tensorboard_logdir)

        threading.Thread(
            target=_make_runner(trainer, training_state, run_id, handler),
            daemon=True,
        ).start()

        return wrap_response(
            {
                "message": "V2 corrected LoKR training started",
                "tensor_dir": request.tensor_dir,
                "output_dir": request.output_dir,
                "config": training_state["config"],
            }
        )
