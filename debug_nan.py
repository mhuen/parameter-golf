"""NaN / overflow watchdog for multi-stream training.

Activated by the ``DEBUG_NAN`` environment variable:

    0  (default)  All methods are no-ops — zero overhead.
    1             Post-step monitoring (compile-safe).  Logs param/grad norms,
                  alpha/beta/q_gain trajectories, loss values.
    2             Full diagnostic mode.  Disables ``torch.compile``, enables
                  ``torch.autograd.set_detect_anomaly(True)``, attaches forward
                  hooks to every module for per-block activation inspection.
"""

from __future__ import annotations

import math
import os
import sys
from collections import deque
from typing import Callable

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tensor_stats(t: torch.Tensor) -> dict:
    """Quick NaN/Inf/norm/mean/std/min/max summary for a single tensor."""
    with torch.no_grad():
        has_nan = torch.isnan(t).any().item()
        has_inf = torch.isinf(t).any().item()
        f = t.float()
        norm = f.norm().item()
        n = t.numel()
        abs_max = f.abs().max().item() if n > 0 else 0.0
        mean = f.mean().item() if n > 0 else 0.0
        std = f.std().item() if n > 1 else 0.0
        vmin = f.min().item() if n > 0 else 0.0
        vmax = f.max().item() if n > 0 else 0.0
    return dict(
        has_nan=has_nan, has_inf=has_inf, norm=norm, abs_max=abs_max,
        mean=mean, std=std, min=vmin, max=vmax,
    )


# ---------------------------------------------------------------------------
# NaNWatchdog
# ---------------------------------------------------------------------------


class NaNWatchdog:
    """Lightweight NaN/overflow debugger for multi-stream training.

    Construct via :meth:`from_env` which reads the ``DEBUG_NAN`` env var.
    """

    def __init__(self, level: int = 0, log_fn: Callable[..., None] | None = None):
        self.level = level
        self._log = log_fn or (lambda *a, **kw: print(*a, **kw, file=sys.stderr))
        self._history: deque[dict] = deque(maxlen=50)
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._first_nan_step: int | None = None
        # Level-2 hook data: populated by forward hooks each step
        self._hook_data: list[dict] = []

    @classmethod
    def from_env(cls, log_fn: Callable[..., None] | None = None) -> "NaNWatchdog":
        level = int(os.environ.get("DEBUG_NAN", "0"))
        return cls(level=level, log_fn=log_fn)

    # ------------------------------------------------------------------
    # Compile gate
    # ------------------------------------------------------------------

    def should_disable_compile(self) -> bool:
        """Return ``True`` when compile must be skipped (level >= 2)."""
        return self.level >= 2

    @property
    def triggered(self) -> bool:
        """Return ``True`` after the first NaN/Inf has been detected."""
        return self._first_nan_step is not None

    # ------------------------------------------------------------------
    # Loss check (called right after forward)
    # ------------------------------------------------------------------

    def check_loss(self, loss: torch.Tensor, step: int) -> None:
        if self.level < 1:
            return
        val = loss.item()
        bad = math.isnan(val) or math.isinf(val)
        if bad and self._first_nan_step is None:
            self._first_nan_step = step
            self._log(f"\n{'='*72}")
            self._log(f"DEBUG_NAN: *** NaN/Inf loss detected at step {step} ***")
            self._log(f"{'='*72}")
            self._dump_diagnostics(step, trigger="loss")

    # ------------------------------------------------------------------
    # Parameter & gradient check (called after backward, before opt.step)
    # ------------------------------------------------------------------

    def check_params_and_grads(self, model: nn.Module, step: int) -> None:
        if self.level < 1:
            return

        record: dict = {"step": step, "params": {}, "grads": {}}
        first_nan_param = None
        first_nan_grad = None

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            ps = _tensor_stats(p.data)
            record["params"][name] = ps
            if (ps["has_nan"] or ps["has_inf"]) and first_nan_param is None:
                first_nan_param = name

            if p.grad is not None:
                gs = _tensor_stats(p.grad)
                record["grads"][name] = gs
                if (gs["has_nan"] or gs["has_inf"]) and first_nan_grad is None:
                    first_nan_grad = name

        self._history.append(record)

        if first_nan_param or first_nan_grad:
            if self._first_nan_step is None:
                self._first_nan_step = step
            self._log(f"\n{'='*72}")
            if first_nan_param:
                self._log(
                    f"DEBUG_NAN: *** NaN/Inf in PARAMETER '{first_nan_param}' "
                    f"at step {step} ***"
                )
            if first_nan_grad:
                self._log(
                    f"DEBUG_NAN: *** NaN/Inf in GRADIENT '{first_nan_grad}' "
                    f"at step {step} ***"
                )
            self._log(f"{'='*72}")
            self._dump_diagnostics(step, trigger="param/grad")

    # ------------------------------------------------------------------
    # Alpha / beta / q_gain tracking
    # ------------------------------------------------------------------

    def log_alpha_beta_gates(self, model: nn.Module, step: int) -> None:
        """Log sigmoid(alpha), sigmoid(beta), q_gain for the LOGIT stream."""
        if self.level < 1:
            return

        info: dict = {"q_gain": [], "alpha_logit": [], "beta_logit": []}
        for name, p in model.named_parameters():
            if "q_gain" in name:
                info["q_gain"].append((name, p.data.float().tolist()))
            # Track alpha/beta for logit stream only (key contains "logit")
            elif "alpha" in name and "logit" in name:
                info["alpha_logit"].append(
                    (name, torch.sigmoid(p.data.float()).tolist())
                )
            elif "beta" in name and "logit" in name:
                info["beta_logit"].append(
                    (name, torch.sigmoid(p.data.float()).tolist())
                )

        # Also track bigram prior magnitude
        for name, p in model.named_parameters():
            if "bigram_logits" in name:
                info["bigram_logits_abs_max"] = p.data.float().abs().max().item()
                break

        self._log(f"DEBUG_NAN step={step} gates:")
        for key in ("q_gain", "alpha_logit", "beta_logit"):
            for pname, vals in info[key]:
                if isinstance(vals, list) and len(vals) > 8:
                    summary = (
                        f"min={min(vals):.4f} max={max(vals):.4f} "
                        f"mean={sum(vals)/len(vals):.4f}"
                    )
                else:
                    summary = (
                        ", ".join(f"{v:.4f}" for v in vals)
                        if isinstance(vals, list)
                        else f"{vals:.4f}"
                    )
                self._log(f"  {pname}: {summary}")
        if "bigram_logits_abs_max" in info:
            self._log(
                f"  bigram_logits abs_max: {info['bigram_logits_abs_max']:.4f}"
            )

    # ------------------------------------------------------------------
    # Step summary (called in the logging block)
    # ------------------------------------------------------------------

    def log_step_summary(self, step: int, train_loss: float) -> None:
        """Periodic summary of top gradient/param norms."""
        if self.level < 1:
            return
        if not self._history:
            return

        rec = self._history[-1]
        if rec["step"] != step:
            return

        # Top-10 gradient norms
        grad_norms = [
            (name, s["norm"]) for name, s in rec["grads"].items()
        ]
        grad_norms.sort(key=lambda x: x[1], reverse=True)

        self._log(f"DEBUG_NAN step={step} loss={train_loss:.6f}")
        self._log("  Top-10 grad norms:")
        for name, gn in grad_norms[:10]:
            ps = rec["params"].get(name, {})
            self._log(
                f"    {name}: grad_norm={gn:.4f} "
                f"param_norm={ps.get('norm', 0):.4f} "
                f"param_max={ps.get('abs_max', 0):.4f}"
            )

        # Top-10 param norms
        param_norms = [
            (name, s["norm"]) for name, s in rec["params"].items()
        ]
        param_norms.sort(key=lambda x: x[1], reverse=True)
        self._log("  Top-10 param norms:")
        for name, pn in param_norms[:10]:
            self._log(f"    {name}: {pn:.4f}")

        # Level-2 hook data
        if self.level >= 2 and self._hook_data:
            self._log("  Forward hook data:")
            for entry in self._hook_data:
                self._log(
                    f"    {entry['name']}: "
                    f"mean={entry.get('mean', 0):.4f} "
                    f"std={entry.get('std', 0):.4f} "
                    f"min={entry.get('min', 0):.4f} "
                    f"max={entry.get('max', 0):.4f} "
                    f"norm={entry.get('norm', 0):.4f}"
                    + (" **NaN**" if entry.get("has_nan") else "")
                    + (" **Inf**" if entry.get("has_inf") else "")
                )
            self._hook_data.clear()

    # ------------------------------------------------------------------
    # Forward hooks (level 2 only)
    # ------------------------------------------------------------------

    def attach_hooks(self, model: nn.Module) -> None:
        """Register forward hooks on key modules for activation monitoring."""
        if self.level < 2:
            return

        from multi_stream_attention import MultiStreamBlock, MultiStreamMLP

        for name, mod in model.named_modules():
            # Hooks on MultiStreamBlock: capture stream dict output
            if isinstance(mod, MultiStreamBlock):
                self._hooks.append(
                    mod.register_forward_hook(self._make_block_hook(name))
                )
            # Hooks on MultiStreamMLP.fc_up: capture pre-squaring activations
            elif isinstance(mod, MultiStreamMLP):
                if hasattr(mod, "fc_up"):
                    self._hooks.append(
                        mod.fc_up.register_forward_hook(
                            self._make_tensor_hook(f"{name}.fc_up")
                        )
                    )

        self._log(f"DEBUG_NAN: attached {len(self._hooks)} forward hooks")

    def _make_block_hook(self, block_name: str):
        """Hook for MultiStreamBlock — inspects per-stream output tensors."""
        watchdog = self

        def hook(module, args, output):
            if not isinstance(output, dict):
                return
            for sid, tensor in output.items():
                stats = _tensor_stats(tensor)
                stats["name"] = f"{block_name}/{sid}"
                watchdog._hook_data.append(stats)
                if stats["has_nan"] or stats["has_inf"]:
                    watchdog._log(
                        f"DEBUG_NAN: *** NaN/Inf in {stats['name']} "
                        f"(max={stats['abs_max']:.4f}) ***"
                    )

        return hook

    def _make_tensor_hook(self, name: str):
        """Hook for a linear layer — inspects the output tensor."""
        watchdog = self

        def hook(module, args, output):
            stats = _tensor_stats(output)
            stats["name"] = name
            watchdog._hook_data.append(stats)
            if stats["has_nan"] or stats["has_inf"]:
                watchdog._log(
                    f"DEBUG_NAN: *** NaN/Inf in {name} "
                    f"(max={stats['abs_max']:.4f}) ***"
                )

        return hook

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Diagnostics dump
    # ------------------------------------------------------------------

    def _dump_diagnostics(self, step: int, trigger: str) -> None:
        """Print the ring buffer history leading up to the failure."""
        self._log(f"\n--- Diagnostic dump (trigger: {trigger}, step {step}) ---")
        self._log(f"Ring buffer has {len(self._history)} entries")

        for rec in self._history:
            s = rec["step"]
            # Find worst gradient
            worst_grad = ("", 0.0)
            for name, gs in rec["grads"].items():
                if gs["has_nan"] or gs["has_inf"] or gs["norm"] > worst_grad[1]:
                    worst_grad = (name, gs["norm"])
                    if gs["has_nan"] or gs["has_inf"]:
                        break

            # Find worst param
            worst_param = ("", 0.0)
            for name, ps in rec["params"].items():
                if ps["has_nan"] or ps["has_inf"] or ps["norm"] > worst_param[1]:
                    worst_param = (name, ps["norm"])
                    if ps["has_nan"] or ps["has_inf"]:
                        break

            self._log(
                f"  step={s}: "
                f"worst_grad=({worst_grad[0]}, norm={worst_grad[1]:.4f}) "
                f"worst_param=({worst_param[0]}, norm={worst_param[1]:.4f})"
            )

        # Detailed dump of the final record
        if self._history:
            rec = self._history[-1]
            self._log(f"\n--- Full param/grad dump at step {rec['step']} ---")
            for name in sorted(rec["params"].keys()):
                ps = rec["params"][name]
                gs = rec["grads"].get(name, {})
                flags = ""
                if ps.get("has_nan"):
                    flags += " PARAM_NAN"
                if ps.get("has_inf"):
                    flags += " PARAM_INF"
                if gs.get("has_nan"):
                    flags += " GRAD_NAN"
                if gs.get("has_inf"):
                    flags += " GRAD_INF"
                if flags:
                    self._log(
                        f"  {name}: param(norm={ps['norm']:.4f}, "
                        f"max={ps['abs_max']:.4f}) "
                        f"grad(norm={gs.get('norm', 0):.4f}, "
                        f"max={gs.get('abs_max', 0):.4f}){flags}"
                    )

        # Level-2 hook data
        if self.level >= 2 and self._hook_data:
            self._log(f"\n--- Forward hook data at failure ---")
            for entry in self._hook_data:
                flags = ""
                if entry.get("has_nan"):
                    flags += " **NaN**"
                if entry.get("has_inf"):
                    flags += " **Inf**"
                self._log(
                    f"  {entry['name']}: "
                    f"mean={entry.get('mean', 0):.4f} "
                    f"std={entry.get('std', 0):.4f} "
                    f"min={entry.get('min', 0):.4f} "
                    f"max={entry.get('max', 0):.4f} "
                    f"norm={entry.get('norm', 0):.4f}{flags}"
                )

        self._log(f"--- End diagnostic dump ---\n")
