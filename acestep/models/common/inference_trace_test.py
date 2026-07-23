"""Tests for deterministic diffusion tensor tracing."""

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from acestep.models.common.dcw_correction import DCWCorrector
from acestep.models.common.inference_trace import TRACE_PATH_ENV, trace_tensor


class InferenceTraceTests(unittest.TestCase):
    """Verify trace output is opt-in, deterministic, and stage-addressable."""

    def test_trace_is_noop_without_environment_path(self):
        """Disabled tracing should not create any output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop(TRACE_PATH_ENV, None)
                trace_tensor("noise.initial", np.ones((1, 2), dtype=np.float32))
            self.assertFalse(path.exists())

    def test_trace_records_repeatable_tensor_fingerprint(self):
        """Identical tensors should produce identical summaries across stages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            tensor = np.array([[1.0, -2.0, 3.0]], dtype=np.float32)
            with patch.dict(os.environ, {TRACE_PATH_ENV: str(path)}):
                trace_tensor("step.latent.after_sampler", tensor, step=0, backend="test")
                trace_tensor("step.latent.after_dcw", tensor, step=0, backend="test")

            records = [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["sha256_f32"], records[1]["sha256_f32"])
            self.assertEqual(records[0]["l2"], records[1]["l2"])
            self.assertEqual(records[0]["shape"], [1, 3])
            self.assertEqual(records[1]["stage"], "step.latent.after_dcw")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode) & 0o077, 0)

    def test_pytorch_dcw_trace_captures_before_and_after(self):
        """PyTorch DCW should emit stage pairs with different fingerprints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            corrector = DCWCorrector(enabled=True, mode="pix", scaler=0.25)
            x_next = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
            denoised = torch.zeros_like(x_next)
            with patch.dict(os.environ, {TRACE_PATH_ENV: str(path)}):
                result = corrector.apply(x_next, denoised, t_curr=0.5)

            records = [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([record["stage"] for record in records], [
                "step.latent.after_sampler",
                "step.latent.after_dcw",
            ])
            self.assertNotEqual(records[0]["sha256_f32"], records[1]["sha256_f32"])
            torch.testing.assert_close(result, x_next * 1.25)

    def test_trace_write_failure_warns_without_aborting_inference(self):
        """Trace filesystem failures should be diagnostic rather than fatal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            with patch.dict(os.environ, {TRACE_PATH_ENV: str(path)}), patch(
                "acestep.models.common.inference_trace.os.open",
                side_effect=OSError("read-only filesystem"),
            ), patch(
                "acestep.models.common.inference_trace.logger.warning"
            ) as warning_mock:
                trace_tensor("noise.initial", np.ones((1, 2), dtype=np.float32))

            warning_mock.assert_called_once()
            self.assertIn("Continuing inference", warning_mock.call_args.args[0])

    @unittest.skipUnless(hasattr(os, "O_NOFOLLOW"), "O_NOFOLLOW is unavailable")
    def test_trace_refuses_to_follow_symbolic_link(self):
        """Trace output should not append through a symbolic-link path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "private.txt"
            target.write_text("private", encoding="utf-8")
            trace_path = Path(tmpdir) / "trace.jsonl"
            trace_path.symlink_to(target)

            with patch.dict(os.environ, {TRACE_PATH_ENV: str(trace_path)}), patch(
                "acestep.models.common.inference_trace.logger.warning"
            ) as warning_mock:
                trace_tensor("noise.initial", np.ones((1, 2), dtype=np.float32))

            self.assertEqual(target.read_text(encoding="utf-8"), "private")
            warning_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
