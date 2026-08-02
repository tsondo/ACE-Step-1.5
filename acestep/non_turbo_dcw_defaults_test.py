"""Regression tests for direct sampler DCW defaults by model family."""

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

NON_TURBO_MODELS = (
    "acestep/models/base/modeling_acestep_v15_base.py",
    "acestep/models/sft/modeling_acestep_v15_base.py",
    "acestep/models/xl_base/modeling_acestep_v15_xl_base.py",
    "acestep/models/xl_sft/modeling_acestep_v15_xl_base.py",
)
TURBO_MODELS = (
    "acestep/models/turbo/modeling_acestep_v15_turbo.py",
    "acestep/models/xl_turbo/modeling_acestep_v15_xl_turbo.py",
)


def _generate_audio_dcw_default(relative_path: str) -> bool:
    """Return the literal ``dcw_enabled`` default from a model source file."""
    tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name != "generate_audio":
                continue
            positional = node.args.posonlyargs + node.args.args
            defaults = [None] * (len(positional) - len(node.args.defaults))
            defaults.extend(node.args.defaults)
            for argument, default in zip(positional, defaults, strict=True):
                if argument.arg == "dcw_enabled":
                    return ast.literal_eval(default)
    raise AssertionError(f"generate_audio(dcw_enabled=...) not found in {relative_path}")


class DirectSamplerDcwDefaultTests(unittest.TestCase):
    """Ensure lower-level defaults match the model-aware handler policy."""

    def test_non_turbo_direct_samplers_default_dcw_off(self):
        """Base and SFT samplers should not accumulate DCW over long schedules."""
        for relative_path in NON_TURBO_MODELS:
            with self.subTest(relative_path=relative_path):
                self.assertFalse(_generate_audio_dcw_default(relative_path))

    def test_turbo_direct_samplers_keep_dcw_on(self):
        """Turbo direct-call compatibility should remain unchanged."""
        for relative_path in TURBO_MODELS:
            with self.subTest(relative_path=relative_path):
                self.assertTrue(_generate_audio_dcw_default(relative_path))


if __name__ == "__main__":
    unittest.main()
