"""Contract tests to catch v22 API drift before expensive GPU initialization."""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V22 = ROOT / "test_phase3_AIF_simplified_v22.py"
E01 = ROOT / "experiments" / "p0_v22_equivalence.py"
P0 = ROOT / "experiments" / "p0_protocol_audit.py"


def module_func(tree, name):
    return next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)


class E01ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v22 = ast.parse(V22.read_text(encoding="utf-8"))
        cls.e01 = ast.parse(E01.read_text(encoding="utf-8"))
        cls.p0 = ast.parse(P0.read_text(encoding="utf-8"))

    def test_canonical_callable_keyword_contract(self):
        sig = module_func(self.v22, "run_phase3_streaming_aif")
        accepted = {x.arg for x in sig.args.kwonlyargs}
        wrapper = module_func(self.e01, "run_canonical")
        calls = [
            n for n in ast.walk(wrapper)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "run_phase3_streaming_aif"
        ]
        self.assertEqual(len(calls), 1)
        provided = {kw.arg for kw in calls[0].keywords}
        self.assertFalse(provided - accepted, f"Unknown v22 args: {provided-accepted}")
        self.assertFalse([k for k in ("dataset", "root", "class_name", "device", "out_json", "csv") if k not in provided])
        keywords = {k.arg:k.value for k in calls[0].keywords}
        for name, expected in (("max_samples", 0), ("use_modes", False),
                               ("use_rotations", False), ("use_global_threshold_initialization", False),
                               ("no_shuffle_test", True)):
            self.assertIsInstance(keywords[name], ast.Constant)
            self.assertEqual(keywords[name].value, expected)

    def test_p0_config_field_contract(self):
        cfg = next(n for n in self.p0.body if isinstance(n, ast.ClassDef) and n.name == "Config")
        allowed = {n.target.id for n in cfg.body if isinstance(n, ast.AnnAssign)
                   and isinstance(n.target, ast.Name)}
        calls = [
            node for node in ast.walk(module_func(self.e01, "audit"))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Config"
        ]
        self.assertEqual(len(calls), 1)
        provided = {kw.arg for kw in calls[0].keywords}
        self.assertEqual(provided, allowed, f"P0 Config API drift: missing={allowed-provided}, extra={provided-allowed}")

    def test_source_module_is_preserved(self):
        src = E01.read_text(encoding="utf-8")
        self.assertIn("max_samples=0", src)
        self.assertNotIn("torch.save(", src)


if __name__ == "__main__":
    unittest.main()
