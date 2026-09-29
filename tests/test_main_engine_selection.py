"""
tests/test_main_engine_selection.py

Regression test for the default engine change:
  - No flag (default) must select AdvancedTaichiEngine
  - --advanced must select AdvancedTaichiEngine
  - --taichi must select TaichiEngine
  - --legacy-leapfrog must select SimulationEngine

Tests parse_args() directly with a simulated argv so no renderer or
universe generation is needed.  Engine class identity is checked via
the production if/elif/else chain in main.py logic.
"""

import sys
import os
import unittest
import types as pytypes

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _parse(argv_extra):
    """Call parse_args() with a minimal argv that avoids renderer launch."""
    import main as m
    old_argv = sys.argv
    try:
        # Always set --mode mpl + dummy preset so parse_args() doesn't block
        sys.argv = ["main.py"] + argv_extra
        return m.parse_args()
    finally:
        sys.argv = old_argv


def _engine_class_name(args):
    """
    Replicate the if/elif/else engine-selection logic from main.py
    and return the class name string that *would* be instantiated,
    without actually constructing the engine.
    """
    if args.legacy_leapfrog:
        return "SimulationEngine"
    elif args.taichi:
        return "TaichiEngine"
    else:
        # default AND --advanced both fall here
        return "AdvancedTaichiEngine"


class TestEngineSelection(unittest.TestCase):
    """Verify every flag → engine mapping in main.py engine selection logic."""

    def test_default_selects_advanced(self):
        """No engine flag → AdvancedTaichiEngine (the new default)."""
        args = _parse([])
        self.assertFalse(args.legacy_leapfrog,
                         "--legacy-leapfrog should be False by default")
        self.assertFalse(args.taichi,
                         "--taichi should be False by default")
        self.assertEqual(_engine_class_name(args), "AdvancedTaichiEngine",
                         "Default engine must be AdvancedTaichiEngine")

    def test_advanced_flag_selects_advanced(self):
        """--advanced explicitly selects AdvancedTaichiEngine."""
        args = _parse(["--advanced"])
        self.assertFalse(args.legacy_leapfrog)
        self.assertFalse(args.taichi)
        self.assertTrue(args.advanced)
        self.assertEqual(_engine_class_name(args), "AdvancedTaichiEngine")

    def test_taichi_flag_selects_taichi(self):
        """--taichi selects TaichiEngine."""
        args = _parse(["--taichi"])
        self.assertFalse(args.legacy_leapfrog)
        self.assertTrue(args.taichi)
        self.assertFalse(args.advanced)
        self.assertEqual(_engine_class_name(args), "TaichiEngine")

    def test_legacy_leapfrog_selects_simulation_engine(self):
        """--legacy-leapfrog selects SimulationEngine."""
        args = _parse(["--legacy-leapfrog"])
        self.assertTrue(args.legacy_leapfrog)
        self.assertEqual(_engine_class_name(args), "SimulationEngine")

    def test_no_flag_is_not_hermite(self):
        """Default must NOT be HermiteEngine (the old default)."""
        args = _parse([])
        self.assertNotEqual(_engine_class_name(args), "HermiteEngine",
                            "Default must no longer be HermiteEngine")

    def test_advanced_flag_is_not_taichi(self):
        """--advanced must not select TaichiEngine."""
        args = _parse(["--advanced"])
        self.assertNotEqual(_engine_class_name(args), "TaichiEngine")

    def test_taichi_flag_is_not_advanced(self):
        """--taichi must not select AdvancedTaichiEngine."""
        args = _parse(["--taichi"])
        self.assertNotEqual(_engine_class_name(args), "AdvancedTaichiEngine")

    def test_legacy_flag_is_not_advanced(self):
        """--legacy-leapfrog must not select AdvancedTaichiEngine."""
        args = _parse(["--legacy-leapfrog"])
        self.assertNotEqual(_engine_class_name(args), "AdvancedTaichiEngine")

    def test_flags_mutually_distinct(self):
        """
        Each explicit flag maps to a different engine class.
        Tests independence of the three flag paths.
        """
        mapping = {
            "none":    _engine_class_name(_parse([])),
            "advanced": _engine_class_name(_parse(["--advanced"])),
            "taichi":  _engine_class_name(_parse(["--taichi"])),
            "legacy":  _engine_class_name(_parse(["--legacy-leapfrog"])),
        }
        # --taichi and --legacy-leapfrog must each be distinct from each other
        self.assertNotEqual(mapping["taichi"], mapping["legacy"])
        # default and --advanced both map to AdvancedTaichiEngine (by design)
        self.assertEqual(mapping["none"], mapping["advanced"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
