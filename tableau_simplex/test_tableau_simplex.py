"""Correctness tests for the two-phase full-tableau simplex solver."""

from __future__ import annotations

import csv
import unittest
from pathlib import Path

import numpy as np

from tableau_simplex import tableau_simplex


HERE = Path(__file__).resolve().parent


def benchmark_root() -> Path:
    candidates = (
        HERE.parent / "shared",
        HERE.parent / "COMP6704_Benchmark_v1_validated",
    )
    for candidate in candidates:
        if (candidate / "VALIDATION_SUMMARY.csv").is_file():
            return candidate
    raise FileNotFoundError("shared benchmark directory was not found")


class TableauSimplexTests(unittest.TestCase):
    def assert_optimal(
        self,
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        expected: float,
        places: int = 8,
    ) -> dict:
        result = tableau_simplex(A, b, c, pivot_rule="bland")
        self.assertEqual(result["status"], "optimal", result["message"])
        self.assertAlmostEqual(result["objective"], expected, places=places)
        self.assertLessEqual(result["primal_residual"], 1e-8)
        self.assertGreaterEqual(result["min_x"], -1e-9)
        return result

    def test_small_feasible_lp_with_known_optimum(self) -> None:
        # min -x - 2y subject to x + y + s = 2.
        A = np.array([[1.0, 1.0, 1.0]])
        b = np.array([2.0])
        c = np.array([-1.0, -2.0, 0.0])
        result = self.assert_optimal(A, b, c, -4.0)
        self.assertAlmostEqual(result["x"][1], 2.0)

    def test_phase_one_is_required(self) -> None:
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])
        c = np.array([1.0, 2.0])
        result = self.assert_optimal(A, b, c, 1.0)
        self.assertGreater(result["phase1_iterations"], 0)

    def test_infeasible_lp(self) -> None:
        A = np.array([[1.0], [1.0]])
        b = np.array([1.0, 2.0])
        c = np.array([1.0])
        result = tableau_simplex(A, b, c)
        self.assertEqual(result["status"], "infeasible")

    def test_unbounded_lp(self) -> None:
        # x - s = 1 permits x to increase without bound.
        A = np.array([[1.0, -1.0]])
        b = np.array([1.0])
        c = np.array([-1.0, 0.0])
        result = tableau_simplex(A, b, c)
        self.assertEqual(result["status"], "unbounded")

    def test_degenerate_lp(self) -> None:
        A = np.array(
            [
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 1.0],
            ]
        )
        b = np.array([1.0, 1.0, 1.0])
        c = np.array([-1.0, -1.0, 0.0, 0.0, 0.0])
        self.assert_optimal(A, b, c, -1.0)

    def test_redundant_equality_is_removed(self) -> None:
        A = np.array([[1.0], [2.0]])
        b = np.array([1.0, 2.0])
        c = np.array([3.0])
        result = self.assert_optimal(A, b, c, 3.0)
        self.assertEqual(len(result["basis"]), 1)

    def test_negative_rhs_is_normalized(self) -> None:
        A = np.array([[-1.0]])
        b = np.array([-2.0])
        c = np.array([1.0])
        self.assert_optimal(A, b, c, 2.0)

    def test_iteration_limit(self) -> None:
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])
        c = np.array([-1.0, 0.0])
        result = tableau_simplex(A, b, c, max_iter=0)
        self.assertEqual(result["status"], "iteration_limit")

    def test_invalid_nonfinite_input(self) -> None:
        with self.assertRaises(ValueError):
            tableau_simplex(
                np.array([[np.nan]]), np.array([1.0]), np.array([1.0])
            )

    def test_small_benchmark_instance(self) -> None:
        root = benchmark_root()
        case_id = "M03_N05_a030_seed2026_base"
        instance = root / "benchmark_v1" / "instances" / f"{case_id}.npz"
        with np.load(instance, allow_pickle=False) as data:
            A, b, c = (data[key].copy() for key in ("A", "b", "c_hat"))

        with (root / "VALIDATION_SUMMARY.csv").open(encoding="utf-8") as handle:
            references = {
                row["case_id"]: float(row["highs_objective"])
                for row in csv.DictReader(handle)
            }
        result = self.assert_optimal(A, b, c, references[case_id], places=7)
        relative_error = abs(result["objective"] - references[case_id]) / max(
            1.0, abs(references[case_id])
        )
        self.assertLessEqual(relative_error, 1e-7)

    def test_matches_scipy_highs_when_available(self) -> None:
        try:
            from scipy.optimize import linprog
        except ImportError:
            self.skipTest("SciPy is not installed")

        A = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 0.0]])
        b = np.array([4.0, 5.0])
        c = np.array([-3.0, -2.0, 0.0])
        expected = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
        self.assertTrue(expected.success)
        result = self.assert_optimal(A, b, c, float(expected.fun))
        self.assertLessEqual(np.linalg.norm(A @ result["x"] - b, np.inf), 1e-8)


if __name__ == "__main__":
    unittest.main()
