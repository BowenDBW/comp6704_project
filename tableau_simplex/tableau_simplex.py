"""Two-phase primal simplex using explicit full-tableau pivots.

The public solver handles LPs in equality standard form::

    minimize c.T @ x
    subject to A @ x = b, x >= 0

The last tableau row stores reduced costs. Its right-hand-side entry stores
the negative objective value, which makes ordinary Gauss-Jordan row updates
valid for both constraint and objective rows.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np


VALID_PIVOT_RULES = {"most_negative", "bland"}


def _validated_inputs(
    A: np.ndarray, b: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A_copy = np.asarray(A, dtype=np.float64).copy()
    b_copy = np.asarray(b, dtype=np.float64).reshape(-1).copy()
    c_copy = np.asarray(c, dtype=np.float64).reshape(-1).copy()

    if np.asarray(A).ndim != 2:
        raise ValueError("A must be two-dimensional")
    m, n = A_copy.shape
    if b_copy.shape != (m,):
        raise ValueError(f"b must have shape ({m},), got {b_copy.shape}")
    if c_copy.shape != (n,):
        raise ValueError(f"c must have shape ({n},), got {c_copy.shape}")
    if not all(np.isfinite(value).all() for value in (A_copy, b_copy, c_copy)):
        raise ValueError("A, b, and c must contain only finite values")
    return A_copy, b_copy, c_copy


def initialize_tableau(
    A: np.ndarray, b: np.ndarray, costs: np.ndarray, basis: list[int]
) -> np.ndarray:
    """Build a tableau and canonicalize its objective for ``basis``.

    ``A`` is expected to already be canonical with respect to ``basis``.
    """
    m, n = A.shape
    if len(basis) != m:
        raise ValueError("basis length must equal the number of constraints")
    tableau = np.zeros((m + 1, n + 1), dtype=np.float64)
    tableau[:m, :n] = A
    tableau[:m, -1] = b
    canonicalize_objective(tableau, basis, costs)
    return tableau


def canonicalize_objective(
    tableau: np.ndarray, basis: list[int], costs: np.ndarray
) -> None:
    """Set the reduced-cost row for the current canonical basis."""
    n = tableau.shape[1] - 1
    if costs.shape != (n,):
        raise ValueError("cost vector does not match tableau width")
    tableau[-1, :] = 0.0
    tableau[-1, :n] = costs
    for row, basic_col in enumerate(basis):
        basic_cost = costs[basic_col]
        if basic_cost != 0.0:
            tableau[-1, :] -= basic_cost * tableau[row, :]


def choose_entering_variable(
    tableau: np.ndarray,
    basis: list[int],
    tol: float,
    pivot_rule: str,
) -> int | None:
    """Choose a nonbasic column with reduced cost below ``-tol``."""
    if pivot_rule not in VALID_PIVOT_RULES:
        raise ValueError(f"unknown pivot rule: {pivot_rule}")
    n = tableau.shape[1] - 1
    basic = set(basis)
    eligible = [j for j in range(n) if j not in basic and tableau[-1, j] < -tol]
    if not eligible:
        return None
    if pivot_rule == "bland":
        return eligible[0]
    return min(eligible, key=lambda j: (tableau[-1, j], j))


def choose_leaving_variable(
    tableau: np.ndarray,
    basis: list[int],
    entering_col: int,
    tol: float,
    pivot_rule: str = "most_negative",
) -> int | None:
    """Apply the positive-coefficient minimum-ratio test."""
    column = tableau[:-1, entering_col]
    rhs = tableau[:-1, -1]
    eligible = np.flatnonzero(column > tol)
    if eligible.size == 0:
        return None

    ratios = np.maximum(rhs[eligible], 0.0) / column[eligible]
    minimum = float(np.min(ratios))
    ratio_tol = tol * max(1.0, abs(minimum))
    tied = [
        int(row)
        for row, ratio in zip(eligible, ratios)
        if abs(float(ratio) - minimum) <= ratio_tol
    ]
    if pivot_rule == "bland":
        return min(tied, key=lambda row: (basis[row], row))
    return tied[0]


def pivot(
    tableau: np.ndarray,
    basis: list[int],
    pivot_row: int,
    pivot_col: int,
    tol: float = 1e-12,
) -> None:
    """Perform an in-place full-tableau Gauss-Jordan pivot."""
    element = tableau[pivot_row, pivot_col]
    if not np.isfinite(element) or abs(element) <= tol:
        raise FloatingPointError("pivot element is zero or numerically unsafe")

    tableau[pivot_row, :] /= element
    factors = tableau[:, pivot_col].copy()
    factors[pivot_row] = 0.0
    tableau[:, :] -= factors[:, None] * tableau[pivot_row, :]
    tableau[:, pivot_col] = 0.0
    tableau[pivot_row, pivot_col] = 1.0
    basis[pivot_row] = pivot_col


def simplex_iterations(
    tableau: np.ndarray,
    basis: list[int],
    tol: float,
    max_iter: int,
    pivot_rule: str,
) -> dict[str, Any]:
    """Run primal simplex pivots on an already feasible tableau."""
    history = [-float(tableau[-1, -1])]
    for iteration in range(max_iter + 1):
        if not np.isfinite(tableau).all():
            return {
                "status": "numerical_failure",
                "iterations": iteration,
                "objective_history": history,
                "message": "tableau contains a non-finite value",
            }
        if float(np.min(tableau[:-1, -1])) < -10.0 * tol:
            return {
                "status": "numerical_failure",
                "iterations": iteration,
                "objective_history": history,
                "message": "primal feasibility was lost during pivoting",
            }

        entering = choose_entering_variable(tableau, basis, tol, pivot_rule)
        if entering is None:
            return {
                "status": "optimal",
                "iterations": iteration,
                "objective_history": history,
                "message": "all nonbasic reduced costs satisfy optimality tolerance",
            }
        if iteration == max_iter:
            return {
                "status": "iteration_limit",
                "iterations": iteration,
                "objective_history": history,
                "message": "maximum pivot count reached",
            }

        leaving = choose_leaving_variable(
            tableau, basis, entering, tol, pivot_rule=pivot_rule
        )
        if leaving is None:
            return {
                "status": "unbounded",
                "iterations": iteration,
                "objective_history": history,
                "message": "entering column has no eligible leaving row",
            }
        try:
            pivot(tableau, basis, leaving, entering, tol=max(tol * 1e-3, 1e-15))
        except FloatingPointError as exc:
            return {
                "status": "numerical_failure",
                "iterations": iteration,
                "objective_history": history,
                "message": str(exc),
            }
        history.append(-float(tableau[-1, -1]))

    raise AssertionError("unreachable simplex loop exit")


def _remove_artificial_variables(
    tableau: np.ndarray,
    basis: list[int],
    original_variable_count: int,
    active_rows: list[int],
    tol: float,
) -> tuple[np.ndarray, list[int], list[int], str | None]:
    """Pivot zero artificial basics out, deleting only redundant rows."""
    row = 0
    while row < len(basis):
        if basis[row] < original_variable_count:
            row += 1
            continue
        if abs(float(tableau[row, -1])) > 10.0 * tol:
            return tableau, basis, active_rows, "positive artificial basic variable"

        basic = set(basis)
        candidates = [
            col
            for col in range(original_variable_count)
            if col not in basic and abs(float(tableau[row, col])) > tol
        ]
        if candidates:
            pivot(
                tableau,
                basis,
                row,
                candidates[0],
                tol=max(tol * 1e-3, 1e-15),
            )
            row += 1
            continue

        # In canonical form this row has no coefficient in any original
        # column, so 0 = 0 is redundant once its zero artificial is removed.
        tableau = np.delete(tableau, row, axis=0)
        basis.pop(row)
        active_rows.pop(row)

    tableau = np.hstack(
        (tableau[:, :original_variable_count], tableau[:, -1:])
    )
    return tableau, basis, active_rows, None


def phase_one(
    A: np.ndarray,
    b: np.ndarray,
    tol: float,
    max_iter: int,
    pivot_rule: str,
) -> dict[str, Any]:
    """Find a BFS using one artificial variable per equality row."""
    m, n = A.shape
    augmented = np.hstack((A, np.eye(m, dtype=np.float64)))
    costs = np.concatenate((np.zeros(n), np.ones(m)))
    basis = list(range(n, n + m))
    tableau = initialize_tableau(augmented, b, costs, basis)
    result = simplex_iterations(tableau, basis, tol, max_iter, pivot_rule)
    phase_one_objective = -float(tableau[-1, -1])
    result.update(
        {
            "tableau": tableau,
            "basis": basis,
            "active_rows": list(range(m)),
            "phase_one_objective": phase_one_objective,
        }
    )
    if result["status"] != "optimal":
        return result
    if phase_one_objective > 10.0 * tol:
        result["status"] = "infeasible"
        result["message"] = "Phase I optimum is positive"
        return result

    cleaned, basis, active_rows, error = _remove_artificial_variables(
        tableau, basis, n, list(range(m)), tol
    )
    result.update(
        {"tableau": cleaned, "basis": basis, "active_rows": active_rows}
    )
    if error is not None:
        result["status"] = "numerical_failure"
        result["message"] = error
    return result


def phase_two(
    phase_one_result: dict[str, Any],
    original_cost: np.ndarray,
    tol: float,
    max_iter: int,
    pivot_rule: str,
) -> dict[str, Any]:
    """Restore the original objective and optimize from the Phase I BFS."""
    tableau = phase_one_result["tableau"]
    basis = phase_one_result["basis"]
    canonicalize_objective(tableau, basis, original_cost)
    return simplex_iterations(tableau, basis, tol, max_iter, pivot_rule)


def _empty_solution(n: int) -> np.ndarray:
    return np.full(n, np.nan, dtype=np.float64)


def tableau_simplex(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    tol: float = 1e-9,
    max_iter: int = 10000,
    pivot_rule: str = "most_negative",
) -> dict[str, Any]:
    """Solve an equality-form minimization LP with two-phase tableau simplex."""
    started = perf_counter()
    A_original, b_original, c_original = _validated_inputs(A, b, c)
    m, n = A_original.shape
    if tol <= 0.0 or not np.isfinite(tol):
        raise ValueError("tol must be a positive finite number")
    if max_iter < 0:
        raise ValueError("max_iter must be nonnegative")
    if pivot_rule not in VALID_PIVOT_RULES:
        raise ValueError(f"pivot_rule must be one of {sorted(VALID_PIVOT_RULES)}")

    row_signs = np.ones(m, dtype=np.float64)
    negative = b_original < 0.0
    row_signs[negative] = -1.0
    A_normalized = A_original * row_signs[:, None]
    b_normalized = b_original * row_signs

    first = phase_one(
        A_normalized, b_normalized, tol, max_iter, pivot_rule
    )
    phase1_iterations = int(first["iterations"])
    status = first["status"]
    message = first["message"]
    phase2_iterations = 0
    phase2_history: list[float] = []

    if status == "optimal":
        remaining = max(0, max_iter - phase1_iterations)
        second = phase_two(first, c_original, tol, remaining, pivot_rule)
        status = second["status"]
        message = second["message"]
        phase2_iterations = int(second["iterations"])
        phase2_history = list(second["objective_history"])

    x = _empty_solution(n)
    objective = float("nan")
    primal_residual = float("nan")
    min_x = float("nan")
    y = np.full(m, np.nan, dtype=np.float64)
    reduced_costs = np.full(n, np.nan, dtype=np.float64)

    if status == "optimal":
        x = np.zeros(n, dtype=np.float64)
        tableau = first["tableau"]
        basis = first["basis"]
        for row, basic_col in enumerate(basis):
            x[basic_col] = tableau[row, -1]
        objective = float(c_original @ x)
        primal_residual = float(np.linalg.norm(A_original @ x - b_original, ord=np.inf))
        min_x = float(np.min(x))

        active_rows = first["active_rows"]
        try:
            active_matrix = A_normalized[np.asarray(active_rows), :]
            y_active = np.linalg.solve(
                active_matrix[:, basis].T, c_original[np.asarray(basis)]
            )
            y_normalized = np.zeros(m, dtype=np.float64)
            y_normalized[np.asarray(active_rows)] = y_active
            y = row_signs * y_normalized
            reduced_costs = c_original - A_original.T @ y
            if not all(
                np.isfinite(value).all()
                for value in (x, y, reduced_costs)
            ):
                raise np.linalg.LinAlgError("non-finite certificate")
        except np.linalg.LinAlgError as exc:
            status = "numerical_failure"
            message = f"could not recover a dual certificate: {exc}"

    runtime = perf_counter() - started
    return {
        "z": x,
        "x": x,
        "y": y,
        "r": reduced_costs,
        "reduced_costs": reduced_costs,
        "basis": list(first.get("basis", [])),
        "objective": objective,
        "status": status,
        "iterations": phase1_iterations + phase2_iterations,
        "phase1_iterations": phase1_iterations,
        "phase2_iterations": phase2_iterations,
        "runtime": runtime,
        "phase1_objective": float(first["phase_one_objective"]),
        "phase1_objective_history": list(first["objective_history"]),
        "phase2_objective_history": phase2_history,
        "primal_residual": primal_residual,
        "min_x": min_x,
        "message": message,
        "tol": tol,
        "pivot_rule": pivot_rule,
    }


def solve(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Repository-style entry point compatible with the shared solver contract."""
    settings = dict(options or {})
    tol = float(settings.pop("tol", 1e-9))
    max_iter = int(settings.pop("max_iter", 10000))
    pivot_rule = str(settings.pop("pivot_rule", "most_negative"))
    if settings:
        raise ValueError(f"unknown solver options: {sorted(settings)}")
    return tableau_simplex(
        A,
        b,
        c,
        tol=tol,
        max_iter=max_iter,
        pivot_rule=pivot_rule,
    )
