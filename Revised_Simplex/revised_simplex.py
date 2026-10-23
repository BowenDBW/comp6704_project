"""Revised Simplex for the released transportation LP.

Call solve(A, b, c_hat, {"M": M, "N": N}). M and N come from manifest.csv.
Inputs are from the dataset generated. Basis indices start at zero. This file performs no file IO.
Invalid inputs raise ValueError; numerical failures return a status.
"""
from time import perf_counter

import numpy as np

# Given allowance for running, the benchmark requires that the final solution satisfy all of these conditions.
DEFAULT_OPTIONS = {
    "reduced_cost_tol": 1e-10,
    "direction_tol": 1e-10,
    "phase1_tol": 1e-8,
    "max_iter": 10000,
}
ACCEPTANCE = {
    "primal_residual_inf": 1e-7,
    "min_z": -1e-8,
    "dual_residual_inf": 1e-7,
    "min_dual_slack": -1e-7,
    "relative_duality_gap": 1e-7,
    "relative_objective_error": 1e-7,
}

# Check feasibility and the duality gap without using the reference objective.
def _optimality(A, b, c, z, y):
    """Check the solution without using a reference objective."""
    r = c - A.T @ y
    objective = float(c @ z)
    dual_objective = float(b @ y)
    gap_scale = max(1.0, abs(objective), abs(dual_objective))

    metrics = {}
    metrics["primal_residual_inf"] = float(np.max(np.abs(A @ z - b)))
    metrics["min_z"] = float(np.min(z))
    metrics["dual_residual_inf"] = float(np.max(np.abs(A.T @ y + r - c)))
    metrics["min_dual_slack"] = float(np.min(r))
    metrics["relative_duality_gap"] = abs(objective - dual_objective) / gap_scale

    finite = True
    for value in [z, y, r, objective, dual_objective]:
        if not np.all(np.isfinite(value)):
            finite = False
    for value in metrics.values():
        if not np.isfinite(value):
            finite = False

    primal_ok = metrics["primal_residual_inf"] <= ACCEPTANCE["primal_residual_inf"]
    variables_ok = metrics["min_z"] >= ACCEPTANCE["min_z"]
    dual_ok = metrics["dual_residual_inf"] <= ACCEPTANCE["dual_residual_inf"]
    reduced_costs_ok = metrics["min_dual_slack"] >= ACCEPTANCE["min_dual_slack"]
    gap_ok = metrics["relative_duality_gap"] <= ACCEPTANCE["relative_duality_gap"]
    passed = finite and primal_ok and variables_ok and dual_ok and reduced_costs_ok and gap_ok
    return objective, r, metrics, bool(passed)

# Pivot between feasible bases until no variable can improve the current objective.
def _simplex(A, b, c, basis, phase, settings, history):
    """Use the same pivot steps for Phase I and Phase II."""
    basis = basis.copy()
    pivots = 0
    z_B = None
    y = None
    objective = None
    try:
        while True:
            B = A[:, basis]
            z_B = np.linalg.solve(B, b)
            basic_costs = c[basis]
            y = np.linalg.solve(B.T, basic_costs)
            reduced = c - A.T @ y
            objective = float(basic_costs @ z_B)

            for value in [z_B, y, reduced, objective]:
                if not np.all(np.isfinite(value)):
                    raise FloatingPointError("Nonfinite basis state.")
            if np.min(z_B) < ACCEPTANCE["min_z"]:
                raise FloatingPointError("Basis lost primal feasibility.")
            entry = {"phase": phase, "iteration": pivots,
                     "event": "iterate", "objective": objective}

            candidates = []
            for column in range(A.shape[1]):
                if column not in basis:
                    if reduced[column] < -settings["reduced_cost_tol"]:
                        candidates.append(column)

            if len(candidates) == 0:
                history.append(entry)
                status = "optimal"
                message = "No negative nonbasic reduced cost within tolerance."
                break
            if pivots >= settings["max_iter"]:
                history.append(entry)
                status = "iteration_limit"
                message = "Pivot limit reached."
                break

            # Visit columns in order. Strict comparison keeps the smaller index on ties.
            entering = candidates[0]
            for column in candidates:
                if reduced[column] < reduced[entering]:
                    entering = column
            direction = np.linalg.solve(B, A[:, entering])
            if not np.all(np.isfinite(direction)):
                raise FloatingPointError("Nonfinite entering direction.")

            # z_B changes to z_B - step * direction. Only positive components limit the step.
            leaving_row = None
            step = None
            for row in range(len(basis)):
                if direction[row] > settings["direction_tol"]:
                    basic_value = z_B[row]
                    if basic_value < 0.0:
                        basic_value = 0.0  # Clamp roundoff only for the ratio, never for outputs.
                    ratio = float(basic_value / direction[row])
                    if not np.isfinite(ratio):
                        raise FloatingPointError("Nonfinite ratio.")
                    if leaving_row is None:
                        leaving_row = row
                        step = ratio
                    elif ratio < step:
                        leaving_row = row
                        step = ratio
                    elif ratio == step:
                        if basis[row] < basis[leaving_row]:
                            leaving_row = row
                            step = ratio

            if leaving_row is None:
                history.append(entry)
                status = "unbounded_claim"
                message = "Improving direction has no positive component."
                break

            entry["entering"] = int(entering)
            entry["leaving"] = int(basis[leaving_row])
            entry["step"] = step
            history.append(entry)
            basis[leaving_row] = entering
            pivots += 1
    except (np.linalg.LinAlgError, FloatingPointError) as error:
        status = "numerical_failure"
        message = str(error)
        z_B = None
        y = None
        objective = None

    return {"basis": basis, "z_B": z_B, "y": y, "objective": objective,
            "pivots": pivots, "status": status, "message": f"Phase {phase}: {message}"}


# Computing a satisfied starting point using temporary demand variables, then remove them.
def _phase_one(A, b, M, N, settings, history):
    """Find a feasible original basis using demand artificial variables."""
    L, V = A.shape
    K = M * N
    artificial = np.zeros((L, N))
    for customer in range(N):
        artificial[M + customer, customer] = 1.0
    augmented = np.column_stack((A, artificial))
    cost = np.zeros(V + N)
    for column in range(V, V + N):
        cost[column] = 1.0

    # Basis order matches the supply, demand and capacity rows.
    columns = []
    for column in range(K, K + M):
        columns.append(column)
    for column in range(V, V + N):
        columns.append(column)
    for column in range(K + M, V):
        columns.append(column)
    basis = np.array(columns, dtype=int)
    if not np.array_equal(augmented[:, basis], np.eye(L)):
        raise ValueError("A does not match the released slack-column ordering.")

    first = _simplex(augmented, b, cost, basis, 1, settings, history)
    pivots = first["pivots"]
    if first["status"] != "optimal":
        return None, pivots, first["status"], first["message"]
    if first["objective"] > settings["phase1_tol"]:
        return None, pivots, "infeasible_claim", "Phase I optimum exceeds phase1_tol."
    basis = first["basis"]

    try:
        for row in range(L):
            if basis[row] < V:
                continue
            # A zero artificial variable may still occupy a basic position.
            if pivots >= settings["max_iter"]:
                return None, pivots, "iteration_limit", "Phase I limit reached during cleanup."
            B = augmented[:, basis]
            z_B = np.linalg.solve(B, b)
            if abs(z_B[row]) > settings["phase1_tol"]:
                raise FloatingPointError("Remaining artificial basic variable is not zero.")
            unit = np.zeros(L)
            unit[row] = 1.0
            weights = np.linalg.solve(B.T, unit)
            transformed_row = weights @ A  # One row of B^{-1} A, without forming an inverse.

            entering = None
            for column in range(V):
                if column not in basis:
                    if abs(transformed_row[column]) > settings["direction_tol"]:
                        entering = column
                        break
            if entering is None:
                raise FloatingPointError("Cannot remove artificial variable; no constraint was deleted.")

            # Either sign is allowed for a nonzero pivot at a zero basic value.
            leaving = int(basis[row])
            basis[row] = entering
            pivots += 1
            z_B = np.linalg.solve(augmented[:, basis], b)
            objective = float(cost[basis] @ z_B)
            if not np.all(np.isfinite(z_B)) or not np.isfinite(objective):
                raise FloatingPointError("Nonfinite artificial-variable cleanup state.")
            if np.min(z_B) < ACCEPTANCE["min_z"]:
                raise FloatingPointError("Artificial-variable cleanup lost feasibility.")
            history.append({"phase": 1, "iteration": pivots, "event": "artificial_exit",
                            "objective": objective, "entering": entering, "leaving": leaving})
    except (np.linalg.LinAlgError, FloatingPointError) as error:
        return None, pivots, "numerical_failure", f"Phase I cleanup: {error}"
    return basis, pivots, "optimal", "Original feasible basis found."

# Validate the inputs and return the checked result.
def solve(A, b, c_hat, options=None):
    """Return the solution, status, pivot counts, timing and in-memory history."""
    start = perf_counter()
    settings = DEFAULT_OPTIONS.copy()
    if options is not None:
        for name in options:
            if name not in DEFAULT_OPTIONS and name not in ["M", "N"]:
                raise ValueError("Unknown option name.")
            settings[name] = options[name]

    for name in ["M", "N", "max_iter"]:
        value = settings.get(name)
        minimum = 1
        if name == "max_iter":
            minimum = 0
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer; supply M,N from manifest.csv.")
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum}.")
    for name in ["reduced_cost_tol", "direction_tol", "phase1_tol"]:
        if not np.isfinite(settings[name]) or settings[name] <= 0:
            raise ValueError(f"{name} must be positive and finite.")

    # Copy inputs so other methods can reuse the original arrays.
    A = np.array(A, dtype=np.float64, copy=True)
    b = np.array(b, dtype=np.float64, copy=True)
    c = np.array(c_hat, dtype=np.float64, copy=True)
    M = settings["M"]
    N = settings["N"]
    L = M + N + M * N
    V = 2 * M * N + M
    if A.shape != (L, V) or b.shape != (L,) or c.shape != (V,):
        raise ValueError(f"Expected A={(L, V)}, b={(L,)}, c_hat={(V,)}.")
    for value in [A, b, c]:
        if not np.all(np.isfinite(value)):
            raise ValueError("The benchmark requires finite arrays.")
    if np.min(b) < 0:
        raise ValueError("The benchmark requires b >= 0.")

    result = {
        "z": None, "y": None, "r": None, "basis": None, "objective": None,
        "status": "numerical_failure", "iterations": 0,
        "phase1_iterations": 0, "phase2_iterations": 0,
        "history": [], "metrics": {}, "message": "",
    }
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            basis, count, status, message = _phase_one(A, b, M, N, settings, result["history"])
            result["phase1_iterations"] = count
            result["status"] = status
            result["message"] = message
            if basis is not None:
                # Original columns and costs are restored for Phase II.
                second = _simplex(A, b, c, basis, 2, settings, result["history"])
                result["phase2_iterations"] = second["pivots"]
                result["status"] = second["status"]
                result["message"] = second["message"]
                if second["z_B"] is not None:
                    z = np.zeros(V)
                    for row in range(L):
                        column = second["basis"][row]
                        z[column] = second["z_B"][row]
                    objective, r, metrics, passed = _optimality(A, b, c, z, second["y"])
                    result["z"] = z
                    result["y"] = second["y"]
                    result["r"] = r
                    result["basis"] = second["basis"]
                    result["objective"] = objective
                    result["metrics"] = metrics
                    if result["status"] == "optimal" and not passed:
                        result["status"] = "numerical_failure"
                        result["message"] = "Final optimality checks failed."
    except (np.linalg.LinAlgError, FloatingPointError) as error:
        result["status"] = "numerical_failure"
        result["message"] = str(error)
    result["iterations"] = result["phase1_iterations"] + result["phase2_iterations"]
    result["runtime_seconds"] = perf_counter() - start
    return result
