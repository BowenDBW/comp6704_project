"""Primal-Dual Interior-Point Method (Mehrotra predictor-corrector) for the
COMP6704 shared standard-form LP:

    min  c_hat^T z   subject to   A z = b,   z >= 0

Design summary (recorded here per PROPOSED_COMPARISON_PROTOCOL.md section 2,
which requires every implementation to document its internal settings):

- Direction: Mehrotra (1992) predictor-corrector on the perturbed KKT
  conditions, reduced to the normal equations (A W A^T) dy = rhs with
  W = diag(z / r). The matrix is factorised once per iteration and reused
  for the corrector.
- Initialization (Mehrotra-style least-squares heuristic):
    y0 = argmin ||A^T y - c||_2   (min-norm least squares)
    r0 = c - A^T y0, shifted strictly positive
    z0 = argmin ||A z - b||_2     (min-norm least squares)
    z0 shifted strictly positive
  Both shifts add max(0, -min(v)) + 1 so every entry is at least 1.
  This initialization is part of the timed solve (protocol section 4).
- Regularization: normal equations solved by Cholesky; if Cholesky fails or
  the solution is non-finite, a diagonal regularization from IPMOptions.
  reg_levels is added and the solve retried (first level that works wins).
- Iterative refinement: one refinement pass against the UNregularized
  normal-equation matrix, for accuracy.
- Fraction-to-boundary: adaptive tau = min(tau_max, max(tau_base, 1 - mu)),
  primal and dual step sizes computed separately.
- Stopping (internal, stricter than the protocol's acceptance thresholds):
    ||Az - b||_inf    <= primal_abs_tol (absolute)
    ||A^T y + r - c||_inf <= dual_abs_tol (absolute)
    |c^Tz - b^Ty| / max(1, |c^Tz|, |b^Ty|) <= gap_rel_tol
  mu is recorded as a diagnostic but is NEVER used alone as a stop condition
  (protocol section 2 explicitly forbids that).
- Failure handling: status is 'optimal', 'iteration_limit', or
  'numerical_failure'. On numerical trouble the best iterate seen so far
  (smallest worst absolute residual) is returned, never garbage.

No external LP solver is used anywhere in this module. HiGHS reference
values are only applied outside, after solving, for validation.

Everything is float64 (protocol section 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["IPMOptions", "IPMResult", "solve_ipm"]


@dataclass
class IPMOptions:
    """Internal solver settings. Defaults are 'ipm_default_v1'."""

    primal_abs_tol: float = 1e-9   # ||Az-b||_inf stop threshold (absolute)
    dual_abs_tol: float = 1e-9     # ||A^T y + r - c||_inf stop threshold (absolute)
    gap_rel_tol: float = 1e-9      # relative duality gap stop threshold
    max_iterations: int = 100
    tau_base: float = 0.995        # fraction-to-boundary lower bound
    tau_max: float = 0.9995        # fraction-to-boundary upper bound near convergence
    reg_levels: tuple = (0.0, 1e-12, 1e-10, 1e-8, 1e-6)
    refine_steps: int = 1          # iterative refinement passes on normal equations
    verbose: bool = False


@dataclass
class IPMResult:
    status: str                    # optimal / iteration_limit / numerical_failure
    z: np.ndarray                  # primal solution (length V), strictly positive path
    y: np.ndarray                  # dual multipliers (length L)
    r: np.ndarray                  # dual slacks c - A^T y (length V)
    iterations: int
    message: str
    history: list = field(default_factory=list)  # one dict per iteration
    init_residual_inf: float = 0.0  # ||[rp; rd]||_inf at the starting point


def _shift_positive(v: np.ndarray, margin: float = 1.0) -> np.ndarray:
    """Shift so that every entry is strictly >= margin (keeps centrality)."""
    s = max(0.0, -float(v.min())) + margin
    return v + s


def _max_step(v: np.ndarray, dv: np.ndarray, tau: float) -> float:
    """Largest alpha in [0, 1] with v + alpha*dv >= 0, scaled by tau."""
    neg = dv < 0.0
    if not np.any(neg):
        return 1.0
    return float(min(1.0, tau * np.min(-v[neg] / dv[neg])))


def solve_ipm(A: np.ndarray, b: np.ndarray, c: np.ndarray,
              options: IPMOptions | None = None) -> IPMResult:
    """Solve min c^T z, A z = b, z >= 0 with a primal-dual IPM.

    A, b, c are treated as read-only inputs (protocol section 3).
    """
    opts = options or IPMOptions()
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()
    c = np.asarray(c, dtype=np.float64).ravel()
    L, V = A.shape

    # ---------------- initialization (timed; part of the solve) -------------
    y0 = np.linalg.lstsq(A.T, c, rcond=None)[0]
    r = _shift_positive(c - A.T @ y0)
    z = _shift_positive(np.linalg.lstsq(A, b, rcond=None)[0])
    y = y0.copy()
    init_residual = max(float(np.abs(A @ z - b).max()), float(np.abs(A.T @ y + r - c).max()))

    history: list[dict] = []
    best = None  # (worst_residual, z, y, r, iter)

    def violation(pr: float, dr: float, gap: float) -> float:
        return max(pr, dr, 1e3 * gap)

    status = "iteration_limit"
    message = ""

    for k in range(1, opts.max_iterations + 1):
        rp = A @ z - b
        rd = A.T @ y + r - c
        zr = z * r
        mu = float(zr.sum()) / V
        pr = float(np.abs(rp).max())
        dr = float(np.abs(rd).max())
        obj = float(c @ z)
        dobj = float(b @ y)
        gap = abs(obj - dobj) / max(1.0, abs(obj), abs(dobj))

        rec = dict(iteration=k, mu=mu, primal_residual_inf=pr,
                   dual_residual_inf=dr, relative_duality_gap=gap,
                   objective=obj)
        history.append(rec)

        cur = (violation(pr, dr, gap), z.copy(), y.copy(), r.copy(), k)
        if best is None or cur[0] < best[0]:
            best = cur

        if pr <= opts.primal_abs_tol and dr <= opts.dual_abs_tol and gap <= opts.gap_rel_tol:
            status = "optimal"
            message = (f"converged in {k} iterations: primal_res={pr:.3e}, "
                       f"dual_res={dr:.3e}, rel_gap={gap:.3e}")
            rec["converged"] = True
            if opts.verbose:
                print(f"[ipm] iter {k:3d}  mu={mu:.3e}  pr={pr:.3e}  dr={dr:.3e}  gap={gap:.3e}  OPTIMAL")
            break
        if opts.verbose:
            print(f"[ipm] iter {k:3d}  mu={mu:.3e}  pr={pr:.3e}  dr={dr:.3e}  gap={gap:.3e}")

        # ---------------- normal-equation matrix (factor once) --------------
        w = z / r                      # W = diag(z/r)
        M = (A * w) @ A.T              # A W A^T
        M = 0.5 * (M + M.T)            # enforce symmetry against round-off

        def normal_rhs(gamma: np.ndarray) -> np.ndarray:
            # RHS of (A W A^T) dy = ... for complementarity RHS gamma
            return -rp - A @ (gamma / r + w * rd)

        def solve_normal(rhs: np.ndarray):
            for reg in opts.reg_levels:
                Mreg = M + reg * np.eye(L)
                try:
                    dy = np.linalg.solve(Mreg, rhs)
                except np.linalg.LinAlgError:
                    continue
                if not np.all(np.isfinite(dy)):
                    continue
                for _ in range(opts.refine_steps):
                    dy = dy + np.linalg.solve(Mreg, rhs - M @ dy)
                if np.all(np.isfinite(dy)):
                    return dy
            return None

        # ---------------- predictor (affine scaling) ------------------------
        gamma_aff = -zr
        dy_a = solve_normal(normal_rhs(gamma_aff))
        if dy_a is None:
            status, message = "numerical_failure", f"normal equations unsolvable at iteration {k} (predictor)"
            break
        dz_a = gamma_aff / r + w * rd + w * (A.T @ dy_a)
        dr_a = -rd - A.T @ dy_a

        a_aff_p = _max_step(z, dz_a, 1.0)
        a_aff_d = _max_step(r, dr_a, 1.0)
        a_aff = min(a_aff_p, a_aff_d)
        mu_aff = float(((z + a_aff * dz_a) * (r + a_aff * dr_a)).sum()) / V
        sigma = min(1.0, max(0.0, (mu_aff / mu) ** 3)) if mu > 0 else 0.0

        # ---------------- corrector -----------------------------------------
        gamma = -zr + sigma * mu - dz_a * dr_a
        dy = solve_normal(normal_rhs(gamma))
        if dy is None:
            status, message = "numerical_failure", f"normal equations unsolvable at iteration {k} (corrector)"
            break
        dz = gamma / r + w * rd + w * (A.T @ dy)
        dr_new = -rd - A.T @ dy

        tau = min(opts.tau_max, max(opts.tau_base, 1.0 - mu))
        a_p = _max_step(z, dz, tau)
        a_d = _max_step(r, dr_new, tau)
        rec["sigma"] = sigma
        rec["alpha_primal"] = a_p
        rec["alpha_dual"] = a_d

        if a_p <= 1e-14 and a_d <= 1e-14:
            status, message = "numerical_failure", f"step size collapsed to {max(a_p, a_d):.3e} at iteration {k}"
            break
        if not (np.all(np.isfinite(dz)) and np.all(np.isfinite(dy)) and np.all(np.isfinite(dr_new))):
            status, message = "numerical_failure", f"non-finite direction at iteration {k}"
            break

        z = z + a_p * dz
        y = y + a_d * dy
        r = r + a_d * dr_new

    if status != "optimal":
        # fall back to the best iterate seen (never return garbage)
        _, z_b, y_b, r_b, k_b = best if best is not None else (0.0, z, y, r, 0)
        z, y, r = z_b, y_b, r_b
        if status == "iteration_limit":
            message = f"iteration limit {opts.max_iterations} reached; best iterate from iteration {k_b}"

    return IPMResult(status=status, z=z, y=y, r=r, iterations=len(history),
                     message=message, history=history, init_residual_inf=init_residual)


def protocol_metrics(A: np.ndarray, b: np.ndarray, c: np.ndarray,
                     z: np.ndarray, y: np.ndarray, r: np.ndarray) -> dict:
    """Compute the common acceptance metrics of PROPOSED_COMPARISON_PROTOCOL.md
    section 2, in the original variable ordering, on the original A, b, c_hat.

    min_z / min_dual_slack keep their signed values (small negatives are NOT
    clipped to zero before evaluation, per the protocol).
    """
    obj = float(c @ z)
    dobj = float(b @ y)
    return {
        "objective": obj,
        "primal_residual_inf": float(np.abs(A @ z - b).max()),
        "min_z": float(z.min()),
        "dual_residual_inf": float(np.abs(A.T @ y + r - c).max()),
        "min_dual_slack": float(r.min()),
        "relative_duality_gap": abs(obj - dobj) / max(1.0, abs(obj), abs(dobj)),
    }
