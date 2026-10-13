"""Turn token attribution scores I_i into a token-level P2T reward.

`prm/attribution.py` computes how much each token moved its step's PRM score; this module
turns those scores into per-token rewards. Everything here is closed-form numpy.

For a step (block) s with reward R_s and policy tokens T_s, N_s = |T_s|, the token weights
are

    power    w_i = |I_i|^a      / sum_{j in T_s} |I_j|^a         (a = alpha, default 1)
    softmax  w_i = exp(|I_i|/T) / sum_{j in T_s} exp(|I_j|/T)    (T = temperature)

and eta = residual_coef (the paper's omega). Reward modes (`p2t_reward_mode`):

    v0   R_i = N_s * R_s * w_i
    v1   R_i = R_s + eta * w_i * R_s                              (paper Eq. 3)
    v2   R_i = R_s + eta * sign(I_i) * w_i * |R_s|
    v3   R_i = R_s + eta * sign(I_i) * lambda_i * |R_s|,  lambda normalized per sign group
    v4   R_i = R_s + eta * sign(I_i) * lambda_i * |R_s|,  sign groups rescaled to a common total

v0
--
Redistributes the step budget: sum_i R_i = N_s * R_s, and every token keeps R_s's sign.
Under power the weights are invariant to rescaling I, so the reward does not depend on the
PRM's attribution scale; softmax is scale sensitive.

v1
--
Uses the same unsigned weights as v0, applied as a residual on the flat step reward. Since
R_i = R_s * (1 + eta * w_i), every token keeps R_s's sign for any eta >= 0, and the step
total is N_s * R_s + eta * R_s. The residual is O(|R_s| / N_s), so eta sets its strength.

v2
--
Keeps the sign of I_i, so tokens that pushed the step score up are rewarded and tokens
that pulled it down are penalized. The step total is N_s * R_s + eta * |R_s| * rho, with
rho = sum_i sign(I_i) * w_i in [-1, 1]; rho is reported as `signed_asym`. For eta <= 1 no
token changes sign relative to R_s; eta > 1 can flip tokens (counted in `n_sign_flipped`).

v3
--
Normalizes the weights within each sign group:

    lambda_i = |I_i|^a / sum_{j : sign(I_j) == sign(I_i)} |I_j|^a   (damped by eps)

Credit and blame then each sum to ~1, so rho = O(eps) and the step total is N_s * R_s,
except on one-directional steps where rho = +-1. Because each group shares a total of 1
regardless of its size, the minority group is amplified by P / M (see `minority_frac`).

v4
--
Starts from the global share s_i = |I_i| / (sum_j |I_j| + eps) and rescales each sign
group to a common total B:

    lambda_i = (B / P) * s_i  if I_i > 0,   (B / M) * s_i  otherwise
    P = sum_{I_i>0} s_i,  M = sum_{I_i<0} s_i,  B = min(P, M) or (P + M) / 2 (`v4_balance`)

so rho = 0 exactly and the step total is N_s * R_s. With B = min no token gets more than
its global share, and the residual mass is 2 * minority_frac; B = mean keeps the full mass
and amplifies the minority by (P + M) / 2M. A step with an empty sign group gets no
residual.

Sharpness and the "auto" solver
-------------------------------
alpha / temperature control how concentrated the weights are: alpha = 0 gives uniform
weights (v0 then equals the flat process reward); larger alpha, or smaller T, concentrates
the budget on the highest-|I| tokens.

Sharpness is exp(entropy(w)) / N_s: 1.0 for uniform weights, 1 / N_s for a single token.
With `sharpness_target` set, the shape parameter is solved per step by bisection so that
sharpness stays at or above the target. It is a cap: power uses min(alpha, solved) and
softmax max(T_0, solved), so the solver only flattens over-concentrated steps. The bound is
alpha = 1 for power and T_0 = mean|I| over the step for softmax ("auto"). Bounding the
sharpness bounds max w_i, which reduces clipping under v0. Under v1-v4 the weights set the
residual's amplitude, so a target weakens the signal.

Epsilon
-------
eps is a degeneracy threshold on the raw sum|I|: a step with sum|I| <= eps uses uniform
weights and no signed residual; otherwise the weights sum to exactly 1. Steps can reach
sum|I| == 0, e.g. a block containing only `\\n\\n` delimiters.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Sequence, Tuple, Union

import numpy as np

P2T_REWARD_MODES = ("v0", "v1", "v2", "v3", "v4")
P2T_WEIGHTINGS = ("power", "softmax")
# v4's common total B for the two sign groups: "min" never amplifies, "mean" keeps the
# full residual mass.
P2T_V4_BALANCES = ("min", "mean")

__all__ = [
    "P2T_REWARD_MODES",
    "P2T_WEIGHTINGS",
    "P2T_V4_BALANCES",
    "P2TShape",
    "p2t_token_reward",
    "p2t_token_reward_batch",
    "validate_p2t_config",
]


_AUTO_ALPHA_CAP = 1.0        # upper bound of alpha="auto"; softmax uses mean|I| per step


def _sharpness(weight: np.ndarray) -> float:
    """exp(entropy(weight)) / N_s: 1.0 for uniform weights, 1/N_s for a single token."""
    nz = weight[weight > 0]
    return float(np.exp(-(nz * np.log(nz)).sum())) / weight.size


def _power_weights(share: np.ndarray, alpha: float) -> np.ndarray:
    """Normalised |I|^alpha weights. `share` is |I| / max|I|."""
    s = share ** alpha if alpha != 1.0 else share
    return s / s.sum()


def _softmax_weights(mag: np.ndarray, t: float) -> np.ndarray:
    """Normalised exp(|I|/T) weights."""
    z = mag / t
    z -= z.max()                        # numerical stability
    w = np.exp(z)
    return w / w.sum()


def _solve_alpha(share: np.ndarray, target: float, cap: float,
                 tol: float = 1e-3, iters: int = 30) -> float:
    """Largest alpha <= `cap` whose sharpness stays at or above `target`.

    Sharpness decreases monotonically in alpha, so bisection finds a unique solution. A
    step that already meets the target at `cap` is returned at `cap`.
    """
    if _sharpness(_power_weights(share, cap)) >= target:
        return cap
    lo, hi = 0.0, cap                               # sharpness(lo) >= target > sharpness(hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s = _sharpness(_power_weights(share, mid))
        if abs(s - target) <= tol:
            return mid
        lo, hi = (mid, hi) if s > target else (lo, mid)
    return 0.5 * (lo + hi)


def _solve_temperature(mag: np.ndarray, target: float, t0: float,
                       tol: float = 1e-3, iters: int = 30) -> float:
    """Smallest T >= `t0` whose sharpness reaches `target`.

    Sharpness increases monotonically in T, so bisection finds a unique solution. A step
    that already meets the target at `t0` (mean|I| over the step) is returned at `t0`.
    """
    if _sharpness(_softmax_weights(mag, t0)) >= target:
        return t0
    hi = t0
    for _ in range(60):                             # bracket upward
        hi *= 2.0
        if _sharpness(_softmax_weights(mag, hi)) >= target:
            break
    lo = t0                                         # sharpness(lo) < target <= sharpness(hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s = _sharpness(_softmax_weights(mag, mid))
        if abs(s - target) <= tol:
            return mid
        lo, hi = (lo, mid) if s > target else (mid, hi)
    return 0.5 * (lo + hi)


def _pool_weights(mag: np.ndarray,
                  weighting: str,
                  alpha_cap: float,
                  temp: Union[float, None],
                  target: Union[float, None]) -> Tuple[np.ndarray, bool, float]:
    """Weights over one pool of tokens, summing to 1.

    v0/v1/v2/v4 pass the whole step; v3 passes each sign group separately. Requires
    mag.size > 0 and mag.sum() > 0.

    Returns (weight, solved, param): `solved` is whether the solver moved the shape
    parameter off its bound, and `param` is alpha under power and T / mean|I| under softmax.
    """
    if weighting == "power":
        # Normalize by the max to avoid overflow / underflow for large alpha.
        share = mag / mag.max()
        a = alpha_cap if target is None else _solve_alpha(share, target, alpha_cap)
        return _power_weights(share, a), a < alpha_cap, float(a)
    # temp is None when temperature="auto", which requires a target.
    mean_mag = float(mag.mean())
    t0 = temp if temp is not None else mean_mag
    t = t0 if target is None else _solve_temperature(mag, target, t0)
    return _softmax_weights(mag, t), t > t0, float(t / mean_mag)


def _resolve_sharpness_target(sharpness_target: Union[float, None]) -> Union[float, None]:
    """Validate `sharpness_target`; None disables the solver."""
    if sharpness_target is None:
        return None
    t = float(sharpness_target)
    if not np.isfinite(t) or not 0.0 < t <= 1.0:
        raise ValueError(
            f"p2t_sharpness_target must be in (0, 1] (1.0 = uniform weights), got {t}")
    return t


def _resolve_alpha(alpha: Union[float, str], target: Union[float, None]) -> float:
    """Validate power's alpha and return the exponent (the solver's bound with a target).

    `alpha` is a number >= 0, or "auto" to solve it per step with 1.0 as the bound.
    """
    if isinstance(alpha, str):
        if alpha.strip().lower() != "auto":
            raise ValueError(f"p2t_alpha must be a number >= 0 or \"auto\", got {alpha!r}")
        if target is None:
            raise ValueError(
                "p2t_alpha=\"auto\" needs p2t_sharpness_target -- the solver has no target "
                "to aim at. Use a fraction in (0, 1], e.g. 0.9."
            )
        return _AUTO_ALPHA_CAP
    cap = float(alpha)
    if not np.isfinite(cap) or cap < 0:
        raise ValueError(f"p2t_alpha must be >= 0 (0 = uniform, 1 = v0), got {cap}")
    return cap


def _resolve_temperature(temperature: Union[float, str],
                         target: Union[float, None]) -> Union[float, None]:
    """Validate softmax's temperature; return None for "auto" (solved per step).

    A numeric T cannot be combined with a target: T carries the unit of |I|, so it is not
    a meaningful bound for the solver.
    """
    if isinstance(temperature, str):
        if temperature.strip().lower() != "auto":
            raise ValueError(
                f"p2t_temperature must be a positive number or \"auto\", got {temperature!r}"
            )
        if target is None:
            raise ValueError(
                "p2t_temperature=\"auto\" needs p2t_sharpness_target -- the solver has no "
                "target to aim at. Use a fraction in (0, 1], e.g. 0.9."
            )
        return None
    t = float(temperature)
    if not np.isfinite(t) or t <= 0:
        raise ValueError(f"p2t_temperature must be > 0 (it divides |I|), got {t}")
    if target is not None:
        raise ValueError(
            f"p2t_temperature={t} cannot be combined with p2t_sharpness_target: an absolute "
            "T is not a scale-free bound for the solver (see _resolve_temperature). Use "
            "p2t_temperature=\"auto\" to solve T per step, or drop the target to pin T."
        )
    return t


class P2TShape(NamedTuple):
    """Resolved reward knobs."""
    target: Union[float, None]        # None = solver off
    alpha_cap: float                  # power: the exponent, and the solver's upper bound
    temperature: Union[float, None]   # softmax: T; None = "auto", solved per step
    residual_coef: float              # v1-v4: eta (ignored by v0)
    v4_balance: str                   # v4: common total of the two sign groups
    warnings: Tuple[str, ...]         # notes about ignored knobs; not fatal


def _is_plain_one(value) -> bool:
    """True for a numeric 1.0 (the default of the shape knobs); bools are excluded."""
    return not isinstance(value, (str, bool)) and float(value) == 1.0


def validate_p2t_config(weighting: str,
                        alpha: Union[float, str] = 1.0,
                        temperature: Union[float, str] = 1.0,
                        sharpness_target: Union[float, None] = None,
                        mode: str = "v0",
                        residual_coef: float = 1.0,
                        v4_balance: str = "min") -> P2TShape:
    """Resolve and check the reward knobs. Raises ValueError on an invalid combination.

    Called by `p2t_token_reward` on every batch and by the trainer at startup. Knobs that
    the selected weighting / mode does not read are not validated; a non-default value
    there is reported in `warnings` instead.
    """
    if mode not in P2T_REWARD_MODES:
        raise ValueError(f"unknown p2t_reward_mode: {mode!r}, want one of {P2T_REWARD_MODES}")
    if weighting not in P2T_WEIGHTINGS:
        raise ValueError(f"unknown p2t_weighting: {weighting!r}, want one of {P2T_WEIGHTINGS}")
    target = _resolve_sharpness_target(sharpness_target)
    warns = []
    if weighting == "power":
        alpha_cap, temp = _resolve_alpha(alpha, target), None
        if not _is_plain_one(temperature):
            warns.append(f"p2t_temperature={temperature!r} is IGNORED under "
                         f"p2t_weighting=power, which reads p2t_alpha instead")
    else:
        alpha_cap, temp = _AUTO_ALPHA_CAP, _resolve_temperature(temperature, target)
        if not _is_plain_one(alpha):
            warns.append(f"p2t_alpha={alpha!r} is IGNORED under p2t_weighting=softmax, "
                         f"which reads p2t_temperature instead")
    if mode == "v4":
        if v4_balance not in P2T_V4_BALANCES:
            raise ValueError(f"unknown p2t_v4_balance: {v4_balance!r}, "
                             f"want one of {P2T_V4_BALANCES}")
        balance = v4_balance
    else:
        balance = "min"
        if v4_balance != "min":
            warns.append(f"p2t_v4_balance={v4_balance!r} is IGNORED under "
                         f"p2t_reward_mode={mode}, which has no sign-group balancing")
    if mode in ("v1", "v2", "v3", "v4"):
        try:
            eta = float(residual_coef)
        except (TypeError, ValueError):
            raise ValueError(f"p2t_residual_coef must be a number, "
                             f"got {residual_coef!r}") from None
        if not np.isfinite(eta) or eta < 0:
            raise ValueError(
                f"p2t_residual_coef must be >= 0 (0 = flat process reward), got {eta}")
    else:
        eta = 1.0                       # unused by v0
        if not _is_plain_one(residual_coef):
            warns.append(f"p2t_residual_coef={residual_coef!r} is IGNORED under "
                         f"p2t_reward_mode={mode}, which has no residual term")
    return P2TShape(target, alpha_cap, temp, eta, balance, tuple(warns))


def p2t_token_reward(
    influence: np.ndarray,           # (L,) I_i per token; 0 where nothing was mapped
    step_idx: np.ndarray,            # (L,) owning block id per token; -1 = none
    step_rewards: Sequence[float],   # R_s per block, indexed by block id
    *,
    mode: str = "v0",
    weighting: str = "power",        # power: |I_i|^alpha | softmax: exp(|I_i|/T)
    alpha: Union[float, str] = 1.0,  # power only: exponent on |I_i|; "auto" = solve per step
    temperature: Union[float, str] = 1.0,   # softmax only: T; "auto" = solve per step
    sharpness_target: Union[float, None] = None,   # target sharpness for the solver
    residual_coef: float = 1.0,      # v1-v4: eta, the residual's strength
    v4_balance: str = "min",         # v4 only: the sign groups' common total B
    epsilon: float = 1e-6,
    clip: float = 0.0,               # 0 = off; else clip R_i into [-clip, clip]
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute the per-token P2T reward of one response.

    Args:
        influence: per-token attribution score I_i, on the same index space as `step_idx`.
        step_idx: per-token block id. Tokens at -1 (EOS / special / unplaced) get 0 here;
            the caller assigns them the last block's R_s.
        step_rewards: R_s of block k at index k. Blocks that own no token are skipped.
        mode: "v0" | "v1" | "v2" | "v3" | "v4"; see the module docstring.
        weighting: "power" (|I_i|^alpha, reads `alpha`) or "softmax" (exp(|I_i|/T), reads
            `temperature`).
        alpha: power only. Exponent on |I_i|: 0 is uniform, 1 is the plain formula, larger
            values concentrate the reward on the highest-|I| tokens. With
            `sharpness_target` set it is the solver's upper bound; "auto" uses 1.0.
        temperature: softmax only. A positive T, or "auto" to solve T per step (requires
            `sharpness_target`).
        sharpness_target: None disables the solver. Otherwise a fraction in (0, 1]: per
            step, the shape parameter is chosen so that sharpness stays at or above this
            value. Under v3 each sign group is solved separately.
        residual_coef: v1-v4 only. eta, the residual's strength; 0 reduces the reward to
            the flat process reward. Under v2-v4, eta > 1 can flip a token's sign.
        v4_balance: v4 only. "min" sets B = min(P, M); "mean" sets B = (P + M) / 2.
        epsilon: degeneracy threshold. A block with sum|I| <= epsilon uses uniform weights:
            R_s under v0, R_s * (1 + eta / N_s) under v1, and R_s under v2 / v3 / v4. Under
            v3 it also damps each group's normalization.
        clip: clip R_i into [-clip, clip]. The clipped amount is dropped and reported as
            `mass_err`.

    Returns:
        reward: (L,) float32 per-token R_i, 0 outside every block.
        stats: summed over steps:
            n_steps: blocks that own at least one token.
            n_uniform: degenerate blocks that fell back to uniform weights.
            n_clipped: tokens changed by `clip`.
            n_sign_flipped: tokens whose R_i has the opposite sign to R_s (v2-v4, eta > 1).
            mass_err: sum of R_i after clipping minus before; 0 without clipping.
            signed_asym_sum / signed_asym_abs_sum: rho = sum_i sign(I_i) * lambda_i per
                step and its absolute value. Under v2 the budget error is eta*|R_s|*rho.
            minority_frac_sum / n_minority_steps: v3 / v4 only, min(P, M) / (P + M).
            sharpness_sum: per-step sharpness of the weights.
            n_solve_steps / solve_param_sum / n_solved: non-degenerate blocks, the shape
                parameter used (alpha, or T / mean|I|), and how many blocks the solver
                moved off the bound.
    """
    shape = validate_p2t_config(weighting, alpha, temperature, sharpness_target,
                                mode, residual_coef, v4_balance)
    target = shape.target                # None = solver off
    alpha_cap = shape.alpha_cap
    temp = shape.temperature             # None = "auto" -> solve T per step
    eta = shape.residual_coef            # v1-v4
    balance = shape.v4_balance           # v4 only
    if epsilon <= 0:
        raise ValueError(f"p2t_epsilon must be > 0 (it guards a division), got {epsilon}")
    if clip < 0:
        raise ValueError(f"p2t_clip must be >= 0 (0 = off), got {clip}")

    influence = np.asarray(influence)
    step_idx = np.asarray(step_idx)
    reward = np.zeros(len(influence), dtype=np.float32)
    n_steps = n_uniform = n_clipped = n_sign_flipped = 0
    n_solve_steps = n_solved = n_minority_steps = 0
    mass_ref = mass_out = sharpness_sum = solve_param_sum = 0.0
    signed_asym_sum = signed_asym_abs_sum = minority_frac_sum = 0.0

    for k, Rs in enumerate(step_rewards):
        idx = np.nonzero(step_idx == k)[0]
        if idx.size == 0:
            continue                                # empty block: owns no token
        n_steps += 1
        signed = influence[idx].astype(np.float64)
        mag = np.abs(signed)
        total = float(mag.sum())
        degenerate = total <= epsilon
        # weight: unsigned weights (v0, v1). signed_share: sign(I_i) * lambda_i (v2-v4).
        if degenerate:
            weight = np.full(idx.size, 1.0 / idx.size)
            n_uniform += 1
            signed_share = np.zeros(idx.size)       # no signed residual
            sharp = 1.0
        elif mode != "v3":
            # v0 / v1 / v2 / v4: one pool over the whole step.
            weight, solved, param = _pool_weights(mag, weighting, alpha_cap, temp, target)
            n_solved += int(solved)
            solve_param_sum += param
            n_solve_steps += 1
            sharp = _sharpness(weight)
            if mode != "v4":
                signed_share = np.sign(signed) * weight
            else:
                # v4: rescale both sign groups to a common total B, so that
                # sum_i sign * lambda = B - B = 0.
                pos, neg = signed > 0.0, signed < 0.0
                p_w, m_w = float(weight[pos].sum()), float(weight[neg].sum())
                signed_share = np.zeros(idx.size)
                if p_w > 0.0 and m_w > 0.0:
                    b = min(p_w, m_w) if balance == "min" else 0.5 * (p_w + m_w)
                    # s_i = |I_i| / (sum|I| + eps)
                    damp = total / (total + epsilon)
                    signed_share[pos] = (b / p_w) * weight[pos] * damp
                    signed_share[neg] = -(b / m_w) * weight[neg] * damp
                # else: one-directional step, no residual.
                denom = p_w + m_w
                if denom > 0.0:
                    minority_frac_sum += min(p_w, m_w) / denom
                n_minority_steps += 1
        else:
            # v3: normalize within each sign group.
            signed_share = np.zeros(idx.size)
            weight = np.full(idx.size, 1.0 / idx.size)   # unused by v3
            sharp = param = 0.0
            solved = False
            # Tokens with I_i == 0 join neither group.
            pos, neg = signed > 0.0, signed < 0.0
            n_signed = int(np.count_nonzero(pos | neg))   # > 0: the step is not degenerate
            for m, sgn in ((pos, 1.0), (neg, -1.0)):
                gmag = mag[m]
                gtot = float(gmag.sum())
                if gmag.size == 0 or gtot <= 0.0:
                    continue
                w, g_solved, g_param = _pool_weights(gmag, weighting, alpha_cap, temp,
                                                     target)
                # lambda_i = |I_i| / (sum_group |I_j| + eps), with eps on the raw |I| scale.
                w = w * (gtot / (gtot + epsilon))
                signed_share[m] = sgn * w
                # Telemetry: average over the two groups, weighted by group size.
                frac = gmag.size / n_signed
                sharp += frac * _sharpness(w)
                param += frac * g_param
                solved = solved or g_solved
            n_solved += int(solved)
            solve_param_sum += param
            n_solve_steps += 1
            pos_tot = float(mag[pos].sum())
            if total > 0.0:
                minority_frac_sum += min(pos_tot, total - pos_tot) / total
            n_minority_steps += 1
        sharpness_sum += sharp
        rho = float(signed_share.sum())
        signed_asym_sum += rho
        signed_asym_abs_sum += abs(rho)
        if mode == "v0":
            r = idx.size * float(Rs) * weight
            mass_ref += idx.size * float(Rs)
        elif mode == "v1":
            r = float(Rs) * (1.0 + eta * weight)
            mass_ref += float(r.sum())
        else:
            # v2 / v3 / v4: baseline + signed residual.
            r = float(Rs) + eta * signed_share * abs(float(Rs))
            mass_ref += float(r.sum())
            if Rs != 0.0:
                n_sign_flipped += int(np.count_nonzero(r * float(Rs) < 0.0))
        if clip > 0.0:
            n_clipped += int(np.count_nonzero(np.abs(r) > clip))
            r = np.clip(r, -clip, clip)
        reward[idx] = r
        mass_out += float(r.sum())

    stats = {
        "n_steps": float(n_steps),
        "n_uniform": float(n_uniform),
        "n_clipped": float(n_clipped),
        "n_sign_flipped": float(n_sign_flipped),
        "mass_err": mass_out - mass_ref,
        "sharpness_sum": sharpness_sum,
        "signed_asym_sum": signed_asym_sum,
        "signed_asym_abs_sum": signed_asym_abs_sum,
        "minority_frac_sum": minority_frac_sum,       # v3 / v4 only
        "n_minority_steps": float(n_minority_steps),
        "n_solve_steps": float(n_solve_steps),
        "n_solved": float(n_solved),
        "solve_param_sum": solve_param_sum,
    }
    return reward, stats


def p2t_token_reward_batch(
    influence: np.ndarray,       # (bs, L) I_i, 0 where nothing was mapped
    step_idx: np.ndarray,        # (bs, L) owning block id; -1 = none (EOS / special / pad)
    process_reward: np.ndarray,  # (bs, L) R_s broadcast within each step
    valid_mask: np.ndarray,      # (bs, L) tokens eligible to take part
    *,
    mode: str = "v0",
    weighting: str = "power",
    alpha: Union[float, str] = 1.0,
    temperature: Union[float, str] = 1.0,
    sharpness_target: Union[float, None] = None,
    residual_coef: float = 1.0,
    v4_balance: str = "min",
    epsilon: float = 1e-6,
    clip: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Apply `p2t_token_reward` to every response of a batch.

    Runs on the driver, before repeat calibration and tail trimming, which are applied
    afterwards as masks on the result.

    Args:
        influence: per-token I_i.
        step_idx: per-token block id. Valid tokens at -1 inherit `process_reward`.
        process_reward: per-token R_s, constant within a step.
        valid_mask: tokens eligible to take part; tokens outside it never join a step.
        mode / weighting / alpha / temperature / sharpness_target / residual_coef /
            v4_balance / epsilon / clip: forwarded to `p2t_token_reward`.

    Returns:
        reward: (bs, L) float32. Zero outside `valid_mask`.
        stats: the stats of `p2t_token_reward` summed over the batch. mass_err is summed
            as |per-response error|.
    """
    influence = np.asarray(influence)
    step_idx = np.asarray(step_idx)
    process_reward = np.asarray(process_reward)
    valid_mask = np.asarray(valid_mask)
    bs, length = influence.shape

    reward = np.zeros((bs, length), dtype=np.float32)
    agg = {"n_steps": 0.0, "n_uniform": 0.0, "n_clipped": 0.0, "n_sign_flipped": 0.0,
           "mass_err": 0.0, "sharpness_sum": 0.0,
           "signed_asym_sum": 0.0, "signed_asym_abs_sum": 0.0,
           "minority_frac_sum": 0.0, "n_minority_steps": 0.0,
           "n_solve_steps": 0.0, "n_solved": 0.0, "solve_param_sum": 0.0}

    for i in range(bs):
        sidx = step_idx[i].astype(np.int64)
        valid = valid_mask[i].astype(bool)
        prew = process_reward[i]

        # Exclude invalid tokens from every step.
        sidx_redis = np.where(valid, sidx, -1)

        present = sidx_redis >= 0
        if present.any():
            # R_s per block id (constant within a step).
            n_blocks = int(sidx_redis.max()) + 1
            step_rewards = np.zeros(n_blocks, dtype=np.float64)
            step_rewards[sidx_redis[present]] = prew[present]
            r, st = p2t_token_reward(
                influence[i], sidx_redis, step_rewards,
                mode=mode, weighting=weighting, alpha=alpha,
                temperature=temperature, sharpness_target=sharpness_target,
                residual_coef=residual_coef, v4_balance=v4_balance,
                epsilon=epsilon, clip=clip,
            )
            reward[i] = r
            for key in agg:
                agg[key] += abs(st[key]) if key == "mass_err" else st[key]

        # Tokens outside every step (e.g. EOS) inherit R_s, as in the flat process reward.
        special = (sidx < 0) & valid
        reward[i][special] = prew[special]

    return reward, agg
