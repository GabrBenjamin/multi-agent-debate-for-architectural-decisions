# guidearch_scorer.py
from __future__ import annotations

from typing import Dict, Any, Tuple, List, Optional
import math


# ============================================================
# Defaults / knobs
# ============================================================

PRIORITY_LABEL_TO_VALUE = {
    "L": 1.0,
    "M": 5.0,
    "H": 9.0,
}

PROFILE_WEIGHTS = {
    "balanced":     (1/3, 1/3, 1/3),
    "conservative": (0.20, 0.60, 0.20),  # emphasize risk
    "bold":         (0.20, 0.20, 0.60),  # emphasize opportunity
}


# Triangular anchors inside the native domain, as fractions of [min, max].
# Here, L/M/H mean LOW/MEDIUM/HIGH NATIVE VALUE, unless impact_semantics="benefit".
LMH_NATIVE_ANCHORS = {
    "L": (0.05, 0.15, 0.30),
    "M": (0.35, 0.50, 0.65),
    "H": (0.70, 0.85, 0.95),
}


# ============================================================
# Utility helpers
# ============================================================

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _normalize_weights(za: float, zn: float, zp: float) -> Tuple[float, float, float]:
    s = za + zn + zp
    if abs(s) <= 1e-12:
        return PROFILE_WEIGHTS["balanced"]
    return za / s, zn / s, zp / s


def _priority_to_numeric(raw: Any) -> float:
    if isinstance(raw, (int, float)):
        return _clamp(float(raw), 0.0, 10.0)
    label = str(raw or "M").strip().upper()
    return PRIORITY_LABEL_TO_VALUE.get(label, PRIORITY_LABEL_TO_VALUE["M"])


def _ensure_domain(domain: Optional[Dict[str, Any]]) -> Tuple[float, float]:
    domain = domain or {}
    dmin = _safe_float(domain.get("min", 0.0), 0.0)
    dmax = _safe_float(domain.get("max", 1.0), 1.0)
    if dmax <= dmin:
        return 0.0, 1.0
    return dmin, dmax


def _sort_triangle_native(t: Tuple[float, float, float]) -> Tuple[float, float, float]:
    a, b, c = sorted(float(x) for x in t)
    return (a, b, c)


def _clamp_triangle_native(
    t: Tuple[float, float, float],
    dmin: float,
    dmax: float,
) -> Tuple[float, float, float]:
    clamped = tuple(_clamp(float(x), dmin, dmax) for x in t)
    return _sort_triangle_native(clamped)


def tri_add(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def tri_weight(w: float, t: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (w * t[0], w * t[1], w * t[2])


def tri_center(t: Tuple[float, float, float]) -> float:
    return t[1]


def tri_risk(t: Tuple[float, float, float]) -> float:
    return t[2] - t[1]


def tri_opp(t: Tuple[float, float, float]) -> float:
    return t[1] - t[0]


def compose_triangles(
    triangles: List[Tuple[float, float, float]],
    compose: str,
) -> Tuple[float, float, float]:
    """
    Compose native-domain triangles.
    """
    if not triangles:
        return (0.0, 0.0, 0.0)

    mode = (compose or "sum").lower()

    if mode == "sum":
        out = (0.0, 0.0, 0.0)
        for t in triangles:
            out = tri_add(out, t)
        return out

    if mode == "product":
        out = (1.0, 1.0, 1.0)
        for t in triangles:
            out = (out[0] * t[0], out[1] * t[1], out[2] * t[2])
        return _sort_triangle_native(out)

    if mode == "min":
        return (
            min(t[0] for t in triangles),
            min(t[1] for t in triangles),
            min(t[2] for t in triangles),
        )

    if mode == "max":
        return (
            max(t[0] for t in triangles),
            max(t[1] for t in triangles),
            max(t[2] for t in triangles),
        )

    # Safe fallback
    out = (0.0, 0.0, 0.0)
    for t in triangles:
        out = tri_add(out, t)
    return out


# ============================================================
# Label / triangle interpretation
# ============================================================

def lmh_to_native_triangle(
    label: str,
    dmin: float,
    dmax: float,
    orientation: str,
    impact_semantics: str = "native_value",
) -> Tuple[float, float, float]:
    """
    Convert label L/M/H to native-domain triangle.

    impact_semantics:
      - "native_value": L/M/H mean low/med/high *native driver value*
      - "benefit":      L/M/H mean low/med/high *benefit to the decision*
                        so for min drivers, H maps to LOW native values
                        and for max drivers, H maps to HIGH native values
    """
    label = str(label or "M").strip().upper()
    if label not in LMH_NATIVE_ANCHORS:
        label = "M"

    semantic = (impact_semantics or "native_value").strip().lower()

    # Native-value semantics: preserve direct anchor meaning.
    native_label = label

    # Benefit semantics: for min drivers, invert label because lower native is better.
    if semantic == "benefit":
        orient = (orientation or "min").strip().lower()
        if orient == "min":
            invert = {"L": "H", "M": "M", "H": "L"}
            native_label = invert[label]
        else:
            native_label = label

    anchors = LMH_NATIVE_ANCHORS[native_label]
    span = dmax - dmin
    t = (
        dmin + anchors[0] * span,
        dmin + anchors[1] * span,
        dmin + anchors[2] * span,
    )
    return _clamp_triangle_native(t, dmin, dmax)


def normalize_driver_triangle_to_cost(
    native_t: Tuple[float, float, float],
    dmin: float,
    dmax: float,
    orientation: str,
) -> Tuple[float, float, float]:
    """
    Convert native triangle to cost in [0,1], lower is better.
    """
    span = dmax - dmin
    if span <= 1e-12:
        return (0.0, 0.0, 0.0)

    orient = (orientation or "min").strip().lower()

    if orient == "max":
        vals = [((dmax - x) / span) for x in native_t]
    else:
        vals = [((x - dmin) / span) for x in native_t]

    vals = [_clamp(v, 0.0, 1.0) for v in vals]
    vals.sort()
    return (vals[0], vals[1], vals[2])


def _neutral_native_triangle(dmin: float, dmax: float) -> Tuple[float, float, float]:
    """
    Safe neutral fallback: exact midpoint, clamped and ordered.
    """
    mid = (dmin + dmax) / 2.0
    return (mid, mid, mid)


# ============================================================
# Main scorer
# ============================================================

def compute_best_option(final_struct: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expected schema (compatible with your current pipeline):
    {
      "drivers": [
        {
          "name": "latency",
          "orientation": "min|max",
          "priority": "H|M|L" or number,
          "domain": {"min": 0, "max": 1000},
          "compose": "sum|product|min|max"
        }
      ],
      "options": [{"name": "A"}, {"name": "B"}],
      "impacts": [
        {"option":"A", "driver":"latency", "label":"H"},
        {"option":"B", "driver":"latency", "triangle":[300,450,700]}
      ],
      "constraints": [
        {"driver":"latency", "type":"max", "value": 700},
        {"driver":"availability", "type":"min", "value": 0.99}
      ],
      "weights": {"za":0.34, "zn":0.33, "zp":0.33},   # optional
      "risk_profile": "balanced|conservative|bold",   # optional
      "impact_semantics": "native_value|benefit",     # NEW: fixes LMH ambiguity
      "require_complete_impacts": true,               # NEW: default True
      "missing_impact_policy": "invalidate|neutral",  # NEW: default "invalidate"
      "include_invalid_totals": false                 # NEW: default False
    }
    """

    drivers = final_struct.get("drivers") or []
    options = final_struct.get("options") or []
    impacts = final_struct.get("impacts") or []
    constraints = final_struct.get("constraints") or []
    dependencies = final_struct.get("dependencies") or []
    conflicts = final_struct.get("conflicts") or []

    impact_semantics = str(final_struct.get("impact_semantics") or "native_value").strip().lower()
    require_complete_impacts = bool(final_struct.get("require_complete_impacts", True))
    missing_impact_policy = str(final_struct.get("missing_impact_policy") or "invalidate").strip().lower()
    include_invalid_totals = bool(final_struct.get("include_invalid_totals", False))

    # ------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------
    weights = final_struct.get("weights") or {}
    if all(k in weights for k in ("za", "zn", "zp")):
        w_a, w_n, w_p = _normalize_weights(
            _safe_float(weights.get("za"), 1/3),
            _safe_float(weights.get("zn"), 1/3),
            _safe_float(weights.get("zp"), 1/3),
        )
    else:
        profile = str(final_struct.get("risk_profile") or "balanced").strip().lower()
        w_a, w_n, w_p = PROFILE_WEIGHTS.get(profile, PROFILE_WEIGHTS["balanced"])
        w_a, w_n, w_p = _normalize_weights(w_a, w_n, w_p)

    # ------------------------------------------------------------
    # Driver metadata
    # ------------------------------------------------------------
    driver_meta: Dict[str, Dict[str, Any]] = {}
    for d in drivers:
        if not isinstance(d, dict) or not d.get("name"):
            continue
        name = str(d["name"])
        orient = str(d.get("orientation") or "min").strip().lower()
        if orient not in {"min", "max"}:
            orient = "min"

        prio = _priority_to_numeric(d.get("priority"))
        dmin, dmax = _ensure_domain(d.get("domain"))
        compose = str(d.get("compose") or "sum").strip().lower()
        if compose not in {"sum", "product", "min", "max"}:
            compose = "sum"

        driver_meta[name] = {
            "orientation": orient,
            "priority": prio,
            "dmin": dmin,
            "dmax": dmax,
            "compose": compose,
            "unit": d.get("unit"),
        }

    option_names = [str(o["name"]) for o in options if isinstance(o, dict) and o.get("name")]
    option_set = set(option_names)

    if not driver_meta or not option_names:
        return {
            "winner": None,
            "ranking": [],
            "details": {
                "error": "missing_drivers_or_options",
                "weights": {"za": w_a, "zn": w_n, "zp": w_p},
            }
        }

    total_prio = sum(m["priority"] for m in driver_meta.values()) or 1.0
    driver_w = {drv: meta["priority"] / total_prio for drv, meta in driver_meta.items()}

    # ------------------------------------------------------------
    # Collect base native triangles
    # ------------------------------------------------------------
    native_tri_by_option_driver: Dict[Tuple[str, str], List[Tuple[float, float, float]]] = {}

    for imp in impacts:
        if not isinstance(imp, dict):
            continue
        opt = imp.get("option")
        drv = imp.get("driver")
        if opt not in option_set or drv not in driver_meta:
            continue

        meta = driver_meta[drv]
        dmin, dmax = meta["dmin"], meta["dmax"]

        t_native: Optional[Tuple[float, float, float]] = None

        if "triangle" in imp and isinstance(imp["triangle"], (list, tuple)) and len(imp["triangle"]) == 3:
            try:
                raw_t = tuple(float(x) for x in imp["triangle"])
                t_native = _clamp_triangle_native(raw_t, dmin, dmax)
            except Exception:
                t_native = None
        else:
            label = imp.get("label")
            if label is not None:
                t_native = lmh_to_native_triangle(
                    label=str(label),
                    dmin=dmin,
                    dmax=dmax,
                    orientation=meta["orientation"],
                    impact_semantics=impact_semantics,
                )

        if t_native is not None:
            native_tri_by_option_driver.setdefault((opt, drv), []).append(t_native)

    base_native_triangles: Dict[str, Dict[str, Tuple[float, float, float]]] = {opt: {} for opt in option_names}
    for opt in option_names:
        for drv, meta in driver_meta.items():
            tris = native_tri_by_option_driver.get((opt, drv), [])
            if tris:
                base_native_triangles[opt][drv] = compose_triangles(tris, meta["compose"])

    # ------------------------------------------------------------
    # Dependencies: build closure
    # ------------------------------------------------------------
    from collections import defaultdict

    dep_map = defaultdict(list)
    for dep in dependencies:
        if not isinstance(dep, dict):
            continue
        opt = dep.get("option")
        reqs = dep.get("requires")
        if isinstance(reqs, str):
            reqs = [reqs]
        if opt not in option_set or not isinstance(reqs, list):
            continue
        clean_reqs = [r for r in reqs if isinstance(r, str) and r in option_set and r != opt]
        if clean_reqs:
            dep_map[opt].append(set(clean_reqs))

    closure_cache: Dict[str, Optional[set]] = {}
    visiting: set = set()
    invalid_deps: set = set()

    def resolve_closure(opt: str) -> Optional[set]:
        if opt in closure_cache:
            return closure_cache[opt]
        if opt in visiting:
            invalid_deps.update(visiting)
            closure_cache[opt] = None
            return None
        if opt not in option_set:
            invalid_deps.add(opt)
            closure_cache[opt] = None
            return None

        visiting.add(opt)
        comps = {opt}
        for reqs in dep_map.get(opt, []):
            for req in reqs:
                closure = resolve_closure(req)
                if closure is None:
                    invalid_deps.add(opt)
                    visiting.discard(opt)
                    closure_cache[opt] = None
                    return None
                comps.update(closure)
        visiting.discard(opt)
        closure_cache[opt] = comps
        return comps

    components_by_option: Dict[str, set] = {}
    for opt in option_names:
        closure = resolve_closure(opt)
        components_by_option[opt] = closure if closure is not None else {opt}

    # ------------------------------------------------------------
    # Conflicts
    # ------------------------------------------------------------
    conflict_sets: List[set] = []
    for conf in conflicts:
        if isinstance(conf, dict):
            members = conf.get("options") or conf.get("set") or conf.get("members") or []
        elif isinstance(conf, (list, tuple, set)):
            members = list(conf)
        elif isinstance(conf, str):
            members = [frag.strip() for frag in conf.split("|") if frag.strip()]
        else:
            continue

        clean = {m for m in members if isinstance(m, str) and m in option_set}
        if len(clean) >= 2:
            conflict_sets.append(clean)

    # ------------------------------------------------------------
    # Aggregate per option after dependency closure
    # ------------------------------------------------------------
    aggregated_native_triangles: Dict[str, Dict[str, Tuple[float, float, float]]] = {opt: {} for opt in option_names}
    validity: Dict[str, bool] = {opt: True for opt in option_names}
    invalid_reasons: Dict[str, List[str]] = {opt: [] for opt in option_names}
    rejected_constraints: Dict[str, List[Dict[str, Any]]] = {opt: [] for opt in option_names}
    missing_impacts: Dict[str, List[str]] = {opt: [] for opt in option_names}

    for opt in option_names:
        if opt in invalid_deps:
            validity[opt] = False
            invalid_reasons[opt].append("invalid_dependency_cycle_or_reference")

        closure = components_by_option.get(opt, {opt})

        # conflicts inside chosen closure
        for cset in conflict_sets:
            overlap = closure.intersection(cset)
            if len(overlap) > 1:
                validity[opt] = False
                invalid_reasons[opt].append(f"conflict:{'|'.join(sorted(overlap))}")

        for drv, meta in driver_meta.items():
            tris = []
            for comp_opt in closure:
                t = base_native_triangles.get(comp_opt, {}).get(drv)
                if t is not None:
                    tris.append(t)

            if tris:
                agg = compose_triangles(tris, meta["compose"])
                agg = _clamp_triangle_native(agg, meta["dmin"], meta["dmax"])
                aggregated_native_triangles[opt][drv] = agg
            else:
                missing_impacts[opt].append(drv)

        # missing impacts policy: one consistent place
        if require_complete_impacts and missing_impacts[opt]:
            if missing_impact_policy == "invalidate":
                validity[opt] = False
                invalid_reasons[opt].append(
                    "missing_impacts:" + ",".join(sorted(missing_impacts[opt]))
                )
            elif missing_impact_policy == "neutral":
                # fill neutral now so constraints/scoring see the same thing
                for drv in missing_impacts[opt]:
                    meta = driver_meta[drv]
                    aggregated_native_triangles[opt][drv] = _neutral_native_triangle(meta["dmin"], meta["dmax"])

    # ------------------------------------------------------------
    # Constraint checking (consistent with whatever native triangle exists now)
    # ------------------------------------------------------------
    for opt in option_names:
        if not validity[opt] and not include_invalid_totals:
            continue

        for c in constraints:
            if not isinstance(c, dict):
                continue
            drv = c.get("driver")
            if drv not in driver_meta:
                continue

            ctype = str(c.get("type") or "max").strip().lower()
            if ctype not in {"min", "max"}:
                ctype = "max"

            threshold = _safe_float(c.get("value"), math.nan)
            if math.isnan(threshold):
                continue

            meta = driver_meta[drv]
            threshold = _clamp(threshold, meta["dmin"], meta["dmax"])

            t = aggregated_native_triangles.get(opt, {}).get(drv)
            if t is None:
                # if still missing here, keep behavior consistent
                validity[opt] = False
                invalid_reasons[opt].append(f"constraint_missing_impact:{drv}")
                rejected_constraints[opt].append({
                    "driver": drv,
                    "type": ctype,
                    "value": threshold,
                    "reason": "missing_impact",
                })
                continue

            # pessimistic = worst native value after sorting
            pess = t[2]

            violated = False
            # NOTE:
            # type="max" means native value must be <= threshold
            # type="min" means native value must be >= threshold
            if ctype == "max" and pess > threshold + 1e-12:
                violated = True
            elif ctype == "min" and pess < threshold - 1e-12:
                violated = True

            if violated:
                validity[opt] = False
                invalid_reasons[opt].append(f"constraint_violation:{drv}")
                rejected_constraints[opt].append({
                    "driver": drv,
                    "type": ctype,
                    "value": threshold,
                    "pess_native": pess,
                })

    # ------------------------------------------------------------
    # Compute totals
    # ------------------------------------------------------------
    driver_norm_triangles: Dict[str, Dict[str, Tuple[float, float, float]]] = {opt: {} for opt in option_names}
    total_tri_by_option: Dict[str, Tuple[float, float, float]] = {}
    total_stats: Dict[str, Dict[str, Any]] = {}

    for opt in option_names:
        if not validity[opt] and not include_invalid_totals:
            continue

        total = (0.0, 0.0, 0.0)
        complete_for_total = True

        for drv, meta in driver_meta.items():
            t_native = aggregated_native_triangles.get(opt, {}).get(drv)
            if t_native is None:
                complete_for_total = False
                break

            t_norm = normalize_driver_triangle_to_cost(
                native_t=t_native,
                dmin=meta["dmin"],
                dmax=meta["dmax"],
                orientation=meta["orientation"],
            )
            driver_norm_triangles[opt][drv] = t_norm
            total = tri_add(total, tri_weight(driver_w[drv], t_norm))

        if complete_for_total:
            total_tri_by_option[opt] = total
            total_stats[opt] = {
                "tri": total,
                "za": tri_center(total),
                "zn": tri_risk(total),
                "zp": tri_opp(total),
            }

    # ------------------------------------------------------------
    # Select valid, fully scorable options
    # ------------------------------------------------------------
    valid_opts = [
        o for o in option_names
        if validity.get(o, False) and o in total_stats
    ]

    if not valid_opts:
        return {
            "winner": None,
            "ranking": [],
            "details": {
                "validity": validity,
                "missing_impacts": missing_impacts,
                "driver_native_triangles_base": base_native_triangles,
                "driver_native_triangles": aggregated_native_triangles,
                "driver_norm_triangles": driver_norm_triangles,
                "dependency_closure": {opt: sorted(list(v)) for opt, v in components_by_option.items()},
                "total": total_stats,
                "mu": {},
                "phi": {},
                "invalid_reasons": invalid_reasons,
                "rejected_constraints": rejected_constraints,
                "conflict_sets": [sorted(list(s)) for s in conflict_sets],
                "weights": {"za": w_a, "zn": w_n, "zp": w_p},
                "impact_semantics": impact_semantics,
            }
        }

    # ------------------------------------------------------------
    # Normalize z-values across valid options
    # ------------------------------------------------------------
    za_vals = [total_stats[o]["za"] for o in valid_opts]
    zn_vals = [total_stats[o]["zn"] for o in valid_opts]
    zp_vals = [total_stats[o]["zp"] for o in valid_opts]

    a_best, a_worst = min(za_vals), max(za_vals)
    n_best, n_worst = min(zn_vals), max(zn_vals)
    p_best, p_worst = max(zp_vals), min(zp_vals)  # higher opportunity is better

    def norm_lower_better(v: float, best: float, worst: float) -> float:
        if abs(worst - best) <= 1e-12:
            return 0.0
        return _clamp((v - best) / (worst - best), 0.0, 1.0)

    def norm_higher_better(v: float, best: float, worst: float) -> float:
        if abs(best - worst) <= 1e-12:
            return 0.0
        return _clamp((best - v) / (best - worst), 0.0, 1.0)

    mu: Dict[str, Dict[str, float]] = {}
    phi: Dict[str, float] = {}

    for o in valid_opts:
        mu_a = norm_lower_better(total_stats[o]["za"], a_best, a_worst)
        mu_n = norm_lower_better(total_stats[o]["zn"], n_best, n_worst)
        mu_p = norm_higher_better(total_stats[o]["zp"], p_best, p_worst)

        mu[o] = {
            "mu_a": mu_a,
            "mu_n": mu_n,
            "mu_p": mu_p,
        }

        phi[o] = max(
            w_a * mu_a,
            w_n * mu_n,
            w_p * mu_p,
        )

    ranking = [opt for opt, _ in sorted(phi.items(), key=lambda kv: (kv[1], kv[0]))]
    winner = ranking[0] if ranking else None

    return {
        "winner": winner,
        "ranking": ranking,
        "details": {
            "validity": validity,
            "missing_impacts": missing_impacts,
            "driver_native_triangles_base": base_native_triangles,
            "driver_native_triangles": aggregated_native_triangles,
            "driver_norm_triangles": driver_norm_triangles,
            "dependency_closure": {opt: sorted(list(v)) for opt, v in components_by_option.items()},
            "total": total_stats,
            "mu": mu,
            "phi": phi,
            "invalid_reasons": invalid_reasons,
            "rejected_constraints": rejected_constraints,
            "conflict_sets": [sorted(list(s)) for s in conflict_sets],
            "weights": {"za": w_a, "zn": w_n, "zp": w_p},
            "impact_semantics": impact_semantics,
        }
    }


# ============================================================
# Optional helper: criticality ranking
# ============================================================

def rank_criticality(
    scored: Dict[str, Any],
    top_t: int = 3,
    decay: float = 0.7,
) -> List[Dict[str, float]]:
    """
    Rough GuideArch-style criticality proxy:
    a driver is more critical if it has
      (a) high weighted impact on top-ranked options
      (b) high uncertainty (risk + opportunity width)

    Returns a list sorted from most critical to least critical.
    """
    ranking = scored.get("ranking") or []
    details = scored.get("details") or {}
    totals = details.get("driver_norm_triangles") or {}
    weights = details.get("weights") or {}
    validity = details.get("validity") or {}

    considered = [o for o in ranking if validity.get(o, False)][:max(1, top_t)]
    if not considered:
        return []

    driver_scores: Dict[str, float] = {}

    for rank_idx, opt in enumerate(considered):
        decay_w = decay ** rank_idx
        per_drv = totals.get(opt, {})
        for drv, t in per_drv.items():
            center = tri_center(t)
            uncertainty = tri_risk(t) + tri_opp(t)
            score = decay_w * (abs(center) + uncertainty)
            driver_scores[drv] = driver_scores.get(drv, 0.0) + score

    return [
        {"driver": drv, "criticality": val}
        for drv, val in sorted(driver_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    ]