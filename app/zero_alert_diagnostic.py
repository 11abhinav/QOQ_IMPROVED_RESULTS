# =====================================================================================
# app/zero_alert_diagnostic.py
# INSTITUTIONAL ZERO-ALERT DIAGNOSTIC & CONSERVATION FUNNEL ENGINE
#
# Provides:
# 1. SingleTerminalTracker — Mutually-exclusive, single-disposition tracking guaranteeing
#    exact conservation of the scanner universe (Sum of Terminal Outcomes == Universe Size).
# 2. StageWaterfallTracker — Calculates population attrition across sequential stages and
#    identifies dominant bottleneck by relative failure rate (eliminated / entered).
# 3. classify_zero_alert_run — Categorizes zero-alert runs into LEGITIMATE_ZERO,
#    SUSPICIOUS_ZERO, CRITICAL_ZERO, or DATA_OR_ENGINE_FAILURE.
# =====================================================================================

import threading
import logging
from typing import Dict, List, Optional, Any, Set, Tuple

logger = logging.getLogger("zero_alert_diagnostic")


class SingleTerminalTracker:
    """
    Guarantees that every stock evaluated in a scanner universe receives EXACTLY ONE
    terminal disposition (first decisive failure wins). Enforces mathematical conservation:
    Universe Size == sum(terminal_dispositions) with Delta == 0.
    """
    def __init__(self, universe: Any, scanner_name: str = "SCANNER"):
        self.scanner_name = scanner_name
        if isinstance(universe, (list, set, tuple)):
            self.universe_symbols: Set[str] = set(universe)
            self.total_universe: int = len(self.universe_symbols)
        elif isinstance(universe, int):
            self.universe_symbols = set()
            self.total_universe = universe
        else:
            self.universe_symbols = set()
            self.total_universe = 0

        self._lock = threading.Lock()
        self._dispositions: Dict[str, Tuple[str, str, Optional[str]]] = {}  # symbol -> (gate, reason, stage)
        self._counts: Dict[str, int] = {}                                    # gate -> count
        self._gate_to_stage: Dict[str, str] = {}                             # gate -> stage
        self._stage_terminals: Dict[str, Dict[str, int]] = {}                # stage -> subgate -> count

    def map_gates_to_stage(self, stage: str, gates: List[str]) -> None:
        """Associates a list of terminal gates/dispositions with a specific pipeline stage."""
        with self._lock:
            for g in gates:
                self._gate_to_stage[g] = stage

    def _resolve_subgate(self, gate: str, reason: str) -> str:
        """Resolves generic gate codes into descriptive sub-reasons when present."""
        if gate == "RISK_REJECTED":
            r_low = reason.lower()
            if "no_valid_structural_target" in r_low or "min rr" in r_low or "natural_rr" in r_low or "poor rr" in r_low:
                return "POOR_RR"
            elif "sl_outside_max" in r_low or "wide" in r_low or "max_stop_loss" in r_low:
                return "WIDE_SL"
            elif "sl_inside_min" in r_low or "tight" in r_low or "min_stop_loss" in r_low:
                return "TIGHT_SL"
            elif "unordered" in r_low:
                return "UNORDERED_TARGETS"
            elif "invalid_atr" in r_low:
                return "INVALID_ATR"
            return "RISK_REJECTED"
        if gate.startswith("ENTRY_CONFIRM_FAILED: "):
            return gate.split(":", 1)[1].strip()
        if gate.startswith("LIVE_ENTRY_FAILED: "):
            return gate.split(":", 1)[1].strip()
        return gate

    def record_terminal(self, symbol: str, gate: str, reason: str = "", stage: Optional[str] = None) -> bool:
        """
        Records the terminal disposition for a symbol.
        Returns True if this was the first recording for this symbol.
        Returns False if the symbol already has an assigned terminal disposition (first-fail wins).
        """
        with self._lock:
            if symbol in self._dispositions:
                return False  # Already assigned terminal outcome, ignore downstream gates

            assigned_stage = stage or self._gate_to_stage.get(gate)
            self._dispositions[symbol] = (gate, reason, assigned_stage)
            self._counts[gate] = self._counts.get(gate, 0) + 1

            if assigned_stage:
                if assigned_stage not in self._stage_terminals:
                    self._stage_terminals[assigned_stage] = {}
                subgate = self._resolve_subgate(gate, reason)
                self._stage_terminals[assigned_stage][subgate] = (
                    self._stage_terminals[assigned_stage].get(subgate, 0) + 1
                )
            return True

    def get_stage_terminal_breakdown(self, stage_name: str) -> Dict[str, int]:
        """
        Returns the granular breakdown of terminal outcomes that occurred within a given stage.
        """
        with self._lock:
            if stage_name in self._stage_terminals and self._stage_terminals[stage_name]:
                return dict(sorted(self._stage_terminals[stage_name].items(), key=lambda x: x[1], reverse=True))

            # Fallback: scan all recorded dispositions where stage matches
            breakdown: Dict[str, int] = {}
            for sym, (gate, reason, stg) in self._dispositions.items():
                target_stg = stg or self._gate_to_stage.get(gate)
                if target_stg == stage_name:
                    subgate = self._resolve_subgate(gate, reason)
                    breakdown[subgate] = breakdown.get(subgate, 0) + 1
            return dict(sorted(breakdown.items(), key=lambda x: x[1], reverse=True))

    def record_untracked_remainder(self, default_gate: str = "UNTRACKED_DROP") -> int:
        """Assigns default_gate to any symbol in universe_symbols that did not receive a disposition."""
        untracked = 0
        with self._lock:
            for s in self.universe_symbols:
                if s not in self._dispositions:
                    self._dispositions[s] = (default_gate, "No explicit terminal assigned")
                    self._counts[default_gate] = self._counts.get(default_gate, 0) + 1
                    untracked += 1
        return untracked

    def get_summary(self) -> Dict[str, Any]:
        """
        Returns conservation accounting summary enforcing:
        Input == Passed + Rejected + Skipped + Error (Delta == 0).
        """
        with self._lock:
            sum_terminal = sum(self._counts.values())
            delta = self.total_universe - sum_terminal
            untracked = self._counts.get("UNTRACKED_DROP", 0)

            passed_count = 0
            skipped_count = 0
            error_count = 0
            rejected_count = 0

            for gate, count in self._counts.items():
                if gate == "UNTRACKED_DROP":
                    continue
                g_upper = gate.upper()
                if g_upper in ("ALERT_GENERATED", "PASSED", "SELECTED", "ALERT_TRIGGERED"):
                    passed_count += count
                elif any(w in g_upper for w in ("SKIP", "DUPLICATE", "ALREADY_ALERTED", "ALREADY_OPEN", "COOLDOWN", "SUPPRESSED", "ADMIN_STOP")):
                    skipped_count += count
                elif any(w in g_upper for w in ("ERROR", "FAIL", "EXCEPTION", "OUTAGE", "PIPELINE_FAILED", "INSERT_FAILED")):
                    if any(w in g_upper for w in ("QUALITY_GATE", "FUNDAMENTAL_FAIL", "SCORE_FAIL", "ENTRY_CONFIRM_FAILED", "LIVE_ENTRY_FAILED", "ATR_FAIL", "SUPPORT_FAIL", "VOLUME_FAIL", "LIQUIDITY_FAIL", "STRUCTURE_FAIL")):
                        rejected_count += count
                    else:
                        error_count += count
                else:
                    rejected_count += count

            reconciled = (self.total_universe == (passed_count + rejected_count + skipped_count + error_count + untracked))

            return {
                "total_universe": self.total_universe,
                "terminal_counts": dict(sorted(self._counts.items(), key=lambda x: x[1], reverse=True)),
                "sum_terminal": sum_terminal,
                "conservation_delta": delta,
                "untracked_drop": untracked,
                "passed_count": passed_count,
                "rejected_count": rejected_count,
                "skipped_count": skipped_count,
                "error_count": error_count,
                "is_conserved": (delta == 0 and untracked == 0 and reconciled),
                "recorded_symbols_count": len(self._dispositions)
            }

    # Alias for backward compatibility with callers expecting get_conservation_summary()
    get_conservation_summary = get_summary


class StageWaterfallTracker:
    """
    Tracks populations entering and surviving sequential pipeline stages:
    Stage 1 -> Stage 2 -> Stage 3 -> ... -> Alerts.
    Computes attrition rates and identifies the Dominant Bottleneck Gate.
    """
    def __init__(self, stage_order: Optional[List[str]] = None):
        self._stage_order: List[str] = stage_order or []
        self._stage_counts: Dict[str, int] = {}
        self._lock = threading.Lock()

    def set_stage_count(self, stage_name: str, count: int) -> None:
        with self._lock:
            if stage_name not in self._stage_order:
                self._stage_order.append(stage_name)
            self._stage_counts[stage_name] = count

    def compute_attrition(self) -> List[Dict[str, Any]]:
        """
        Computes stage-by-stage attrition:
        loss = entered - passed
        loss_pct = (loss / entered) * 100
        """
        with self._lock:
            results = []
            for i in range(len(self._stage_order) - 1):
                cur_stage = self._stage_order[i]
                next_stage = self._stage_order[i + 1]
                entered = self._stage_counts.get(cur_stage, 0)
                passed = self._stage_counts.get(next_stage, 0)
                eliminated = max(0, entered - passed)
                attrition_pct = (eliminated / entered * 100.0) if entered > 0 else 0.0
                results.append({
                    "stage": cur_stage,
                    "next_stage": next_stage,
                    "entered": entered,
                    "passed": passed,
                    "eliminated": eliminated,
                    "attrition_pct": round(attrition_pct, 1)
                })
            return results

    def get_dominant_bottleneck(self) -> Optional[Dict[str, Any]]:
        """
        Returns the gate with the highest relative loss rate (eliminated / entered)
        among stages where candidates entered (entered > 0 and eliminated > 0).
        """
        stages = self.compute_attrition()
        candidates = [s for s in stages if s["entered"] > 0 and s["eliminated"] > 0]
        if not candidates:
            return None
        # Sort by attrition_pct descending, then by eliminated count descending
        candidates.sort(key=lambda x: (x["attrition_pct"], x["eliminated"]), reverse=True)
        return candidates[0]


def classify_zero_alert_run(
    scanner_name: str,
    universe_size: int,
    valid_data_count: int,
    initial_setups_count: int,
    finalist_candidates_count: int,
    alerts_generated: int,
    near_miss_count: int = 0,
    regime: str = "NEUTRAL",
    execution_mode: str = "LIVE",
    stage_waterfall: Optional[List[Dict[str, Any]]] = None,
    persistence_failures_count: int = 0,
    candidates_persisted_count: Optional[int] = None,
    lifecycle_summary: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Classifies a zero-alert run into an institutional anomaly category based on
    data health, persistence integrity, stage-by-stage candidate penetration, and execution mode:
    
    EVALUATION HIERARCHY (Strictly prioritized to prevent mode context from masking failures):
    1. DATA/ENGINE FAILURE? (Provider down, coverage < 75%, or 0 valid data)
    2. PERSISTENCE / LIFECYCLE FAILURE?
       - Finalists existed at alert gate but 0 alerts persisted
       - Direct database write errors / persistence exceptions > 0
       - Setups armed in PREARM but 0 persisted to watchlist database
       - Overnight candidates survived & monitor-eligible but 0 loaded into live monitor
    3. DEEP FUNNEL COLLAPSE? (Candidates penetrated into downstream stages or near-misses existed)
    4. NO VIABLE STRUCTURES? (0 structural setups formed under prevailing regime)
    5. PREARM / OUTSIDE_WINDOW? (Execution mode context explaining zero alerts when setups armed and persisted)
    """
    if alerts_generated > 0:
        return {
            "classification": "NORMAL_ALERT_GENERATION",
            "severity": "INFO",
            "explanation": f"{scanner_name} emitted {alerts_generated} valid alerts.",
            "recommendation": "None",
            "last_stage_with_candidates": "ALERT_PERSISTENCE"
        }

    # 1. DATA OR ENGINE FAILURE: Critical data deficit or provider failure
    if universe_size > 0:
        data_ratio = valid_data_count / universe_size
        if valid_data_count == 0 or data_ratio < 0.75:
            return {
                "classification": "DATA_OR_ENGINE_FAILURE",
                "severity": "CRITICAL",
                "explanation": f"Data coverage insufficient ({valid_data_count}/{universe_size} = {data_ratio*100:.1f}%). Provider down or blocked.",
                "recommendation": "Inspect data provider health, API tokens, and market session connectivity.",
                "last_stage_with_candidates": "DATA_ACQUISITION"
            }
    elif execution_mode != "MONITOR":
        return {
            "classification": "DATA_OR_ENGINE_FAILURE",
            "severity": "CRITICAL",
            "explanation": f"Universe size is 0 for {scanner_name}. Database watchlist query returned empty set.",
            "recommendation": "Inspect watchlist table and database connection.",
            "last_stage_with_candidates": "UNIVERSE_SELECTION"
        }

    # 2. PERSISTENCE OR LIFECYCLE FAILURE: Candidates reached final gate or state lifecycle failed
    # Note: Even in PREARM/MONITOR mode, persistence failures are a critical defect, not a legitimate zero.

    # Check 2a: Candidates reached final risk/alert gate or direct persistence exceptions occurred
    if finalist_candidates_count > 0 or persistence_failures_count > 0:
        fail_count = finalist_candidates_count or persistence_failures_count
        return {
            "classification": "CRITICAL_ZERO",
            "severity": "CRITICAL",
            "explanation": f"{fail_count} candidates reached final risk/persistence gate or failed during state persistence, but 0 alerts/records were persisted.",
            "recommendation": "Verify SL/Target engine thresholds, live price recheck buy-zone shift, and database write connectivity.",
            "last_stage_with_candidates": "FINAL_RISK_AND_PERSISTENCE"
        }

    # Check 2b: PREARM candidate arming vs DB persistence discrepancy
    # E.g. 100 setups armed, but 0 persisted to mtf_v2_watchlist database
    if execution_mode in ("PREARM", "NON_MARKET", "OUTSIDE_WINDOW") and initial_setups_count > 0:
        persisted_zero = (candidates_persisted_count == 0) or (
            lifecycle_summary is not None and lifecycle_summary.get("total_in_watchlist", 0) == 0
        )
        if persisted_zero:
            return {
                "classification": "CRITICAL_ZERO",
                "severity": "CRITICAL",
                "explanation": f"{initial_setups_count} candidates were identified for arming in {execution_mode}, but 0 records persisted to watchlist database.",
                "recommendation": "Check database table mtf_v2_watchlist write permissions, schema constraints, and persist_new_watchlist_candidate calls.",
                "last_stage_with_candidates": "ARMED_STATE_PERSISTENCE"
            }

    # Check 2c: Live MONITOR lifecycle load discrepancy
    # E.g. 30 candidates survived overnight and are eligible, but 0 were loaded into live monitor
    if execution_mode == "MONITOR" and universe_size == 0:
        if lifecycle_summary and lifecycle_summary.get("live_monitor_eligible", 0) > 0:
            eligible = lifecycle_summary["live_monitor_eligible"]
            return {
                "classification": "CRITICAL_ZERO",
                "severity": "CRITICAL",
                "explanation": f"{eligible} candidates survived overnight and are live-monitor eligible, but 0 were loaded into {scanner_name}.",
                "recommendation": "Inspect get_active_armed_candidates() query, active substate filters, and cooldown timestamps.",
                "last_stage_with_candidates": "MONITOR_LOAD_QUERY"
            }

    # Inspect stage waterfall progression to identify deepest stage reached
    deepest_stage = "UNIVERSE"
    downstream_candidates_reached = False

    if stage_waterfall and len(stage_waterfall) > 1:
        for idx, stg in enumerate(stage_waterfall):
            if stg.get("entered", 0) > 0:
                deepest_stage = stg.get("stage", "UNKNOWN")
                # If candidates penetrated past the first 2 stages into scoring/conviction/risk
                if idx >= 2:
                    downstream_candidates_reached = True
            if stg.get("passed", 0) > 0 and idx == len(stage_waterfall) - 1:
                deepest_stage = stg.get("next_stage", deepest_stage)

    # 3. DEEP FUNNEL COLLAPSE: Setups reached downstream gates or notable near-misses existed
    if downstream_candidates_reached or near_miss_count > 0:
        return {
            "classification": "SUSPICIOUS_ZERO",
            "severity": "WARNING",
            "explanation": f"Discovered {initial_setups_count} technical structures with candidates penetrating into stage '{deepest_stage}', but downstream filters eliminated 100% ({near_miss_count} near misses).",
            "recommendation": "Inspect dominant bottleneck attrition rate to assess if volume, conviction, or score gates are overly restrictive.",
            "last_stage_with_candidates": deepest_stage
        }

    # 4. NO VIABLE STRUCTURES: Clean structural legitimate zero under prevailing regime
    if initial_setups_count == 0 and near_miss_count == 0:
        return {
            "classification": "LEGITIMATE_ZERO",
            "severity": "INFO",
            "explanation": f"Market regime is {regime}. 0 structural setups formed from {universe_size} stocks. Clean legitimate zero under current market conditions.",
            "recommendation": "No action required. Market conditions do not exhibit required setup characteristics.",
            "last_stage_with_candidates": deepest_stage
        }

    # 5. PREARM / OUTSIDE WINDOW: Normal scheduled setup screening (with surviving armed pool)
    if execution_mode in ("PREARM", "NON_MARKET", "OUTSIDE_WINDOW"):
        persisted_info = f" ({candidates_persisted_count} verified in DB)" if candidates_persisted_count is not None else ""
        return {
            "classification": "LEGITIMATE_ZERO",
            "severity": "INFO",
            "explanation": f"Execution mode is {execution_mode}. Successfully screened and preserved {initial_setups_count} armed candidate setups{persisted_info} for next session open; new live trade entries intentionally suppressed.",
            "recommendation": "Monitor armed candidate pool for execution eligibility at 09:15 open.",
            "last_stage_with_candidates": deepest_stage
        }

    return {
        "classification": "LEGITIMATE_ZERO",
        "severity": "INFO",
        "explanation": f"Market regime is {regime}. Found {initial_setups_count} preliminary candidates, but all failed standard early technical filtering. No late-stage survivors.",
        "recommendation": "No action required. Preliminary candidates lacked necessary confirmation.",
        "last_stage_with_candidates": deepest_stage
    }


def format_zero_alert_diagnostic_block(
    scanner_name: str,
    execution_mode: str,
    regime: str,
    classification_result: Dict[str, Any],
    dominant_bottleneck: Optional[Dict[str, Any]],
    conservation_summary: Dict[str, Any],
    stage_waterfall: Optional[List[Dict[str, Any]]] = None,
    near_miss_count: int = 0,
    extra_specs: Optional[List[str]] = None,
    bottleneck_terminal_breakdown: Optional[Dict[str, int]] = None
) -> List[str]:
    """Generates a standardized ASCII diagnostic block for scanner logs."""
    cls_name = classification_result.get("classification", "UNKNOWN")
    severity = classification_result.get("severity", "INFO")
    icon = "🚨" if severity == "CRITICAL" else ("⚠️" if severity == "WARNING" else "ℹ️")

    lines = [
        "",
        f"{icon} ZERO_ALERT_DIAGNOSTIC ({scanner_name}):",
        f"  • Classification            : {cls_name} [{severity}]",
        f"  • Execution Mode            : {execution_mode}",
        f"  • Market Regime             : {regime}",
        f"  • Finding                   : {classification_result.get('explanation', '')}",
    ]

    deepest = classification_result.get("last_stage_with_candidates")
    if deepest and deepest not in ("NONE", "UNIVERSE"):
        lines.append(f"  • Deepest Stage Reached     : {deepest}")

    if extra_specs:
        for spec in extra_specs:
            lines.append(f"  • {spec}")

    if dominant_bottleneck:
        b_stage = dominant_bottleneck.get("stage", "UNKNOWN")
        b_entered = dominant_bottleneck.get("entered", 0)
        b_failed = dominant_bottleneck.get("eliminated", 0)
        b_rate = dominant_bottleneck.get("attrition_pct", 0.0)
        lines.append(f"  • Dominant Bottleneck Gate  : {b_stage} (Attrition: {b_failed}/{b_entered} = {b_rate:.1f}%)")
        if bottleneck_terminal_breakdown:
            lines.append(f"      ├── Terminal Breakdown within {b_stage} ({b_failed} eliminated):")
            b_items = sorted(bottleneck_terminal_breakdown.items(), key=lambda x: x[1], reverse=True)
            for subgate, count in b_items:
                pct = (count / max(b_failed, 1)) * 100.0
                lines.append(f"      │     • {subgate:<26}: {count:>3} ({pct:>5.1f}%)")

    if stage_waterfall:
        lines.append("  • Stage Waterfall Funnel    :")
        for stg in stage_waterfall:
            lines.append(f"      ├── {stg['stage']:<24}: {stg['entered']:>4} entered → {stg['passed']:>4} passed (loss: {stg['eliminated']} [{stg['attrition_pct']:.1f}%])")

    untracked = conservation_summary.get("untracked_drop", 0)
    delta = conservation_summary.get("conservation_delta", 0)
    total_u = conservation_summary.get("total_universe", 0)
    sum_t = conservation_summary.get("sum_terminal", 0)
    if delta == 0 and untracked == 0:
        cons_str = f"{sum_t}/{total_u} terminal outcomes (Delta: 0, UNTRACKED: 0) [PASS]"
    else:
        cons_str = f"{sum_t}/{total_u} terminal outcomes (Delta: {delta}, UNTRACKED: {untracked}) [ANOMALY DETECTED]"
    lines.append(f"  • Conservation Check        : {cons_str}")
    lines.append(f"  • Recommendation            : {classification_result.get('recommendation', 'None')}")

    return lines
