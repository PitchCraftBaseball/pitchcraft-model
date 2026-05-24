from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import logging
import json 
from pathlib import Path

from model_shared.feature_engineering.feature_calculator import count_situation
from model_shared.transition_inference import predict_pitch_transition_outcome
from rnn_support_models.out_type_model.out_type_inference_helper import (
    predict_pitch_out_type_outcome,
)
from model_shared.feature_engineering.pitch_constants import BASE_LABELS
from model_shared.optimal_out_handler import get_optimal_out
from model_shared.location_helper import (
    precompute_location_context,
    get_optimal_location_from_context,
    BUCKET_TO_ZONE,
    FALLBACK_LOCATION,
)

from .feature_store import FeatureStore
from .pitch_state_builder import build_pitch_state_from_features, MissingFeaturesError
from .pitchcraft_inference_helper import build_pitch_probabilities, build_tensors
from .players_accessor import fetch_handedness

COMPOSITE_CONFIG = {
    "rnn_weight": 2,
    "out_type_weight": 2,
    "run_expectancy_weight": 1
}

Strategy = Literal["argmax", "sample", "optimal_out", "preferred"]

VALID_OUT_TYPES = ("strikeout", "groundout", "flyout")

PA_OUTCOMES = ("walk", "strikeout", "groundout", "flyout", "hard_hit_flyball", "in_progress")

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


class PlayerNotFoundError(Exception):
    def __init__(self, player_id: str, role: str) -> None:
        super().__init__(f"{role} {player_id} not found")
        self.player_id = player_id
        self.role = role


@dataclass
class PitchStep:
    pitch_index: int
    pitch_type: str
    rnn_pitch_probs: Dict[str, float]
    p_strike: float
    p_ball: float
    out_type_probs: Dict[str, float]
    transition_event: str
    out_type_event: str
    balls_after: int
    strikes_after: int
    terminal: bool
    outcome: Optional[str] = None
    target_location: Optional[str] = None


@dataclass
class SimulationResult:
    outcome: str
    pitch_count: int
    sequence: List[PitchStep] = field(default_factory=list)


class InferenceEngine:
    """Owns the loaded RNN + transition + out-type models and runs the
    plate-appearance simulation loop. Construct once in ``create_app``.
    """

    def __init__(self, group_artifacts, group_rnn, sub_bundles: Dict[str, Tuple], feature_store: FeatureStore) -> None:

        self.group_artifacts = group_artifacts
        self.group_rnn = group_rnn
        self.sub_bundles = sub_bundles  # {"fastball": (arts, rnn), "offspeed": ..., "breaking": ...}
        self.feature_store = feature_store

        re288_path = Path(__file__).resolve().parent / "re288.json"
        with open(re288_path, "r") as f:
            self._re288 = json.load(f)

        arsenal_path = Path(__file__).resolve().parent.parent.parent.parent / "pitch_arsenal" / "arsenals_all.json"
        with open(arsenal_path, "r") as f:
            self._arsenals = json.load(f)

        # Map each group name to the set of pitch types its sub-model covers.
        self._group_to_pitches: Dict[str, set] = {
            group_name: set(arts.y_vocab.keys())
            for group_name, (arts, _) in sub_bundles.items()
        }

    def simulate_plate_appearance(
        self,
        *,
        pitcher: str,
        batter: str,
        year: int,
        state_features: Dict[str, Any],
        strategy: Strategy = "argmax",
        max_pitches: int = 12,
        rng: Optional[np.random.Generator] = None,
        preferred_out_type: Optional[str] = None,
    ) -> SimulationResult:
        if strategy == "sample" and rng is None:
            rng = np.random.default_rng()

        batter_side = fetch_handedness(batter, is_batter=True)
        if batter_side is None:
            raise PlayerNotFoundError(batter, "batter")
        pitcher_arm = fetch_handedness(pitcher, is_batter=False)
        if pitcher_arm is None:
            raise PlayerNotFoundError(pitcher, "pitcher")

        balls = int(state_features["balls"])
        strikes = int(state_features["strikes"])
        prev_pitch_type = state_features.get("prev_pitch_type", "START")
        prev_pitch_group = state_features.get("prev_pitch_group", "START")

        sequence: List[PitchStep] = []
        outcome: Optional[str] = None
        pa_states: List = []

        if strategy == "preferred":
            if preferred_out_type not in VALID_OUT_TYPES:
                raise ValueError(
                    f"preferred_out_type must be one of {VALID_OUT_TYPES}, got {preferred_out_type!r}"
                )
            optimal_out = {ot: (1.0 if ot == preferred_out_type else 0.0) for ot in VALID_OUT_TYPES}
            optimal_out_type = preferred_out_type
            logger.debug(
                "\n--- PREFERRED_OUT ---\n"
                "  pitcher=%s  batter=%s  preferred=%s\n",
                pitcher, batter, preferred_out_type,
            )
        else:
            optimal_out = get_optimal_out(pitcher, batter, state_features)
            optimal_out_type = max(optimal_out, key=optimal_out.get)  # "strikeout", "groundout", or "flyout"
            logger.debug(
                "\n--- OPTIMAL_OUT ---\n"
                "  pitcher=%s  batter=%s\n"
                "  scores=%s\n"
                "  target=%s\n",
                pitcher, batter, optimal_out, optimal_out_type,
            )
        loc_context = precompute_location_context(pitcher, batter)

        for pitch_idx in range(1, max_pitches + 1):
            rnn_pitch_probs, current_state = self._predict_next_pitch(
                pitcher=pitcher,
                batter=batter,
                state_features=state_features,
                balls=balls,
                strikes=strikes,
                prev_pitch_type=prev_pitch_type,
                prev_pitch_group=prev_pitch_group,
                batter_side=batter_side,
                pitcher_arm=pitcher_arm,
                prior_states=pa_states,
            )
            pa_states.append(current_state)

            logger.debug(
                "\n--- PITCH %d  [%dB-%dS] ---\n"
                "  rnn_probs=%s\n",
                pitch_idx, balls, strikes, rnn_pitch_probs,
            )

            game_context = self._build_game_context(
                state_features=state_features,
                balls=balls,
                strikes=strikes,
                batter_side=batter_side,
                pitcher_arm=pitcher_arm,
                prev_pitch_type=prev_pitch_type,
            )

            outs = int(state_features["outs_when_up"])
            on_1b = bool(state_features.get("on_1b", 0))
            on_2b = bool(state_features.get("on_2b", 0))
            on_3b = bool(state_features.get("on_3b", 0))

            best_score: Optional[float] = None
            chosen_pitch: Optional[str] = None
            target_location: str = FALLBACK_LOCATION
            transition_probs = None
            out_type_probs: Dict[str, float] = {}
            p_strike = 0.0
            p_ball = 0.0

            current_state = {**state_features, "balls": balls, "strikes": strikes}

            for candidate_pitch, rnn_score in rnn_pitch_probs.items():
                candidate_location = get_optimal_location_from_context(
                    loc_context, candidate_pitch, current_state,
                    optimal_out_type if strategy in ("optimal_out", "preferred") else None,
                )
                candidate_zone = BUCKET_TO_ZONE[candidate_location]

                cand_transition_probs = predict_pitch_transition_outcome(
                    batter_id=batter,
                    pitcher_id=pitcher,
                    pitch_type=candidate_pitch,
                    year=year,
                    game_context=game_context,
                    location=candidate_zone,
                )
                cand_p_strike = float(cand_transition_probs["p_strike"][0])
                cand_p_ball = float(cand_transition_probs["p_ball"][0])

                cand_out_type_raw = predict_pitch_out_type_outcome(
                    batter_id=batter,
                    pitcher_id=pitcher,
                    pitch_type=candidate_pitch,
                    year=year,
                    game_context=game_context,
                    location=candidate_zone,
                )
                cand_out_type_probs = {
                    k: float(np.asarray(v).item()) for k, v in cand_out_type_raw.items()
                }

                # Always use the weighted blend so secondary out types still contribute.
                # optimal_out scores are already normalized-ish but we re-normalize to be safe.
                _ot_total = sum(optimal_out.values()) or 1.0
                out_type_score = (
                    (optimal_out['strikeout'] / _ot_total) * cand_out_type_probs.get("p_so", 0.0)
                    + (optimal_out['groundout'] / _ot_total) * cand_out_type_probs.get("p_go", 0.0)
                    + (optimal_out['flyout']    / _ot_total) * cand_out_type_probs.get("p_fo", 0.0)
                )

                re_score = self._lookup_run_expectancy(
                    outs=outs,
                    on_1b=on_1b,
                    on_2b=on_2b,
                    on_3b=on_3b,
                    balls=balls,
                    strikes=strikes,
                    pitch_classification=candidate_pitch,
                )
                if re_score is None:
                    re_score = 0.5

                composite = self._calculate_power_mean(
                    rnn_score=rnn_score,
                    out_type_score=out_type_score,
                    run_expectancy_score=re_score,
                    rnn_weight=COMPOSITE_CONFIG["rnn_weight"],
                    out_type_weight=COMPOSITE_CONFIG["out_type_weight"],
                    run_expectancy_weight=COMPOSITE_CONFIG["run_expectancy_weight"],
                    p=1,
                )

                logger.debug(
                    "  CANDIDATE  %-4s  loc=%-20s  rnn=%.4f  out_type=%.4f  re=%.4f  composite=%.4f",
                    candidate_pitch, candidate_location, rnn_score, out_type_score,
                    re_score if re_score is not None else 0.5, composite,
                )

                if best_score is None or composite > best_score:
                    best_score = composite
                    chosen_pitch = candidate_pitch
                    target_location = candidate_location
                    out_type_probs = cand_out_type_probs
                    p_strike = cand_p_strike
                    p_ball = cand_p_ball

            logger.debug(
                "\n  >> CHOSEN  %-4s  loc=%-20s  composite=%.4f\n"
                "     out_type_probs=%s\n",
                chosen_pitch, target_location, best_score, out_type_probs,
            )

            transition_event = self._resolve_event(
                {"strike": p_strike, "ball": p_ball}, strategy, rng
            )

            _normalized_out_type = {k.removeprefix("p_"): v for k, v in out_type_probs.items()}
            out_type_event = self._resolve_event(_normalized_out_type, strategy, rng)
            logger.debug(
                "  EVENTS  transition=%-6s  out_type_pre_override=%-4s  normalized_probs=%s",
                transition_event, out_type_event, _normalized_out_type,
            )
            logger.debug(
                "  EVENTS  transition=%-6s  out_type_final=%-4s  count=%dB-%dS\n",
                transition_event, out_type_event, balls, strikes,
            )

            balls, strikes, terminal_label = self._apply_outcome(
                balls=balls,
                strikes=strikes,
                transition_event=transition_event,
                out_type_event=out_type_event,
            )
            terminal = terminal_label is not None

            sequence.append(
                PitchStep(
                    pitch_index=pitch_idx,
                    pitch_type=chosen_pitch,
                    rnn_pitch_probs=rnn_pitch_probs,
                    p_strike=p_strike,
                    p_ball=p_ball,
                    out_type_probs=out_type_probs,
                    transition_event=transition_event,
                    out_type_event=out_type_event,
                    balls_after=balls,
                    strikes_after=strikes,
                    terminal=terminal,
                    outcome=terminal_label,
                    target_location=target_location,
                )
            )

            if terminal:
                outcome = terminal_label
                break

            prev_pitch_type = chosen_pitch
            prev_pitch_group = next(
                (g for g, pitches in self._group_to_pitches.items() if chosen_pitch in pitches),
                "START",
            )

        if outcome is None:
            outcome = "in_progress"

        return SimulationResult(outcome=outcome, pitch_count=len(sequence), sequence=sequence)

    def _predict_next_pitch(
        self,
        *,
        pitcher: str,
        batter: str,
        state_features: Dict[str, Any],
        balls: int,
        strikes: int,
        prev_pitch_type: str,
        prev_pitch_group: str,
        batter_side: str,
        pitcher_arm: str,
        prior_states: Optional[List] = None,
    ) -> Tuple[Dict[str, float], Any]:
        situation = count_situation(balls, strikes)
        pitcher_splits = self.feature_store.get_pitcher_situation_splits(pitcher, situation)
        batter_splits = self.feature_store.get_batter_situation_splits(batter, situation)
        pitcher_family_splits = self.feature_store.get_pitcher_family_splits(pitcher)

        enriched_state = {
            **state_features,
            "balls": balls,
            "strikes": strikes,
            "stand": batter_side,
            "p_throws": pitcher_arm,
            "prev_pitch_type": prev_pitch_type,
            "prev_pitch_group": prev_pitch_group,
            **pitcher_family_splits,
        }

        all_required = set(self.group_artifacts.cat_cols) | set(self.group_artifacts.num_cols)
        for sub_arts, _ in self.sub_bundles.values():
            all_required |= set(sub_arts.cat_cols) | set(sub_arts.num_cols)

        state = build_pitch_state_from_features(
            pitcher, batter, enriched_state, batter_splits, pitcher_splits,
            required_cols=list(all_required),
        )

        seq_states = (prior_states or []) + [state]
        seq_states = seq_states[-4:]

        x_cat, x_num, seq_len = build_tensors(seq_states, self.group_artifacts)

        with torch.no_grad():
            group_logits = self.group_rnn(x_cat, x_num)
            group_probs = torch.softmax(group_logits, dim=-1)[0]
        last_group_probs = group_probs[seq_len - 1 : seq_len]
        group_dist = build_pitch_probabilities(
            last_group_probs, self.group_artifacts, 1, pitch_keys=["pitch_one"]
        )["pitch_one"]

        logger.debug(
            "\n  GROUP_PROBS=%s\n",
            group_dist,
        )

        arsenal = self._get_arsenal(pitcher)

        logger.debug("  ARSENAL      pitcher=%s  arsenal=%s", pitcher, arsenal)

        # Determine which groups are active (have >= 1 arsenal pitch)
        active_groups: Dict[str, set] = {}
        for group_name in group_dist:
            group_pitches = self._group_to_pitches.get(group_name, set())
            pitches_in_group = group_pitches & arsenal if arsenal is not None else group_pitches
            if pitches_in_group:
                active_groups[group_name] = pitches_in_group

        # Re-normalize group probs over active groups
        total_group_prob = sum(group_dist[g] for g in active_groups)
        norm_group_dist = {
            g: group_dist[g] / total_group_prob for g in active_groups
        } if total_group_prob > 0 else {g: group_dist[g] for g in active_groups}

        final_dist: Dict[str, float] = {}
        for group_name, group_prob in norm_group_dist.items():  # <-- was norm_group_dist
            arsenal_in_group = active_groups[group_name]

            if len(arsenal_in_group) == 1:
                # Single pitch in this group — skip sub-model entirely.
                final_dist[next(iter(arsenal_in_group))] = group_prob
            else:
                sub_artifacts, sub_rnn = self.sub_bundles[group_name]
                sub_x_cat, sub_x_num, sub_seq_len = build_tensors(seq_states, sub_artifacts)
                with torch.no_grad():
                    sub_logits = sub_rnn(sub_x_cat, sub_x_num)
                    sub_probs = torch.softmax(sub_logits, dim=-1)[0]
                last_sub_probs = sub_probs[sub_seq_len - 1 : sub_seq_len]
                sub_dist = build_pitch_probabilities(
                    last_sub_probs, sub_artifacts, 1, pitch_keys=["pitch_one"]
                )["pitch_one"]

                logger.debug(
                    "  SUB_DIST  group=%-10s  dist=%s",
                    group_name, sub_dist,
                )

                # Filter to arsenal pitches then re-normalize within the group.
                sub_dist = {pt: p for pt, p in sub_dist.items() if pt in arsenal_in_group}
                total_sub = sum(sub_dist.values())
                if total_sub > 0:
                    sub_dist = {pt: p / total_sub for pt, p in sub_dist.items()}

                for pitch_type, sub_prob in sub_dist.items():
                    final_dist[pitch_type] = group_prob * sub_prob

        return final_dist, state
    
    def _get_arsenal(self, pitcher: str) -> Optional[set]:
        pitcher_data = self._arsenals.get(pitcher)
        if pitcher_data is None:
            return None
        year_data = pitcher_data.get("2025")
        if year_data is None:
            return None
        return set(year_data.get("arsenal_mask", []))

    def _lookup_run_expectancy(
        self,
        *,
        outs: int,
        on_1b: bool,
        on_2b: bool,
        on_3b: bool,
        balls: int,
        strikes: int,
        pitch_classification: str,
    ) -> Optional[float]:
        base_key = f"{'O' if on_1b else 'X'}{'O' if on_2b else 'X'}{'O' if on_3b else 'X'}"
        count_key = f"{balls}-{strikes}"
        node = (
            self._re288.get(str(outs), {})
            .get(base_key, {})
            .get(count_key, {})
            .get(pitch_classification)
        )
        if node is None:
            logger.debug(
                "re288 miss: outs=%s base=%s count=%s pitch=%s",
                outs, base_key, count_key, pitch_classification,
            )
            return None
        return float(node["run_expectancy"])

    @staticmethod
    def _calculate_power_mean(rnn_score, out_type_score, run_expectancy_score,
                              rnn_weight, out_type_weight, run_expectancy_weight, p=1):
        adj_re_score = 1 / (1 + run_expectancy_score)
        if p == 0:
            # Geometric mean
            product = (rnn_score ** rnn_weight) * (out_type_score ** out_type_weight) * (adj_re_score ** run_expectancy_weight)
            return product
        # if p == 1, it is the weighted arithmetic mean
        # if p approaches infinity, it approaches the maximum score among the three
        # if p approaches negative infinity, it approaches the minimum score among the three
        weighted_sum = (rnn_weight * (rnn_score ** p)) + (out_type_weight * (out_type_score ** p)) 
        + (run_expectancy_weight * (adj_re_score ** p))
        return weighted_sum ** (1/p)
    
    @staticmethod
    def _resolve_event(
        probs: Dict[str, float],
        strategy: Strategy,
        rng: Optional[np.random.Generator],
    ) -> str:
        if strategy in ("argmax", "optimal_out", "preferred"):
            return max(probs, key=probs.get)
        if strategy == "sample":
            keys = list(probs)
            values = np.array([probs[k] for k in keys], dtype=float)
            total = values.sum()
            if total <= 0:
                return keys[int(np.argmax(values))]
            values = values / total
            return rng.choice(keys, p=values)
        raise ValueError(f"Unknown strategy: {strategy}")

    @staticmethod
    def _build_game_context(
        *,
        state_features: Dict[str, Any],
        balls: int,
        strikes: int,
        batter_side: str,
        pitcher_arm: str,
        prev_pitch_type: str,
    ) -> Dict[str, Any]:
        return {
            "balls": balls,
            "strikes": strikes,
            "stand": batter_side,
            "p_throws": pitcher_arm,
            "inning": int(state_features["inning"]),
            "inning_topbot": state_features["inning_topbot"],
            "bat_score_diff": int(state_features.get("bat_score_diff", 0)),
            "on_1b": int(bool(state_features.get("on_1b", 0))),
            "on_2b": int(bool(state_features.get("on_2b", 0))),
            "on_3b": int(bool(state_features.get("on_3b", 0))),
            "outs_when_up": int(state_features["outs_when_up"]),
            "prev_pitch_type": prev_pitch_type,
        }

    @staticmethod
    def _apply_outcome(
        *,
        balls: int,
        strikes: int,
        transition_event: str,
        out_type_event: str,
    ) -> Tuple[int, int, Optional[str]]:
        # Resolution table:
        #   ball  + (any out_type)  -> balls += 1; if balls == 4 -> walk
        #   strike + so             -> strikeout
        #   strike + go             -> groundout
        #   strike + fo             -> flyout
        #   strike + hhfb           -> hard_hit_flyball
        #   strike + none           -> strikes += 1; if strikes == 3 -> strikeout
        if transition_event == "ball":
            balls += 1
            if balls >= 4:
                return balls, strikes, "walk"
            return balls, strikes, None

        if transition_event != "strike":
            raise ValueError(f"Unknown transition_event: {transition_event}")

        if out_type_event == "so":
            return balls, strikes, "strikeout"
        if out_type_event == "go":
            return balls, strikes, "groundout"
        if out_type_event == "fo":
            return balls, strikes, "flyout"
        if out_type_event == "hhfb":
            return balls, strikes, "hard_hit_flyball"

        # out_type_event == "none": strike that doesn't end the PA on its own
        strikes += 1
        if strikes >= 3:
            return balls, strikes, "strikeout"
        return balls, strikes, None
    
    def _get_family_usage(self, pitcher: str) -> Optional[Dict[str, float]]:
        pitcher_data = self._arsenals.get(pitcher)
        if pitcher_data is None:
            return None
        year_data = pitcher_data.get("2025")
        if year_data is None:
            return None
        return year_data.get("families")

