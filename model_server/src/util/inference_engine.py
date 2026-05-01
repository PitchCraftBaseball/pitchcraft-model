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

from .feature_store import FeatureStore
from .pitch_state_builder import build_pitch_state_from_features, MissingFeaturesError
from .pitchcraft_inference_helper import build_pitch_probabilities, build_tensors
from .players_accessor import fetch_handedness

COMPOSITE_CONFIG = {
    "rnn_weight": 1, 
    "out_type_weight": 1,
    "run_expectancy_weight": 1
}

Strategy = Literal["argmax", "sample"]

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


@dataclass
class SimulationResult:
    outcome: str
    pitch_count: int
    sequence: List[PitchStep] = field(default_factory=list)


class InferenceEngine:
    """Owns the loaded RNN + transition + out-type models and runs the
    plate-appearance simulation loop. Construct once in ``create_app``.
    """

    def __init__(self, artifacts, rnn, feature_store: FeatureStore) -> None:
        
        self.artifacts = artifacts
        self.rnn = rnn
        self.feature_store = feature_store

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
        # None breaks xgboost (object dtype); 0 is effectively the same.
        zone_raw = state_features.get("zone")
        zone = int(zone_raw) if zone_raw is not None else 0

        sequence: List[PitchStep] = []
        outcome: Optional[str] = None

        for pitch_idx in range(1, max_pitches + 1):
            rnn_pitch_probs = self._predict_next_pitch(
                pitcher=pitcher,
                batter=batter,
                state_features=state_features,
                balls=balls,
                strikes=strikes,
                prev_pitch_type=prev_pitch_type,
                batter_side=batter_side,
                pitcher_arm=pitcher_arm,
            )
            
            # chosen_pitch = self._resolve_event(rnn_pitch_probs, strategy, rng)

            """
            for each output in rnn_pitch_probs, 1) predict transition_probs 2) predict_pitch_out_type_outcome 3) apply composite equation 
            then at the end we'll just pick the pitch that has the highest composite score
            """
            
            game_context = self._build_game_context(
                state_features=state_features,
                balls=balls,
                strikes=strikes,
                batter_side=batter_side,
                pitcher_arm=pitcher_arm,
                prev_pitch_type=prev_pitch_type,
            )

            transition_probs = predict_pitch_transition_outcome(
                batter_id=batter,
                pitcher_id=pitcher,
                pitch_type=chosen_pitch,
                year=year,
                game_context=game_context,
                zone=zone,
            )
            p_strike = float(transition_probs["p_strike"][0])
            p_ball = float(transition_probs["p_ball"][0])

            out_type_raw = predict_pitch_out_type_outcome(
                batter_id=batter,
                pitcher_id=pitcher,
                pitch_type=chosen_pitch,
                year=year,
                game_context=game_context,
                zone=zone,
            )
            
            # TODO: add RE here 
            
            
            # logger.debug("out_type_raw:")
            # logger.debug(out_type_raw)

            out_type_probs = {
                k: float(np.asarray(v).item()) for k, v in out_type_raw.items()
            }

            logger.debug(out_type_probs)


            transition_event = self._resolve_event(
                {"strike": p_strike, "ball": p_ball}, strategy, rng
            )
            
            # override this with strategy 
            out_type_event = self._resolve_event(
                out_type_probs,
                strategy,
                rng,
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
                )
            )

            if terminal:
                outcome = terminal_label
                break

            prev_pitch_type = chosen_pitch

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
        batter_side: str,
        pitcher_arm: str,
    ) -> Dict[str, float]:
        situation = count_situation(balls, strikes)
        pitcher_splits = self.feature_store.get_pitcher_situation_splits(pitcher, situation)
        batter_splits = self.feature_store.get_batter_situation_splits(batter, situation)

        enriched_state = {
            **state_features,
            "balls": balls,
            "strikes": strikes,
            "stand": batter_side,
            "p_throws": pitcher_arm,
            "prev_pitch_type": prev_pitch_type,
        }

        required_cols = list(self.artifacts.cat_cols) + list(self.artifacts.num_cols)
        state = build_pitch_state_from_features(
            pitcher, batter, enriched_state, batter_splits, pitcher_splits,
            required_cols=required_cols,
        )

        x_cat, x_num, seq_len = build_tensors([state], self.artifacts)
        with torch.no_grad():
            logits = self.rnn(x_cat, x_num)
            probs = torch.softmax(logits, dim=-1)[0]

        pitch_dist = build_pitch_probabilities(
            probs, self.artifacts, seq_len, pitch_keys=["pitch_one"]
        )
        return pitch_dist["pitch_one"]
    
    def parse_re288_pitch_results(outs, first_base, second_base, third_base, balls, strikes, pitch_classification):
        base_key = f"{'O' if first_base else 'X'}{'O' if second_base else 'X'}{'O' if third_base else 'X'}"
        count_key = f"{balls}-{strikes}"   
        
        json_path = Path(__file__).resolve().parent / "re288.json"
        
        with (open(json_path), "r") as file: 
            data = json.load(file)
            
        data.get()
    
    def _calculate_power_mean(rnn_score, out_type_score, run_expectancy_score,
                              rnn_weight, out_type_weight, run_expectancy_weight, p): 
        adj_re_score = 1 / (1 + run_expectancy_score)
        if p == 0:
            # Geometric mean
            product = (rnn_score ** rnn_weight) * (out_type_score ** out_type_weight) * (adj_re_score ** run_expectancy_weight)
            return product
        # if p == 1, it is the weighted arithmetic mean
        # if p approaches infinity, it approaches the maximum score among the three
        # if p approaches negative infinity, it approaches the minimum score among the three
        weighted_sum = (rnn_weight * (rnn_score ** p)) + (out_type_weight * (out_type_score ** p)) + (run_expectancy_weight * (adj_re_score ** p))
        return weighted_sum ** (1/p)
    
    @staticmethod
    def _resolve_event(
        probs: Dict[str, float],
        strategy: Strategy,
        rng: Optional[np.random.Generator],
        optimal_out: str = None 
    ) -> str:
        
        if strategy == "optimal_out": 
            return -1
        
        if strategy == "argmax":
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
        # Translate the API's `bat_score_diff` and `on_*` fields into the
        # column names the support models were trained on.
        bat_score_diff = int(state_features.get("bat_score_diff", 0))
        bat_score = bat_score_diff if bat_score_diff > 0 else 0
        fld_score = -bat_score_diff if bat_score_diff < 0 else 0

        return {
            "balls": balls,
            "strikes": strikes,
            "stand": batter_side,
            "p_throws": pitcher_arm,
            "inning": int(state_features["inning"]),
            "inning_topbot": state_features["inning_topbot"],
            "bat_score": bat_score,
            "fld_score": fld_score,
            "runner_on_1b": int(bool(state_features.get("on_1b", 0))),
            "runner_on_2b": int(bool(state_features.get("on_2b", 0))),
            "runner_on_3b": int(bool(state_features.get("on_3b", 0))),
            "outs_when_up": int(state_features["outs_when_up"]),
            "prev_pitch_type": prev_pitch_type,
            "prev_zone": int(state_features.get("prev_zone", 0)),
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
