from __future__ import annotations

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from fastapi import FastAPI, HTTPException

from model_shared.parameter_loaders import (
    latest_parameters_for_specific_model,
    latest_vocab_json_for_specific_model,
    load_vocabs_from_json,
)
from model_shared.rnn_definition import PitchRNN

from .dto import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)
from .util.request_handler import RequestHandler
from .util.feature_store import FeatureStore, ParquetHistoricalPitchesFeatureStore
from .util.inference_engine import (
    InferenceEngine,
    PlayerNotFoundError,
    SimulationResult,
    Strategy,
)
from .util.pitch_state_builder import MissingFeaturesError


logger = logging.getLogger(__name__)


class Artifacts:
    def __init__(self, config_path: Path, vocab_path: Path) -> None:
        data = json.loads(config_path.read_text())
        self.max_len = int(data.get("max_len", 8))
        self.pad_id = int(data.get("pad_id", 0))
        self.emb_dims: Dict[str, int] = dict(data["emb_dims"])
        self.hidden = int(data.get("hidden", 128))
        self.num_layers = int(data.get("num_layers", 1))

        self.cat_vocabs, self.y_vocab, feature_spec = load_vocabs_from_json(vocab_path)
        self.cat_cols = list(feature_spec["cat_cols"])
        self.num_cols = list(feature_spec["num_cols"])
        self.bool_cols = list(feature_spec.get("bool_cols", []))

        self.id_to_pitch = {int(v): k for k, v in self.y_vocab.items()}

    def cat_vocab_sizes(self) -> Dict[str, int]:
        sizes = {}
        for col, vocab in self.cat_vocabs.items():
            max_id = max([0] + [int(v) for v in vocab.values()])
            sizes[col] = max_id + 1
        return sizes

    def num_classes(self) -> int:
        return max([0] + [int(v) for v in self.y_vocab.values()]) + 1


def _load_model_bundle(config_path: Path, model_type: str) -> Tuple[Artifacts, PitchRNN]:
    vocab_path = latest_vocab_json_for_specific_model(model_type)
    param_path = latest_parameters_for_specific_model(model_type)
    arts = Artifacts(config_path, vocab_path)
    model = PitchRNN(
        cat_vocab_sizes=arts.cat_vocab_sizes(),
        num_features=len(arts.num_cols),
        emb_dims=arts.emb_dims,
        hidden=arts.hidden,
        num_classes=arts.num_classes(),
        num_layers=arts.num_layers,
        pad_id=arts.pad_id,
    )
    model.load_state_dict(torch.load(param_path, map_location="cpu"))
    model.eval()
    return arts, model


def create_app(feature_store: Optional[FeatureStore] = None) -> FastAPI:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config_path = Path(__file__).resolve().parent / "model_config.json"
    if not config_path.exists():
        raise RuntimeError(
            "Missing model_config.json. Provide feature spec + vocabs before starting the API."
        )

    group_artifacts, group_rnn = _load_model_bundle(config_path, "group")
    sub_bundles: Dict[str, Tuple[Artifacts, PitchRNN]] = {
        "fastball": _load_model_bundle(config_path, "fastball"),
        "offspeed": _load_model_bundle(config_path, "offspeed"),
        "breaking": _load_model_bundle(config_path, "breaking"),
    }

    store: FeatureStore = feature_store or ParquetHistoricalPitchesFeatureStore()
    engine = InferenceEngine(
        group_artifacts=group_artifacts,
        group_rnn=group_rnn,
        sub_bundles=sub_bundles,
        feature_store=store,
    )
    handler = RequestHandler(engine)

    inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pitch-inference")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            inference_executor.shutdown(wait=True)

    app = FastAPI(title="Pitch RNN Inference API", lifespan=lifespan)

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest) -> PredictResponse:
        start = time.perf_counter()
        logger.info(
            "predict request: pitcher=%s batter=%s year=%s strategy=%s max_pitches=%s",
            req.pitcher, req.batter, req.year, req.strategy, req.max_pitches,
        )

        missing = handler.validate_state(req.state_features)
        if missing:
            raise HTTPException(
                status_code=422,
                detail={
                    "missing_features": missing,
                    "message": "state_features is missing required keys.",
                },
            )

        strategy, pref = handler.resolve_strategy(req)

        loop = asyncio.get_running_loop()
        try:
            result: SimulationResult = await loop.run_in_executor(
                inference_executor, partial(handler.run_simulation, req, strategy, pref)
            )
        except PlayerNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except MissingFeaturesError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "missing_features": exc.missing,
                    "message": "Request is missing features required by the loaded model.",
                },
            )

        response = handler.result_to_response(result)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "predict response: pitcher=%s batter=%s outcome=%s pitch_count=%d elapsed_ms=%.2f",
            req.pitcher, req.batter, result.outcome, result.pitch_count, elapsed_ms,
        )
        return response

    @app.post("/predict/batch", response_model=BatchPredictResponse)
    async def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
        start = time.perf_counter()
        logger.info("predict_batch request: size=%d", len(req.requests))

        prepared: List[Tuple[PredictRequest, Strategy, Optional[str]]] = []
        for i, item in enumerate(req.requests):
            missing = handler.validate_state(item.state_features)
            if missing:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "failed_index": i,
                        "missing_features": missing,
                        "message": "state_features is missing required keys.",
                    },
                )
            strategy, pref = handler.resolve_strategy(item, failed_index=i)
            prepared.append((item, strategy, pref))

        loop = asyncio.get_running_loop()
        try:
            results: List[SimulationResult] = await loop.run_in_executor(
                inference_executor, partial(handler.run_batch, prepared)
            )
        except PlayerNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail={
                    "failed_index": getattr(exc, "failed_index", None),
                    "message": str(exc),
                },
            )
        except MissingFeaturesError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "failed_index": getattr(exc, "failed_index", None),
                    "missing_features": exc.missing,
                    "message": "Request is missing features required by the loaded model.",
                },
            )

        response = handler.results_to_batch_response(results)
        elapsed_ms = (time.perf_counter() - start) * 1000
        outcomes: Dict[str, int] = {}
        for r in results:
            outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1
        logger.info(
            "predict_batch response: size=%d outcomes=%s elapsed_ms=%.2f",
            len(results), outcomes, elapsed_ms,
        )
        return response

    return app


app = create_app()
