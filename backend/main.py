"""FastAPI service for the SLR-GAT delay forecaster.

    uvicorn main:app --reload --port 8000     (from backend/)

Endpoints
    GET  /api/info                      deployed model metadata
    GET  /api/dates                     dates present in the bundle
    GET  /api/services?date=YYYY-MM-DD  services running that day
    POST /api/forecast                  {train_id, date, seeds[]}  -> Manual
    POST /api/replay                    {train_id, date}           -> Replay
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine import Engine

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = Path(os.environ.get("SLRGAT_BUNDLE", ROOT / "bundle"))
FRONTEND = ROOT / "frontend"

app = FastAPI(title="SLR-GAT Delay Forecaster", version="1.0")

try:
    engine = Engine(BUNDLE)
    LOAD_ERROR = None
except Exception as exc:                      # surfaced at /api/info
    engine, LOAD_ERROR = None, str(exc)


def _engine():
    if engine is None:
        raise HTTPException(503, f"model bundle not loaded: {LOAD_ERROR}")
    return engine


class ForecastIn(BaseModel):
    train_id: str
    date: str
    seeds: list[float] = Field(..., min_length=1, max_length=8)
    others_on_time: bool = True


class ReplayIn(BaseModel):
    train_id: str
    date: str


@app.get("/api/info")
def info():
    if engine is None:
        return {"loaded": False, "error": LOAD_ERROR, "bundle_path": str(BUNDLE)}
    return {"loaded": True, **engine.info()}


@app.get("/api/dates")
def dates():
    return _engine().dates()


@app.get("/api/services")
def services(date: str = Query(..., description="YYYY-MM-DD")):
    out = _engine().services(date)
    if not out:
        raise HTTPException(404, f"no services found for {date}")
    return out


@app.post("/api/forecast")
def forecast(body: ForecastIn):
    e = _engine()
    if len(body.seeds) < e.SEED_LEN:
        raise HTTPException(
            422, f"this model needs {e.SEED_LEN} observed delays, "
                 f"got {len(body.seeds)}")
    try:
        return e.forecast(body.train_id, body.date, seeds=body.seeds,
                          others_on_time=body.others_on_time)
    except ValueError as exc:
        raise HTTPException(404, str(exc))


@app.post("/api/replay")
def replay(body: ReplayIn):
    try:
        return _engine().replay(body.train_id, body.date)
    except ValueError as exc:
        raise HTTPException(404, str(exc))


if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

    @app.get("/")
    def index():
        return FileResponse(FRONTEND / "index.html")
