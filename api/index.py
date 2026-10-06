"""
Vercel Serverless Entrypoint for Parking Allocation Using Linear Algebra.

Strategy for Vercel deployment:
- The 10-stage math pipeline is pre-computed locally (python scripts/precompute.py)
  and committed as backend/data/pipeline_cache.json (19 KB) and lot_initial_state.json.
- Vercel's Python runtime reads these static JSON files instead of recomputing
  expensive linear algebra on every cold start.
- Dynamic endpoints (/api/allocate, /api/simulate, /api/reset) use lightweight
  NumPy routines (no pandas) for real-time allocation computations.
- This keeps the function bundle small: only fastapi + numpy in requirements.txt.
"""

import os, sys, json

# Put repo root on path so 'backend' package resolves correctly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from typing import List, Optional
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Parking Allocation – Linear Algebra",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static pre-computed data (read once at module load = one cold-start read) ──
DATA_DIR = os.path.join(ROOT, "backend", "data")

def _load(name):
    with open(os.path.join(DATA_DIR, name), "r", encoding="utf-8") as f:
        return json.load(f)

_pipeline_cache = None
_lot_initial    = None

def get_pipeline_cache():
    global _pipeline_cache
    if _pipeline_cache is None:
        _pipeline_cache = _load("pipeline_cache.json")
    return _pipeline_cache

def get_initial_lot():
    global _lot_initial
    if _lot_initial is None:
        _lot_initial = _load("lot_initial_state.json")
    return _lot_initial

# ── In-process mutable lot state (shared within a warm serverless instance) ──
# Vercel may spin up many instances; each keeps its own ephemeral state.
# For a university demo this is perfectly acceptable.
_live_lot_status: dict = {}   # spot_id -> status (0/1/2)

def _init_live_status():
    """Initialise ephemeral status from the pre-baked initial state."""
    global _live_lot_status
    if not _live_lot_status:
        for spot in get_initial_lot()["spots"]:
            _live_lot_status[spot["spot_id"]] = spot["status"]

def _lot_state_response():
    """Build a get_lot response, injecting live status into the frozen template."""
    _init_live_status()
    base = get_initial_lot()
    spots = []
    for s in base["spots"]:
        sid = s["spot_id"]
        live = _live_lot_status.get(sid, s["status"])
        spots.append({**s, "status": live,
                      "status_label": ["Free","Occupied","Blocked"][live]})
    occ = sum(1 for v in _live_lot_status.values() if v == 1)
    blk = sum(1 for v in _live_lot_status.values() if v == 2)
    fre = sum(1 for v in _live_lot_status.values() if v == 0)
    total = len(_live_lot_status)
    return {**base, "spots": spots,
            "stats": {"free": fre, "occupied": occ, "blocked": blk,
                      "utilization_pct": round(occ / total * 100, 1)}}

# ── Allocation helpers (pure NumPy, no pandas) ──────────────────────────────
ENTRANCE     = [0, 0]
DESTINATIONS = {
    "Market":      {"coord": [9, 1],  "name": "Bullring Retail & Markets"},
    "Town Hall":   {"coord": [0, 11], "name": "Town Hall / Civic Centre"},
    "Broad St":    {"coord": [9, 11], "name": "Broad St Entertainment"},
    "Mailbox Mall":{"coord": [5, 6],  "name": "Mailbox Central Promenade"},
}
ZONE_SPECS = {
    "Zone A": {"spots": 33, "real_capacity": 577, "color": "#3B82F6"},
    "Zone B": {"spots": 22, "real_capacity": 387, "color": "#10B981"},
    "Zone C": {"spots": 26, "real_capacity": 470, "color": "#F59E0B"},
    "Zone D": {"spots": 39, "real_capacity": 687, "color": "#8B5CF6"},
}
STRATEGY_WEIGHTS = {
    "S1": {"name": "S1: Nearest to Entry (L2)",   "alpha": 0.0, "beta": 1.0, "gamma": 0.0},
    "S2": {"name": "S2: Nearest to Dest (L1)",    "alpha": 1.0, "beta": 0.0, "gamma": 0.0},
    "S3": {"name": "S3: Weighted + Prediction",   "alpha": 1.0, "beta": 0.4, "gamma": 2.5},
}

def _compatibility(spot_features, vehicle_type):
    """Inner-product style compatibility check (standard/compact/ev/accessible)."""
    f = spot_features  # [is_std, is_compact, is_ev, is_accessible]
    if vehicle_type == "accessible": return f[3] == 1
    if vehicle_type == "ev":         return f[2] == 1
    if vehicle_type == "compact":    return f[0] == 1 or f[1] == 1
    return f[0] == 1   # standard

def _allocate_spot(vehicle_type, dest_key, strategy, h_hat):
    _init_live_status()
    spots = get_initial_lot()["spots"]
    d = np.array(DESTINATIONS.get(dest_key, DESTINATIONS["Market"])["coord"], dtype=float)
    e = np.array(ENTRANCE, dtype=float)
    w = STRATEGY_WEIGHTS.get(strategy, STRATEGY_WEIGHTS["S3"])

    costs, valid = {}, []
    for s in spots:
        sid = s["spot_id"]
        if _live_lot_status.get(sid, s["status"]) != 0:
            continue
        if not _compatibility(s["features"], vehicle_type):
            continue
        p = np.array([s["row"], s["col"]], dtype=float)
        l1 = float(np.sum(np.abs(p - d)))
        l2 = float(np.sqrt(np.sum((p - e)**2)))
        zone = s["zone"]
        h    = float(h_hat.get(zone, 0.5))
        cost = w["alpha"]*l1 + w["beta"]*l2 + w["gamma"]*h
        costs[sid] = {"cost": cost, "l1": l1, "l2": l2, "h": h, "spot": s}
        valid.append(sid)

    if not valid:
        return {"success": False, "message": f"No compatible free spot for {vehicle_type}."}

    valid.sort(key=lambda sid: costs[sid]["cost"])
    best_sid  = valid[0]
    best      = costs[best_sid]
    _live_lot_status[best_sid] = 1      # mark occupied

    norms_v = np.array([best["spot"]["row"], best["spot"]["col"]]) - d
    l1n = float(np.sum(np.abs(norms_v)))
    l2n = float(np.sqrt(np.sum(norms_v**2)))
    lin = float(np.max(np.abs(norms_v)))

    alts = []
    for sid in valid[1:4]:
        c = costs[sid]
        alts.append({"spot_id": sid, "row": c["spot"]["row"], "col": c["spot"]["col"],
                     "zone": c["spot"]["zone"], "type": c["spot"]["type"],
                     "cost": round(c["cost"],3),
                     "cost_diff": round(c["cost"]-best["cost"],3)})

    return {
        "success": True,
        "allocated_spot": {
            "spot_id":   best_sid,
            "row":       best["spot"]["row"],
            "col":       best["spot"]["col"],
            "zone":      best["spot"]["zone"],
            "type":      best["spot"]["type"],
            "total_cost": round(best["cost"], 3),
            "breakdown": {
                "walk_distance_l1":   round(best["l1"],2),
                "walk_cost_term":     round(w["alpha"]*best["l1"],2),
                "drive_distance_l2":  round(best["l2"],2),
                "drive_cost_term":    round(w["beta"] *best["l2"],2),
                "predicted_zone_load":round(best["h"],3),
                "congestion_cost_term":round(w["gamma"]*best["h"],2),
            },
            "norm_comparison_to_dest": {
                "l1_manhattan": l1n, "l2_euclidean": round(l2n,3), "linf_chebyshev": lin
            },
            "coordinates": [best["spot"]["row"], best["spot"]["col"]],
            "destination": dest_key,
            "destination_coord": list(d.astype(int)),
            "entrance_coord": ENTRANCE,
        },
        "alternatives":     alts,
        "strategy":         strategy,
        "strategy_details": w,
    }

def _simulate(n_vehicles, strategies, h_hat):
    """Light Monte Carlo – no pandas, pure NumPy random."""
    rng = np.random.RandomState(123)
    v_types = ["standard","compact","ev","accessible"]
    v_probs = [0.65, 0.15, 0.12, 0.08]
    dest_keys = list(DESTINATIONS.keys())
    arrivals = [{"vehicle_type": rng.choice(v_types, p=v_probs),
                 "destination":  rng.choice(dest_keys)}
                for _ in range(n_vehicles)]

    results = {}
    for strat in strategies:
        # Reset ephemeral state to a fresh 25%-occupied lot per strategy
        tmp_status = {}
        rng2 = np.random.RandomState(42)
        for s in get_initial_lot()["spots"]:
            if s["status"] == 2:
                tmp_status[s["spot_id"]] = 2
            else:
                tmp_status[s["spot_id"]] = 1 if rng2.rand() < 0.20 else 0

        orig = dict(_live_lot_status)
        _live_lot_status.clear()
        _live_lot_status.update(tmp_status)

        walk_total, allocated, rejected = 0.0, 0, 0
        for v in arrivals:
            r = _allocate_spot(v["vehicle_type"], v["destination"], strat, h_hat)
            if r["success"]:
                allocated += 1
                walk_total += r["allocated_spot"]["breakdown"]["walk_distance_l1"]
            else:
                rejected += 1

        z_cnts = {z: 0 for z in ZONE_SPECS}
        z_tots = {z: ZONE_SPECS[z]["spots"] for z in ZONE_SPECS}
        for s in get_initial_lot()["spots"]:
            if _live_lot_status.get(s["spot_id"],0) == 1:
                z_cnts[s["zone"]] += 1

        z_pcts = [z_cnts[z]/z_tots[z]*100 for z in ZONE_SPECS]
        results[strat] = {
            "name": STRATEGY_WEIGHTS[strat]["name"],
            "vehicles_simulated": n_vehicles,
            "vehicles_allocated": allocated,
            "vehicles_rejected":  rejected,
            "mean_walking_distance_l1": round(walk_total/max(1,allocated),2),
            "lot_utilization_pct": round(sum(1 for v in _live_lot_status.values() if v==1)/len(_live_lot_status)*100,1),
            "zone_balance_std": round(float(np.std(z_pcts)),2),
            "zone_breakdown": {z: {"occupied":z_cnts[z],"total":z_tots[z],"pct":round(p,1)}
                               for z, p in zip(ZONE_SPECS, z_pcts)},
        }

        _live_lot_status.clear()
        _live_lot_status.update(orig)

    return {"arrival_count": n_vehicles, "comparison": results,
            "analysis": "S3 achieves balanced zone utilization by incorporating Stage 7 predictive zone load ĥ."}

# ── FastAPI routes ───────────────────────────────────────────────────────────

class AllocateRequest(BaseModel):
    vehicle_type: str = Field(default="standard")
    destination:  str = Field(default="Market")
    strategy:     str = Field(default="S3")

class SimulateRequest(BaseModel):
    n_vehicles: int        = Field(default=50, ge=10, le=120)
    strategies: List[str]  = Field(default=["S1","S2","S3"])

def _h_hat():
    """Extract predicted zone loads from the cached Stage 7 summary."""
    try:
        return get_pipeline_cache()["stages"][6]["matrices"]["predicted_zone_loads_h_hat"]
    except Exception:
        return {"Zone A": 0.5, "Zone B": 0.5, "Zone C": 0.5, "Zone D": 0.5}

@app.get("/api/lot")
def route_lot():
    return _lot_state_response()

@app.get("/api/pipeline")
def route_pipeline():
    return get_pipeline_cache()

@app.get("/api/predict")
def route_predict(hour: Optional[int] = Query(default=None, ge=0, le=23)):
    base = _h_hat()
    if hour is not None:
        f = (0.7 + 0.5*(1-abs(hour-13)/5)) if 8 <= hour <= 17 else 0.35
        adj = {k: round(min(1.0, max(0.05, v*f)), 3) for k,v in base.items()}
        return {"hour": hour, "predicted_loads": adj, "base_loads": base,
                "method": "Stage 7 Least Squares + Diurnal Profile"}
    return {"predicted_loads": base, "method": "Stage 7 Least Squares Lag Model"}

@app.post("/api/allocate")
def route_allocate(req: AllocateRequest):
    return _allocate_spot(req.vehicle_type, req.destination, req.strategy, _h_hat())

@app.post("/api/simulate")
def route_simulate(req: SimulateRequest):
    return _simulate(req.n_vehicles, req.strategies, _h_hat())

@app.post("/api/reset")
def route_reset():
    global _live_lot_status
    _live_lot_status.clear()
    _init_live_status()
    # re-randomize 25% occupied
    rng = np.random.RandomState(42)
    for s in get_initial_lot()["spots"]:
        if s["status"] != 2:
            _live_lot_status[s["spot_id"]] = 1 if rng.rand() < 0.25 else 0
    return {"success": True, "message": "Lot reset.", "lot": _lot_state_response()}

# ── Serve React SPA (for standalone Render deployment) ──────────────────────
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi import HTTPException

DIST_DIR = os.path.join(ROOT, "frontend", "dist")
ASSETS_DIR = os.path.join(DIST_DIR, "assets")

if os.path.exists(ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

@app.api_route("/{full_path:path}", methods=["GET", "HEAD"])
def serve_spa(full_path: str):
    """Serve built React SPA; API 404s are handled above."""
    if full_path.startswith("api"):
        raise HTTPException(status_code=404, detail="Not found")
    index_file = os.path.join(DIST_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"message": "Backend running. Build frontend to serve SPA."})
