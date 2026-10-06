"""
FastAPI Backend Application for Parking Allocation Using Linear Algebra.

Features:
- CORS middleware enabled
- GET  /api/lot       - Live parking lot state, zones, dimensions, spots, and model disclosure
- GET  /api/pipeline  - Precomputed and cached 10-stage linear algebra pipeline
- GET  /api/predict   - Predicted zone loads from Stage 7 least squares forecast
- POST /api/allocate  - Real-time optimal spot allocation using cost functional argmin C(v, s)
- POST /api/simulate  - Multi-strategy comparative simulation (S1, S2, S3)
- POST /api/reset     - Resets lot grid to clean state
- Compatible with Vercel Serverless Functions and standalone Uvicorn serving
"""

import os
from typing import List, Optional
from fastapi import FastAPI, APIRouter, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.core.pipeline import get_math_pipeline
from backend.core.lot_model import get_lot, ZONE_SPECS, DESTINATIONS, STRATEGY_WEIGHTS
from backend.core.data_processor import get_data_processor

app = FastAPI(
    title="Parking Allocation Using Linear Algebra",
    description="University Linear Algebra Mini-Project Web Application grounded in UCI Parking Birmingham dataset.",
    version="1.0.0"
)

# Enable CORS for local Vite development server and Vercel deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schemas
class AllocateRequest(BaseModel):
    vehicle_type: str = Field(default="standard", description="standard, compact, ev, accessible")
    destination: str = Field(default="Market", description="Market, Town Hall, Broad St, Mailbox Mall")
    strategy: str = Field(default="S3", description="S1 (Entry L2), S2 (Dest L1), S3 (Full Weighted + Prediction)")


class SimulateRequest(BaseModel):
    n_vehicles: int = Field(default=50, ge=10, le=120)
    strategies: List[str] = Field(default=["S1", "S2", "S3"])


# API Router - defined once, mounted at both /api and root to handle any Vercel URL rewrite mode
api_router = APIRouter()


@api_router.get("/lot")
def get_lot_state():
    """Returns the live parking lot state, grid matrix L, spots, and zones."""
    lot = get_lot()
    return lot.get_state()


@api_router.get("/pipeline")
def get_pipeline():
    """Returns the full 10-stage linear algebra pipeline with Concept, Purpose, and Outcome."""
    pipeline = get_math_pipeline()
    return pipeline.pipeline_results


@api_router.get("/predict")
def get_predictions(hour: Optional[int] = Query(default=None, ge=0, le=23)):
    """
    Returns Stage 7 predicted occupancy and normalized load fractions for the 4 zones.
    If hour is specified, scales predicted load based on diurnal Birmingham traffic patterns.
    """
    pipeline = get_math_pipeline()
    stage7 = pipeline.pipeline_results["stages"][6]
    base_loads = dict(stage7["matrices"]["predicted_zone_loads_h_hat"])

    # If specific hour provided, apply realistic diurnal adjustment
    if hour is not None:
        if 8 <= hour <= 17:
            factor = 0.7 + 0.5 * (1.0 - abs(hour - 13.0) / 5.0)
        else:
            factor = 0.35
        adjusted_loads = {k: round(min(1.0, max(0.05, v * factor)), 3) for k, v in base_loads.items()}
        return {
            "hour": hour,
            "predicted_loads": adjusted_loads,
            "base_loads": base_loads,
            "method": "Stage 7 Least Squares Autoregressive Lag Model + Diurnal Hourly Profile"
        }

    return {
        "predicted_loads": base_loads,
        "method": "Stage 7 Least Squares Autoregressive Lag Model (Next 30-min Slot)"
    }


@api_router.post("/allocate")
def allocate_spot(req: AllocateRequest):
    """
    Allocates the lowest-cost parking spot according to C(v, s) = alpha*||p-d||_1 + beta*||p-e||_2 + gamma*h_hat,
    checks vehicle compatibility, updates lot state matrix L, and returns cost breakdown + alternatives.
    """
    lot = get_lot()
    pipeline = get_math_pipeline()
    stage7 = pipeline.pipeline_results["stages"][6]
    predicted_loads = stage7["matrices"]["predicted_zone_loads_h_hat"]

    result = lot.allocate(
        vehicle_type=req.vehicle_type,
        dest_key=req.destination,
        strategy=req.strategy,
        predicted_zone_loads=predicted_loads
    )
    return result


@api_router.post("/simulate")
def simulate_strategies(req: SimulateRequest):
    """
    Simulates multi-vehicle arrivals across strategies S1, S2, S3 on an identical random sequence.
    """
    lot = get_lot()
    pipeline = get_math_pipeline()
    stage7 = pipeline.pipeline_results["stages"][6]
    predicted_loads = stage7["matrices"]["predicted_zone_loads_h_hat"]

    result = lot.simulate_strategies(
        n_vehicles=req.n_vehicles,
        strategies=req.strategies,
        predicted_zone_loads=predicted_loads
    )
    return result


@api_router.post("/reset")
def reset_lot():
    """Resets the live lot occupancy matrix back to initial state."""
    lot = get_lot()
    lot.reset(initial_occupied_rate=0.25)
    return {"success": True, "message": "Lot reset to initial state.", "lot": lot.get_state()}


# Mount routes under /api (standard) AND root (for Vercel rewrites)
app.include_router(api_router, prefix="/api")
app.include_router(api_router)


# Standalone Uvicorn serving: mounts static files if running locally outside Vercel
DIST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
ASSETS_DIR = os.path.join(DIST_DIR, "assets")

if os.path.exists(ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

@app.api_route("/{full_path:path}", methods=["GET", "HEAD"])
def serve_frontend_spa(full_path: str):
    """Serves built React SPA for standalone single-command run."""
    if full_path.startswith("api"):
        raise HTTPException(status_code=404, detail="API endpoint not found")
    index_file = os.path.join(DIST_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {
        "message": "FastAPI Backend running. For local standalone UI, run: cd frontend && npm run build"
    }
