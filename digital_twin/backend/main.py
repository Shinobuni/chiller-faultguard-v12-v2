"""FastAPI Read-Only Service for Chiller FaultGuard v12.0 (Quan Frozen HGB + Causal DAG + EVN TOU + Air-Gapped RCA + Real-World Vision ML & Live Audio FFT)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .rca_agent import run_airgapped_rca_agent
from .runtime import ChillerFaultGuardV6Runtime, StateConflictError

BASE_DIR = Path(__file__).resolve().parent.parent          # digital_twin/
FRONTEND_DIR = BASE_DIR / "frontend"
ASSETS_DIR = (BASE_DIR / "assets") if (BASE_DIR / "assets").exists() else (BASE_DIR.parent / "assets")
CAD_DIR = (
    (BASE_DIR / "hardware_cad_v12")
    if (BASE_DIR / "hardware_cad_v12").exists()
    else (BASE_DIR.parent / "hardware_cad_v12")
)

FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="7" fill="#0B1120"/>'
    '<rect x="3" y="3" width="26" height="26" rx="5" fill="none" stroke="#38BDF8" stroke-width="2"/>'
    '<path d="M9 19 L14 11 L18 16 L23 9" fill="none" stroke="#34D399" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>'
    '<circle cx="23" cy="9" r="2.2" fill="#38BDF8"/>'
    '<line x1="8" y1="23" x2="24" y2="23" stroke="#94A3B8" stroke-width="1.8" stroke-linecap="round"/>'
    "</svg>"
)


class StepRequest(BaseModel):
    steps: int = Field(default=1, ge=1, le=120)


class ScenarioRequest(BaseModel):
    scenario: str
    seek_eligible: bool = True


class JumpRequest(BaseModel):
    kind: str = Field(default="alert")


class SeekRequest(BaseModel):
    row_index: int = Field(ge=0, le=1439)
    scenario: Optional[str] = None


class InjectFaultRequest(BaseModel):
    fault_type: str


class IllustrativeParamsRequest(BaseModel):
    imposed_excess_kw: Optional[float] = Field(default=None, ge=1.0, le=50.0)
    earlier_pm_day: Optional[int] = Field(default=None, ge=22, le=160)
    existing_pm_day: Optional[int] = Field(default=None, ge=60, le=180)
    restored_fraction: Optional[float] = Field(default=None, ge=0.25, le=1.0)
    pm_cost_vnd: Optional[float] = Field(default=None, ge=1_000_000.0, le=50_000_000.0)
    reset_defaults: bool = False


class PmOptimizerRequest(BaseModel):
    pm_cost_vnd: float = Field(ge=1_000_000.0, le=50_000_000.0)


class RcaAgentRequest(BaseModel):
    query: str = Field(default="", max_length=300)
    preset: Optional[str] = Field(default=None, max_length=40)
    basis_mode: str = Field(default="live")
    lang: str = Field(default="auto")


class ReviewRequest(BaseModel):
    verdict: str = Field(default="CONFIRMED")
    engineer_name: str = Field(default="Shift A Supervisor", max_length=80)
    completed_steps: List[str] = Field(default_factory=lambda: ["I", "II", "III", "IV", "V"])
    note: str = Field(default="Verified Steps I–VI.", max_length=240)


class HardwareSimulateRequest(BaseModel):
    preset: str = Field(min_length=1, max_length=60)
    vision_mode: Optional[str] = Field(default=None, max_length=40)


class HardwareInspectFrameRequest(BaseModel):
    mode: str = Field(default="fill_strainer", min_length=1, max_length=40)
    image_base64: str = Field(min_length=8)
    source: str = Field(default="iphone11_upload", max_length=80)


class HardwareWeatherRequest(BaseModel):
    preset: Optional[str] = Field(default=None, max_length=60)
    dry_bulb_c: Optional[float] = Field(default=None, ge=-20.0, le=55.0)
    rh_pct: Optional[float] = Field(default=None, ge=5.0, le=100.0)
    wet_bulb_override_c: Optional[float] = Field(default=None, ge=-25.0, le=60.0)
    location: Optional[str] = Field(default=None, max_length=100)


class HardwareCalibrateRequest(BaseModel):
    image_base64: Optional[str] = Field(default=None)
    mode: str = Field(default="fill_strainer", min_length=1, max_length=40)
    label: str = Field(default="Clean Part 07", max_length=80)


class HardwareInspectAudioRequest(BaseModel):
    pcm_samples: List[float] = Field(min_length=256, max_length=240000)
    sample_rate: int = Field(default=16000, ge=4000, le=96000)
    source: str = Field(default="iphone11_mic", max_length=80)


def create_app(runtime_instance: Optional[ChillerFaultGuardV6Runtime] = None) -> FastAPI:
    app = FastAPI(
        title="Chiller FaultGuard v12.0 — Quan Frozen HGB, Causal DAG, EVN TOU, Air-Gapped RCA & Real-World Vision ML",
        version="12.0.0",
        description="Read-Only Advisory Digital Twin running Quan's exact frozen HistGradientBoostingClassifier with 8-node Causal DAG, EVN TOU Break-Even Optimizer, Air-Gapped RCA Agent, and Benchtop Mini-CT1 v12 Real-World Vision ML + Live PCM FFT.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=500, compresslevel=6)

    @app.middleware("http")
    async def smart_cache_headers(request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/api/") and path != "/api/hardware/aruco-sheet":
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        elif path.endswith((".css", ".js", ".svg", ".ico")) or path in ("/hardware-3d-viewer", "/api/hardware/aruco-sheet"):
            response.headers["Cache-Control"] = "public, max-age=3600, stale-while-revalidate=86400"
        else:
            response.headers["Cache-Control"] = "no-cache"
        return response

    rt = runtime_instance or ChillerFaultGuardV6Runtime(auto_seek_eligible=True)
    app.state.runtime = rt

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "version": "12.0.0",
            "release": "chiller-faultguard-v12-quan-hgb-hw",
            "service": "Chiller FaultGuard Digital Twin",
            "mode": "OFFLINE_READ_ONLY_ADVISORY",
            "read_only": True,
            "controls": False,
            "ai_engine": "QUAN_FROZEN_HGB",
            "model_sha256": rt.adapter.model_sha256,
            "proxy_fallback": False,
            "auto_swap": False,
            "causal_dag": True,
            "tou_optimizer": True,
            "air_gapped_rca": True,
            "hardware_vision_ai": True,
            "vision_ml_enabled": True,
            "cad_parts_count": 15,
            "source": "Published LBNL Oct-01 Held-Out Test Replay",
            "energy_status": "Pending weather & cooling-service validation",
            "scenario": rt.scenario,
        }

    @app.get("/api/state")
    def get_state() -> dict:
        with rt.lock:
            return rt.state()

    @app.post("/api/step")
    def step_replay(req: StepRequest = StepRequest()) -> dict:
        return rt.advance(req.steps)

    @app.post("/api/scenario")
    def set_scenario(req: ScenarioRequest) -> dict:
        try:
            return rt.reset(req.scenario, seek_eligible=req.seek_eligible)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jump")
    def jump_replay(req: JumpRequest = JumpRequest()) -> dict:
        try:
            return rt.jump(req.kind)
        except StateConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/seek")
    def seek_replay(req: SeekRequest) -> dict:
        try:
            return rt.seek(req.row_index, scenario=req.scenario)
        except StateConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/reset")
    def reset_replay() -> dict:
        return rt.reset("normal", seek_eligible=True)

    @app.post("/api/inject-fault")
    def inject_fault(req: InjectFaultRequest) -> dict:
        try:
            return rt.inject_fault(req.fault_type)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/illustrative")
    def update_illustrative(req: IllustrativeParamsRequest) -> dict:
        return rt.update_illustrative_params(
            imposed_excess_kw=req.imposed_excess_kw,
            earlier_pm_day=req.earlier_pm_day,
            existing_pm_day=req.existing_pm_day,
            restored_fraction=req.restored_fraction,
            pm_cost_vnd=req.pm_cost_vnd,
            reset_defaults=req.reset_defaults,
        )

    @app.post("/api/pm-optimizer")
    def update_pm_optimizer(req: PmOptimizerRequest) -> dict:
        try:
            return rt.update_pm_optimizer(req.pm_cost_vnd)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/rca-agent")
    def rca_agent_query(req: RcaAgentRequest) -> dict:
        try:
            rt._ensure_hardware_ready()
            with rt.lock:
                snap = rt.state()
            return run_airgapped_rca_agent(
                snap,
                query=req.query,
                preset=req.preset,
                basis_mode=req.basis_mode,
                lang=req.lang,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/acknowledge")
    def record_review(req: ReviewRequest) -> dict:
        try:
            return rt.record_review(
                verdict=req.verdict,
                engineer_name=req.engineer_name,
                completed_steps=req.completed_steps,
                note=req.note,
            )
        except StateConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/hardware/status")
    def get_hardware_status(
        img_fmt: str = Query(default="png"),
        lite: bool = Query(default=False),
    ) -> dict:
        fmt_low = "jpeg" if img_fmt.strip().lower() in ("jpg", "jpeg") else "png"
        return rt.get_hardware_status(img_fmt=fmt_low, lite=lite)

    @app.post("/api/hardware/simulate")
    def simulate_hardware(
        req: HardwareSimulateRequest,
        img_fmt: str = Query(default="png"),
    ) -> dict:
        fmt_low = "jpeg" if img_fmt.strip().lower() in ("jpg", "jpeg") else "png"
        try:
            return rt.simulate_hardware(preset=req.preset, vision_mode=req.vision_mode, img_fmt=fmt_low)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/hardware/inspect-frame")
    def inspect_hardware_frame(
        req: HardwareInspectFrameRequest,
        img_fmt: str = Query(default="png"),
    ) -> dict:
        fmt_low = "jpeg" if img_fmt.strip().lower() in ("jpg", "jpeg") else "png"
        try:
            return rt.inspect_hardware_frame(
                mode=req.mode,
                image_base64=req.image_base64,
                source=req.source,
                img_fmt=fmt_low,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/hardware/weather")
    def update_hardware_weather(
        req: HardwareWeatherRequest,
        img_fmt: str = Query(default="png"),
        lite: bool = Query(default=False),
    ) -> dict:
        fmt_low = "jpeg" if img_fmt.strip().lower() in ("jpg", "jpeg") else "png"
        try:
            return rt.update_hardware_weather(
                preset=req.preset,
                dry_bulb_c=req.dry_bulb_c,
                rh_pct=req.rh_pct,
                wet_bulb_override_c=req.wet_bulb_override_c,
                location=req.location,
                img_fmt=fmt_low,
                lite=lite,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/hardware/apply-to-step4")
    def apply_hardware_to_step4() -> dict:
        return rt.apply_hardware_to_step4()

    @app.post("/api/hardware/calibrate-bench")
    def calibrate_hardware_bench(
        req: HardwareCalibrateRequest = HardwareCalibrateRequest(),
        img_fmt: str = Query(default="png"),
    ) -> dict:
        fmt_low = "jpeg" if img_fmt.strip().lower() in ("jpg", "jpeg") else "png"
        try:
            return rt.calibrate_hardware_bench(
                image_base64=req.image_base64,
                mode=req.mode,
                label=req.label,
                img_fmt=fmt_low,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/api/hardware/calibrate-bench")
    def delete_hardware_calibration(img_fmt: str = Query(default="png")) -> dict:
        fmt_low = "jpeg" if img_fmt.strip().lower() in ("jpg", "jpeg") else "png"
        return rt.reset_hardware_calibration(img_fmt=fmt_low)

    @app.get("/api/hardware/calibration-status")
    def get_hardware_calibration_status() -> dict:
        return rt.get_hardware_calibration_status()

    @app.post("/api/hardware/inspect-audio")
    def inspect_hardware_audio(
        req: HardwareInspectAudioRequest,
        img_fmt: str = Query(default="png"),
    ) -> dict:
        fmt_low = "jpeg" if img_fmt.strip().lower() in ("jpg", "jpeg") else "png"
        try:
            return rt.inspect_hardware_audio(
                pcm_samples=req.pcm_samples,
                sample_rate=req.sample_rate,
                source=req.source,
                img_fmt=fmt_low,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/hardware/aruco-sheet")
    def serve_aruco_sheet(fmt: str = Query(default="png", alias="format")) -> FileResponse:
        fmt_low = fmt.strip().lower()
        if fmt_low not in ("png", "pdf"):
            raise HTTPException(status_code=422, detail="format must be 'png' or 'pdf'")
        media = "application/pdf" if fmt_low == "pdf" else "image/png"
        candidates = (
            f"aruco_calibration_sheet_A4.{fmt_low}",
            f"aruco_markers_printable_A4.{fmt_low}",
        )
        for fname in candidates:
            fpath = CAD_DIR / fname
            if fpath.exists():
                return FileResponse(fpath, media_type=media, filename=fname)
        raise HTTPException(status_code=404, detail=f"{candidates[0]} not yet generated")

    @app.get("/mobile-cam")
    def serve_mobile_cam() -> FileResponse:
        mobile_path = FRONTEND_DIR / "mobile_cam.html"
        if not mobile_path.exists():
            raise HTTPException(status_code=404, detail="mobile_cam.html not found")
        return FileResponse(mobile_path, media_type="text/html")

    @app.get("/hardware-3d-viewer")
    def serve_hardware_3d_viewer() -> FileResponse:
        for rel in ("assemblies/MiniCT1_Interactive_3D_Viewer.html", "assembly/MiniCT1_Interactive_3D_Viewer.html"):
            viewer_path = CAD_DIR / rel
            if viewer_path.exists():
                return FileResponse(viewer_path, media_type="text/html")
        raise HTTPException(status_code=404, detail="MiniCT1_Interactive_3D_Viewer.html not found")

    @app.get("/assets/chiller_plant_pid.svg")
    def serve_pid_svg() -> FileResponse:
        svg_path = ASSETS_DIR / "chiller_plant_pid.svg"
        if not svg_path.exists():
            raise HTTPException(status_code=404, detail="chiller_plant_pid.svg not found")
        return FileResponse(svg_path, media_type="image/svg+xml")

    @app.get("/assets/system_pipeline_architecture.svg")
    def serve_pipeline_svg() -> FileResponse:
        svg_path = ASSETS_DIR / "system_pipeline_architecture.svg"
        if not svg_path.exists():
            raise HTTPException(status_code=404, detail="system_pipeline_architecture.svg not found")
        return FileResponse(svg_path, media_type="image/svg+xml")

    @app.get("/favicon.ico", include_in_schema=False)
    def serve_favicon() -> Response:
        return Response(content=FAVICON_SVG, media_type="image/svg+xml")

    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

    return app


app = create_app()
