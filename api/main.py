from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from api.schema import PredictionRequest, PredictionResponse, ReportListResponse
from src.explainability import ensure_default_reports
from src.prediction import PredictiveMaintenanceModel

BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports"
ALLOWED_REPORT_SUFFIXES = {".html", ".png"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_default_reports(REPORTS_DIR)
    yield


app = FastAPI(
    title="Predictive Maintenance System",
    description="Predicts equipment failure probability and remaining useful life.",
    version="0.1.0",
    lifespan=lifespan,
)
model = PredictiveMaintenanceModel(base_dir=BASE_DIR)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    prediction = model.predict(payload.cycles)
    return PredictionResponse(**prediction)


@app.get("/reports", response_model=ReportListResponse)
def list_reports() -> ReportListResponse:
    ensure_default_reports(REPORTS_DIR)
    reports = sorted(path.name for path in REPORTS_DIR.iterdir() if path.suffix in ALLOWED_REPORT_SUFFIXES)
    return ReportListResponse(reports=reports)


@app.get("/reports/{report_name}")
def get_report(report_name: str) -> FileResponse:
    reports_root = REPORTS_DIR.resolve()
    report_path = (REPORTS_DIR / report_name).resolve()
    if not report_path.is_relative_to(reports_root) or report_path.suffix not in ALLOWED_REPORT_SUFFIXES:
        raise HTTPException(status_code=404, detail="Report not found.")
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")
    media_type = "text/html" if report_path.suffix == ".html" else "image/png"
    return FileResponse(report_path, media_type=media_type, filename=report_path.name)
