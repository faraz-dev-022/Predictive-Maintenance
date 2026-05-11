import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from api.schema import SensorWindow, PredictionOutput
from src.predict import predict_failure

app = FastAPI(title="Predictive Maintenance API")

@app.get("/")
def read_root():
    return {"status": "online", "message": "Predictive Maintenance API is active"}

@app.post('/predict', response_model=PredictionOutput)
def predict(payload: SensorWindow):
    """
    Predict failure risk based on the last 50 cycles of sensor readings.
    """
    from fastapi import HTTPException
    
    window = np.array(payload.cycles)
    
    # Validate shape (must be 50 cycles x N features)
    from src.predict import SEQ_COLS, SEQ_LEN
    
    if window.shape != (SEQ_LEN, len(SEQ_COLS)):
        raise HTTPException(
            status_code=422, 
            detail=f"Invalid shape {window.shape}. Expected ({SEQ_LEN}, {len(SEQ_COLS)})."
        )
        
    result = predict_failure(window)
    
    if 'error' in result:
        raise HTTPException(status_code=500, detail=result['error'])
        
    return result

@app.get("/reports/{report_name}")
def get_report(report_name: str):
    """
    Serve diagnostic reports (HTML or PNG).
    """
    report_path = Path("reports") / report_name
    
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
        
    return FileResponse(report_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
