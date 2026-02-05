import subprocess
from fastapi import FastAPI
from datetime import datetime

app = FastAPI(title="MLOps Training API")

@app.post("/train")
def train_model():
    run_ref = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    result = subprocess.run(
        ["python", "src/train.py"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return {
            "status": "error",
            "stderr": result.stderr
        }

    return {
        "status": "success",
        "message": "Modelo treinado e registrado no MLflow",
        "run_reference": run_ref
    }
