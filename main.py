from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
import uvicorn

from processor import process_project_report

app = FastAPI()


@app.get('/')
def index():
    return {'message': 'Hello Ji'}


@app.post("/process-report")
async def process_report(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        summary = process_project_report(tmp_path)
        return {"summary": summary}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
