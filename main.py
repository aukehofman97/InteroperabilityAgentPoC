from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json

from keytransformer_module import map_all_keys_to_ontology
from event_creator_module import generate_event_for_record

app = FastAPI(
    title="Logistics Event Transformer API",
    description="Upload anonymized logistics records and get back FEDeRATED-compatible events.",
    version="1.0.0"
)

# Enable CORS for external access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust if you want to restrict access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "LLM Transformation API is live 🚀"}

@app.post("/transform/")
async def transform_file(file: UploadFile = File(...)):
    try:
        raw_data = json.loads(await file.read())
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON file."})

    try:
        # Step 1: Map keys to ontology
        mapped_data = map_all_keys_to_ontology(raw_data)

        # Step 2: Generate events
        events = []
        for record in mapped_data:
            event = generate_event_for_record(record)
            if event:
                events.append(event)

        return {"events": events}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})