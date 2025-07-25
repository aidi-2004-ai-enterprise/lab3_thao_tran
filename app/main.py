import xgboost as xgb
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "model.json")

# Initialize FastAPI app
app = FastAPI(
    title="Penguin Classifier API",
    description="API for predicting penguin species based on physical measurements",
    version="1.0.0"
)

# Add CORS middleware - must be before any route definitions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    year: int
    sex: str
    island: str

def load_model():
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    return model

model = load_model()

def preprocess_features(features: PenguinFeatures):
    """Preprocess input features for prediction."""
    input_dict = features.dict()  # This returns a dictionary of the model's fields
    X_input = pd.DataFrame([input_dict]) 
    X_input = pd.get_dummies(X_input, columns=["sex", "island"]) # Ensure the same 
    expected_cols = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "year",
        "sex_Female",
        "sex_Male",
        "island_Biscoe",
        "island_Dream",
        "island_Torgersen",
    ]
    X_input = X_input.reindex(columns=expected_cols, fill_value=0)
    return X_input.astype(float)

@app.get("/", tags=["Root"])
async def root():
    return {"message": "Penguin Classifier API is running! Visit /docs for interactive documentation."}

@app.post("/predict", tags=["Prediction"])
async def predict(features: PenguinFeatures):
    """Predict penguin species based on input features."""
    X_input = preprocess_features(features)
    pred = model.predict(X_input.values)
    return {"prediction": int(pred[0]), "species": ["Adelie", "Chinstrap", "Gentoo"][int(pred[0])]}

@app.get("/health", tags=["Health"])
async def health():
    """Health check endpoint."""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
