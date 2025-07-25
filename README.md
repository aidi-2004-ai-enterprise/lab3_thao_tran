# Penguins Classification with XGBoost and FastAPI

This project implements a machine learning pipeline for classifying penguin species using the Seaborn penguins dataset. It includes a FastAPI web service for making predictions.

## Project Structure

```
.
├── train.py              # Script for training and evaluating the model
├── app/
│   ├── main.py          # FastAPI application
│   └── data/            # Directory for model and encoders
│       ├── model.json   # Trained XGBoost model
│       └── encoders.joblib  # Saved encoders for preprocessing
├── pyproject.toml       # Project dependencies
└── README.md            # This file
```

## Setup

1. **Install uv** (recommended Python package manager):
   ```bash
   curl -sSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install dependencies**:
   ```bash
   uv pip install -e .
   ```

3. **Install development dependencies** (optional):
   ```bash
   uv pip install -e ".[dev]"
   ```

## Usage

### 1. Train the Model

Run the training script to preprocess the data, train the model, and save it to disk:

```bash
python train.py
```

This will:
- Load the penguins dataset
- Preprocess the data (handle missing values, encode categorical variables)
- Train an XGBoost classifier
- Evaluate the model on training and test sets
- Save the model and encoders to `app/data/`

### 2. Start the FastAPI Server

Start the FastAPI development server:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

### 3. Make Predictions

You can use the API in several ways:

#### Using the Interactive API Documentation
Visit `http://127.0.0.1:8000/docs` in your browser to access the interactive API documentation (Swagger UI).

#### Using `curl`

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181.0,
    "body_mass_g": 3750.0,
    "year": 2007,
    "sex": "male",
    "island": "Torgersen"
  }'
```

#### Using Python

```python
import requests

url = "http://127.0.0.1:8000/predict"
data = {
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181.0,
    "body_mass_g": 3750.0,
    "year": 2007,
    "sex": "male",
    "island": "Torgersen"
}

response = requests.post(url, json=data)
print(response.json())
```

### 4. Check API Health

```bash
curl http://127.0.0.1:8000/health
```

## API Endpoints

- `POST /predict` - Make a prediction
- `GET /health` - Check API health and model status
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)

## Model Performance

The model's performance will be displayed in the console when you run the training script. It includes metrics such as accuracy, precision, recall, and F1-score for each class.

## Error Handling

The API includes comprehensive error handling for:
- Invalid input values
- Missing required fields
- Model not loaded
- Internal server errors

## Dependencies

- Python 3.8+
- XGBoost
- FastAPI
- scikit-learn
- pandas
- numpy
- seaborn
- uvicorn
- pydantic

## License

This project is licensed under the MIT License.
