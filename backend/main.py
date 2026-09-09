from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form
from typing import List
from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
pd.set_option("mode.string_storage", "python")
import aiofiles
from backend.llm.interpreter import generate_tabular_interpretation, generate_image_interpretation
from backend.assistant.schemas import AssistantQueryRequest
from backend.assistant.service import run_assistant_query
import os

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
IMG_SIZE = (224,224)
TABULAR_MODEL_PATH = Path(__file__).parent / "model" / "lung_cancer_classifier.joblib"
IMAGE_MODEL_PATH = Path(__file__).parent / "model" / "best.keras"
MAX_LLM_INTERPRETATIONS = int(os.getenv("MAX_LLM_INTERPRETATIONS", "5"))
IMAGE_MODEL_NAME = "Modelo de visão computacional (CNN - best.keras)"

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def preprocess_image_for_model(file_path: Path, model) -> np.ndarray:
    """Load an uploaded image and adapt its channels to the model input."""
    import tensorflow as tf

    img_data = tf.io.read_file(str(file_path))

    input_shape = getattr(model, "input_shape", None)
    expected_channels = None
    if input_shape and len(input_shape) >= 4:
        expected_channels = input_shape[-1]

    if expected_channels == 1:
        img = tf.image.decode_image(img_data, channels=1, expand_animations=False)
    else:
        img = tf.image.decode_image(img_data, channels=3, expand_animations=False)

    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)

    if expected_channels == 1 and img.shape[-1] == 3:
        img = tf.image.rgb_to_grayscale(img)
    elif expected_channels not in (None, 1, 3):
        raise ValueError(f"Unsupported model input channels: {expected_channels}")

    image_array = np.expand_dims(img.numpy(), axis=0)
    return image_array


def normalize_column_name(column_name: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", str(column_name).strip().upper())
    return normalized.strip("_")


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed_columns = {}
    seen = set()

    for column in df.columns:
        normalized = normalize_column_name(column)
        if normalized in seen:
            raise ValueError(f"Duplicate normalized column name detected: {normalized}")
        seen.add(normalized)
        renamed_columns[column] = normalized

    return df.rename(columns=renamed_columns)


def align_dataframe_to_expected_columns(
    df: pd.DataFrame,
    expected_columns: List[str],
) -> pd.DataFrame:
    normalized_source_columns = {
        normalize_column_name(column): column for column in df.columns
    }

    aligned_columns = {}
    missing_columns = []

    for expected_column in expected_columns:
        normalized_expected = normalize_column_name(expected_column)
        source_column = normalized_source_columns.get(normalized_expected)
        if source_column is None:
            missing_columns.append(expected_column)
            continue
        aligned_columns[source_column] = expected_column

    if missing_columns:
        raise ValueError(f"CSV missing required columns: {missing_columns}")

    return df.rename(columns=aligned_columns)[expected_columns].copy()


def load_tabular_artifact(app: FastAPI):
    """Load the tabular model artifact and cache the last load error, if any."""
    if not TABULAR_MODEL_PATH.exists():
        return None

    try:
        return joblib.load(str(TABULAR_MODEL_PATH))
    except Exception as exc:
        app.state.tabular_artifact_error = str(exc)
        return None


def load_image_model(app: FastAPI):
    """Load the image model and cache the last load error, if any."""
    if not IMAGE_MODEL_PATH.exists():
        return None

    try:
        from tensorflow import keras

        return keras.models.load_model(str(IMAGE_MODEL_PATH))
    except Exception as exc:
        app.state.image_model_error = str(exc)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model after the process has started to avoid pre-fork initialization issues
    app.state.image_model_error = None
    app.state.model = load_image_model(app)

    app.state.tabular_artifact_error = None
    app.state.tabular_artifact = load_tabular_artifact(app)
    yield
    # Clean up the ML model and release resources
    app.state.model = None
    app.state.image_model_error = None
    app.state.tabular_artifact = None
    app.state.tabular_artifact_error = None


app = FastAPI(debug=True, lifespan=lifespan)


@app.post("/analyze-tabular")
async def analyze_tabular(csv_file: UploadFile = File(...)):
    artifact = getattr(app.state, "tabular_artifact", None)
    if artifact is None:
        artifact_error = getattr(app.state, "tabular_artifact_error", None)
        if artifact_error is not None:
            return {
                "status": "error",
                "message": (
                    "Could not load tabular model artifact. "
                    "Rebuild the backend image with scikit-learn==1.6.1."
                ),
                "details": artifact_error,
            }

        if not TABULAR_MODEL_PATH.exists():
            return {
                "status": "error",
                "message": f"Tabular model artifact not found at {TABULAR_MODEL_PATH}"
            }

        artifact = load_tabular_artifact(app)
        if artifact is None:
            artifact_error = getattr(app.state, "tabular_artifact_error", None)
            return {
                "status": "error",
                "message": (
                    "Could not load tabular model artifact. "
                    "Rebuild the backend image with scikit-learn==1.6.1."
                ),
                "details": artifact_error,
            }
        app.state.tabular_artifact = artifact

    pipeline = artifact["pipeline"]
    threshold = float(artifact.get("threshold", 0.5))

    if Path(csv_file.filename).suffix.lower() != ".csv":
        return {
            "status": "error",
            "message": "Invalid file type. Only .csv files are allowed"
        }

    try:
        input_frame = pd.read_csv(csv_file.file)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Could not read CSV file: {e}"
        }

    expected_columns = artifact.get("feature_columns", [])
    try:
        input_frame = align_dataframe_to_expected_columns(input_frame, expected_columns)
    except ValueError as e:
        return {
            "status": "error",
            "message": str(e)
        }

    probabilities = pipeline.predict_proba(input_frame)[:, 1]

    results = []
    model_name = artifact.get("model_name", "unknown")

    for index, probability in enumerate(probabilities):
        probability_float = float(probability)

        if probability_float >= 0.7:
            risk_level = "alto"
        elif probability_float >= 0.4:
            risk_level = "moderado"
        else:
            risk_level = "baixo"

        result = {
            "row_index": index,
            "disease_detected": bool(probability_float >= threshold),
            "probability": probability_float,
            "risk_level": risk_level
        }

        if index < MAX_LLM_INTERPRETATIONS:
            result["llm_interpretation"] = generate_tabular_interpretation(
                result=result,
                model_name=model_name,
                threshold=threshold
            )
        else:
            result["llm_interpretation"] = "Interpretação por LLM não gerada para este registro para evitar excesso de chamadas."

        results.append(result)

    return {
        "status": "success",
        "model_name": model_name,
        "threshold": threshold,
        "interpretation_engine": "gemini-2.5-flash",
        "results": results
    }


@app.post("/analyze-images")
async def analyze_images(files: List[UploadFile] = File(...), probability_threshold: float = Form(...)):
    results = []
    print(f"Received {len(files)} files for analysis with threshold {probability_threshold}")

    model = getattr(app.state, "model", None)
    if model is None:
        image_model_error = getattr(app.state, "image_model_error", None)
        if image_model_error is None and IMAGE_MODEL_PATH.exists():
            model = load_image_model(app)
            app.state.model = model
            image_model_error = getattr(app.state, "image_model_error", None)

        if model is None:
            if image_model_error is not None:
                return {
                    "results": [
                        {
                            "filename": file.filename,
                            "status": "error",
                            "message": "Could not load image model artifact.",
                            "details": image_model_error,
                        }
                        for file in files
                    ]
                }

            return {
                "results": [
                    {
                        "filename": file.filename,
                        "status": "error",
                        "message": f"Image model artifact not found at {IMAGE_MODEL_PATH}",
                    }
                    for file in files
                ]
            }
    
    for index, file in enumerate(files):
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": "Invalid file type. Only ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff') are allowed"
            })
            continue
            
        # Save the file
        file_path = UPLOAD_DIR / file.filename
        try:
            content = await file.read()
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(content)

            # Load image, preprocess and predict with the model
            image_array = preprocess_image_for_model(file_path, model)
            prediction = model.predict(image_array)
            probability = float(prediction[0][0])
            has_disease = probability >= float(probability_threshold)

            image_result = {
                "filename": file.filename,
                "status": "success",
                "file_path": str(file_path),
                "disease_detected": has_disease,
                "probability": probability,
                "threshold_used": float(probability_threshold),
            }

            if index < MAX_LLM_INTERPRETATIONS:
                image_result["llm_interpretation"] = generate_image_interpretation(
                    result=image_result,
                    model_name=IMAGE_MODEL_NAME,
                    threshold=float(probability_threshold)
                )
            else:
                image_result["llm_interpretation"] = "Interpretação por LLM não gerada para esta imagem para evitar excesso de chamadas."

            results.append(image_result)

        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e)
            })
            
    return {"results": results}


@app.post("/assistant/query")
async def assistant_query(request: AssistantQueryRequest):
    return run_assistant_query(request)
