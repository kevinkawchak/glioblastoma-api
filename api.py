"""
Glioblastoma Clinical Trial Eligibility Screening API
======================================================
Production-ready FastAPI application for real-time eligibility prediction.

Author: Clinical AI Team
Version: 1.0.0
License: Apache 2.0

Model: TabPFN v2 (Prior-Labs/TabPFN-v2-clf)
Purpose: Screen glioblastoma patients for clinical trial eligibility
"""

import os
import time
import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from functools import wraps
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Depends, Header, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import uvicorn

from models import (
    PatientInput, PredictionResponse, BatchPredictionRequest, 
    BatchPredictionResponse, HealthResponse, FeatureSchemaResponse,
    FeatureInfo, SHAPExplanation, SHAPFeature, ErrorResponse
)
from predict import get_predictor, FEATURE_COLUMNS, MODEL_VERSION

# ============================================================================
# CONFIGURATION
# ============================================================================

# Environment variables with defaults
API_KEY_ENABLED = os.getenv("API_KEY_ENABLED", "true").lower() == "true"
API_KEYS = set(os.getenv("API_KEYS", "glioblastoma-api-key-001,test-key").split(","))
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "3600"))
MODEL_PATH = os.getenv("MODEL_PATH", None)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Configure logging (HIPAA compliant - no PHI logging)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# APPLICATION SETUP
# ============================================================================

# API metadata
app = FastAPI(
    title="Glioblastoma Clinical Trial Eligibility Screening API",
    description="""
## Overview
Production-ready API for real-time glioblastoma clinical trial eligibility prediction
using TabPFN v2 (Prior-Labs/TabPFN-v2-clf).

## Features
- **Real-time predictions** with probability scores
- **SHAP explanations** for clinical interpretability
- **Age-stratified thresholds** for bias mitigation
- **Batch processing** for multiple patients
- **HIPAA-compliant** logging (no PHI)

## Key Endpoints
- `POST /predict` - Single patient prediction
- `POST /batch` - Batch predictions (up to 100 patients)
- `GET /health` - API health check
- `GET /features` - Feature schema documentation

## Clinical Domains (66 features)
1. Demographics (4)
2. Molecular Markers (2) - IDH, MGMT
3. Treatment History (5)
4. Imaging (6)
5. Laboratory Values (12)
6. Vital Signs (4)
7. Medications (8)
8. Comorbidities (10)
9. Performance Status (1) - KPS
10. Special Populations (1) - Pregnancy

## Model Information
- **Architecture**: TabPFN v2 (Transformer-based)
- **License**: Apache 2.0
- **Version**: 1.0.0
    """,
    version="1.0.0",
    contact={
        "name": "Clinical AI Team",
        "email": "clinical-ai@example.com"
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0"
    },
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration for web application integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track startup time
START_TIME = datetime.utcnow()

# Rate limiting storage (in production, use Redis)
rate_limit_storage: Dict[str, List[datetime]] = defaultdict(list)


# ============================================================================
# DEPENDENCIES
# ============================================================================

async def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """
    Verify API key for authentication.
    
    Args:
        x_api_key: API key from request header
        
    Raises:
        HTTPException: If API key is invalid
    """
    if not API_KEY_ENABLED:
        return "anonymous"
    
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Provide X-API-Key header."
        )
    
    if x_api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key."
        )
    
    return x_api_key


async def check_rate_limit(api_key: str = Depends(verify_api_key)):
    """
    Check rate limit for the API key.
    
    Args:
        api_key: Verified API key
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    current_time = datetime.utcnow()
    window_start = current_time - timedelta(seconds=RATE_LIMIT_WINDOW_SECONDS)
    
    # Clean old entries
    rate_limit_storage[api_key] = [
        t for t in rate_limit_storage[api_key] if t > window_start
    ]
    
    # Check limit
    if len(rate_limit_storage[api_key]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_REQUESTS} requests per hour."
        )
    
    # Record this request
    rate_limit_storage[api_key].append(current_time)
    
    return api_key


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_prediction_id() -> str:
    """Generate anonymous prediction ID for tracking."""
    return f"pred_{uuid.uuid4().hex[:12]}"


def patient_to_dict(patient: PatientInput) -> Dict[str, Any]:
    """Convert Pydantic model to dictionary for prediction."""
    data = patient.model_dump()
    
    # Convert enum values to strings
    for key, value in data.items():
        if hasattr(value, 'value'):
            data[key] = value.value
    
    return data


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors with detailed messages."""
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(x) for x in error["loc"])
        errors.append({
            "field": field,
            "message": error["msg"],
            "value": str(error.get("input", ""))[:100]  # Truncate for safety
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": "Input data failed validation",
            "validation_errors": errors,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors gracefully."""
    logger.error(f"Unexpected error: {type(exc).__name__}: {exc}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred. Please try again.",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """API root endpoint with welcome message."""
    return {
        "message": "Glioblastoma Clinical Trial Eligibility Screening API",
        "version": MODEL_VERSION,
        "documentation": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Health check endpoint.
    
    Returns API status, model information, and uptime.
    """
    predictor = get_predictor(MODEL_PATH)
    uptime = (datetime.utcnow() - START_TIME).total_seconds()
    
    status_value = "healthy" if predictor.is_loaded else "unhealthy"
    
    return HealthResponse(
        status=status_value,
        model_loaded=predictor.is_loaded,
        model_version=predictor.model_version,
        uptime_seconds=uptime,
        timestamp=datetime.utcnow()
    )


@app.get("/features", response_model=FeatureSchemaResponse, tags=["Documentation"])
async def get_feature_schema():
    """
    Get complete feature schema with descriptions and validation rules.
    
    Returns information about all 66 clinical features organized by domain.
    """
    # Define feature domains
    domains = {
        "Demographics": ["age", "sex", "race", "ethnicity"],
        "Molecular Markers": ["IDH_mutation_status", "MGMT_methylation"],
        "Treatment History": [
            "surgery_type", "radiation_completed", "temozolomide_received",
            "bevacizumab_received", "time_since_diagnosis_months"
        ],
        "Imaging": [
            "tumor_size_mm", "tumor_location", "enhancement_pattern",
            "edema_extent", "midline_shift", "multifocality"
        ],
        "Laboratory Values": [
            "hemoglobin", "WBC", "platelets", "creatinine", "BUN", "GFR",
            "AST", "ALT", "bilirubin", "albumin", "alkaline_phosphatase", "INR"
        ],
        "Vital Signs": ["systolic_bp", "diastolic_bp", "heart_rate", "temperature"],
        "Medications": [
            "corticosteroid_type", "corticosteroid_dose", "anticoagulants",
            "anticonvulsants", "immunosuppressants", "chemotherapy_agents",
            "targeted_therapies", "other_medications"
        ],
        "Comorbidities": [
            "cardiovascular_disease", "diabetes", "hepatic_disease", "renal_disease",
            "pulmonary_disease", "HIV_immunodeficiency", "active_infection",
            "prior_malignancy", "psychiatric_conditions", "substance_abuse"
        ],
        "Performance Status": ["karnofsky_performance_status"],
        "Special Populations": ["pregnancy_status"]
    }
    
    # Build feature info list
    features = []
    schema = PatientInput.model_json_schema()
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    
    for name, props in properties.items():
        if name == "patient_id":
            continue
            
        feature_info = FeatureInfo(
            name=name,
            type=props.get("type", "unknown"),
            required=name in required,
            description=props.get("description", ""),
            range=None,
            allowed_values=None
        )
        
        # Add range for numeric types
        if "minimum" in props or "maximum" in props:
            min_val = props.get("minimum", props.get("exclusiveMinimum", ""))
            max_val = props.get("maximum", props.get("exclusiveMaximum", ""))
            feature_info.range = f"{min_val} - {max_val}"
        
        # Add allowed values for enums
        if "enum" in props:
            feature_info.allowed_values = props["enum"]
        elif "$ref" in props or "allOf" in props:
            # Handle Pydantic enum references
            ref_name = props.get("$ref", "").split("/")[-1]
            if not ref_name and "allOf" in props:
                ref_name = props["allOf"][0].get("$ref", "").split("/")[-1]
            if ref_name and ref_name in schema.get("$defs", {}):
                enum_def = schema["$defs"][ref_name]
                if "enum" in enum_def:
                    feature_info.allowed_values = enum_def["enum"]
        
        features.append(feature_info)
    
    return FeatureSchemaResponse(
        total_features=66,
        domains=domains,
        features=features
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_eligibility(
    patient: PatientInput,
    api_key: str = Depends(check_rate_limit)
):
    """
    Predict clinical trial eligibility for a single patient.
    
    ## Input
    JSON object with 66 clinical features across 10 domains.
    
    ## Output
    - `prediction`: Binary eligibility (Eligible/Not Eligible)
    - `confidence_score`: Probability score (0-1)
    - `adjusted_prediction`: Eligibility after age-stratified threshold
    - `shap_explanation`: Top 5 features driving the prediction
    - `requires_manual_review`: Flag for borderline/elderly cases
    
    ## Age-Stratified Thresholds
    - Age <70: threshold = 0.50
    - Age ≥70: threshold = 0.45 (bias mitigation)
    
    ## Manual Review Triggers
    - Borderline predictions (0.40-0.60)
    - Age ≥70
    - Prior bevacizumab exposure
    """
    start_time = time.time()
    
    try:
        # Get predictor
        predictor = get_predictor(MODEL_PATH)
        
        # Convert to dictionary
        patient_dict = patient_to_dict(patient)
        
        # Make prediction
        result = predictor.predict(patient_dict)
        
        # Generate anonymous ID
        prediction_id = generate_prediction_id()
        
        # Log prediction (without PHI)
        logger.info(
            f"Prediction completed | ID: {prediction_id} | "
            f"Result: {result['prediction']} | "
            f"Confidence: {result['confidence_score']:.3f} | "
            f"Manual Review: {result['requires_manual_review']} | "
            f"Time: {(time.time() - start_time)*1000:.1f}ms"
        )
        
        # Build response
        return PredictionResponse(
            patient_id=prediction_id,
            prediction=result['prediction'],
            confidence_score=result['confidence_score'],
            adjusted_prediction=result['adjusted_prediction'],
            threshold_used=result['threshold_used'],
            requires_manual_review=result['requires_manual_review'],
            review_reasons=result['review_reasons'],
            shap_explanation=SHAPExplanation(
                top_5_features=[
                    SHAPFeature(**f) for f in result['shap_explanation']['top_5_features']
                ],
                recommendation=result['shap_explanation']['recommendation'],
                base_value=result['shap_explanation']['base_value']
            ),
            timestamp=datetime.utcnow(),
            model_version=MODEL_VERSION
        )
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. Please check input data and try again."
        )


@app.post("/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    request: BatchPredictionRequest,
    api_key: str = Depends(check_rate_limit)
):
    """
    Batch prediction for multiple patients (up to 100).
    
    ## Input
    JSON object with `patients` array containing patient data objects.
    
    ## Output
    - `predictions`: List of prediction results
    - `total_patients`: Number of patients processed
    - `eligible_count`: Number predicted eligible
    - `not_eligible_count`: Number predicted not eligible
    - `manual_review_count`: Number requiring manual review
    - `processing_time_ms`: Total processing time
    
    ## Notes
    - Maximum 100 patients per batch
    - Each patient is processed independently
    - Failed predictions are included with error flags
    """
    start_time = time.time()
    
    predictions = []
    eligible_count = 0
    not_eligible_count = 0
    manual_review_count = 0
    
    predictor = get_predictor(MODEL_PATH)
    
    for i, patient in enumerate(request.patients):
        try:
            patient_dict = patient_to_dict(patient)
            result = predictor.predict(patient_dict)
            
            prediction_id = generate_prediction_id()
            
            pred_response = PredictionResponse(
                patient_id=prediction_id,
                prediction=result['prediction'],
                confidence_score=result['confidence_score'],
                adjusted_prediction=result['adjusted_prediction'],
                threshold_used=result['threshold_used'],
                requires_manual_review=result['requires_manual_review'],
                review_reasons=result['review_reasons'],
                shap_explanation=SHAPExplanation(
                    top_5_features=[
                        SHAPFeature(**f) for f in result['shap_explanation']['top_5_features']
                    ],
                    recommendation=result['shap_explanation']['recommendation'],
                    base_value=result['shap_explanation']['base_value']
                ),
                timestamp=datetime.utcnow(),
                model_version=MODEL_VERSION
            )
            
            predictions.append(pred_response)
            
            # Update counts
            if result['adjusted_prediction'] == "Eligible":
                eligible_count += 1
            else:
                not_eligible_count += 1
            
            if result['requires_manual_review']:
                manual_review_count += 1
                
        except Exception as e:
            logger.error(f"Batch prediction error for patient {i}: {e}")
            # Create error response for this patient
            error_response = PredictionResponse(
                patient_id=f"error_{generate_prediction_id()}",
                prediction="Not Eligible",
                confidence_score=0.0,
                adjusted_prediction="Not Eligible",
                threshold_used=0.5,
                requires_manual_review=True,
                review_reasons=[f"Processing error: {str(e)[:100]}"],
                shap_explanation=SHAPExplanation(
                    top_5_features=[],
                    recommendation="Error during processing. Manual review required.",
                    base_value=0.0
                ),
                timestamp=datetime.utcnow(),
                model_version=MODEL_VERSION
            )
            predictions.append(error_response)
            not_eligible_count += 1
            manual_review_count += 1
    
    processing_time = (time.time() - start_time) * 1000
    
    logger.info(
        f"Batch prediction completed | "
        f"Total: {len(predictions)} | "
        f"Eligible: {eligible_count} | "
        f"Not Eligible: {not_eligible_count} | "
        f"Manual Review: {manual_review_count} | "
        f"Time: {processing_time:.1f}ms"
    )
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_patients=len(predictions),
        eligible_count=eligible_count,
        not_eligible_count=not_eligible_count,
        manual_review_count=manual_review_count,
        processing_time_ms=processing_time
    )


# ============================================================================
# EXAMPLE DATA ENDPOINTS (for documentation)
# ============================================================================

@app.get("/examples/eligible", tags=["Examples"])
async def get_eligible_example():
    """
    Get example patient data that typically predicts as Eligible.
    
    This is a synthetic example for testing and documentation purposes.
    """
    return {
        "description": "Example patient with favorable eligibility profile",
        "patient_data": {
            "age": 55,
            "sex": "M",
            "race": "White",
            "ethnicity": "Non-Hispanic",
            "IDH_mutation_status": "Wildtype",
            "MGMT_methylation": "Methylated",
            "surgery_type": "Gross Total Resection",
            "radiation_completed": True,
            "temozolomide_received": True,
            "bevacizumab_received": False,
            "time_since_diagnosis_months": 6.0,
            "tumor_size_mm": 28.0,
            "tumor_location": "Frontal lobe",
            "enhancement_pattern": "Ring-enhancing",
            "edema_extent": "Mild",
            "midline_shift": False,
            "multifocality": False,
            "hemoglobin": 14.2,
            "WBC": 6.8,
            "platelets": 280,
            "creatinine": 0.85,
            "BUN": 14,
            "GFR": 98,
            "AST": 24,
            "ALT": 28,
            "bilirubin": 0.5,
            "albumin": 4.2,
            "alkaline_phosphatase": 65,
            "INR": 1.0,
            "systolic_bp": 122,
            "diastolic_bp": 76,
            "heart_rate": 68,
            "temperature": 36.7,
            "corticosteroid_type": "Dexamethasone",
            "corticosteroid_dose": 2.0,
            "anticoagulants": False,
            "anticonvulsants": True,
            "immunosuppressants": False,
            "chemotherapy_agents": 1,
            "targeted_therapies": 0,
            "other_medications": 3,
            "cardiovascular_disease": False,
            "diabetes": False,
            "hepatic_disease": False,
            "renal_disease": False,
            "pulmonary_disease": False,
            "HIV_immunodeficiency": False,
            "active_infection": False,
            "prior_malignancy": False,
            "psychiatric_conditions": False,
            "substance_abuse": False,
            "karnofsky_performance_status": 90,
            "pregnancy_status": False
        },
        "expected_outcome": "Likely Eligible",
        "key_factors": [
            "Good KPS (90)",
            "No prior bevacizumab",
            "MGMT methylated (favorable)",
            "Adequate lab values",
            "No significant comorbidities"
        ]
    }


@app.get("/examples/not-eligible", tags=["Examples"])
async def get_not_eligible_example():
    """
    Get example patient data that typically predicts as Not Eligible.
    
    This is a synthetic example for testing and documentation purposes.
    """
    return {
        "description": "Example patient with exclusion criteria",
        "patient_data": {
            "age": 72,
            "sex": "F",
            "race": "Asian",
            "ethnicity": "Non-Hispanic",
            "IDH_mutation_status": "Wildtype",
            "MGMT_methylation": "Unmethylated",
            "surgery_type": "Biopsy Only",
            "radiation_completed": True,
            "temozolomide_received": True,
            "bevacizumab_received": True,  # Key exclusion
            "time_since_diagnosis_months": 18.0,
            "tumor_size_mm": 52.0,
            "tumor_location": "Multifocal",
            "enhancement_pattern": "Heterogeneous",
            "edema_extent": "Severe",
            "midline_shift": True,
            "multifocality": True,
            "hemoglobin": 10.8,
            "WBC": 3.2,
            "platelets": 95,
            "creatinine": 1.6,
            "BUN": 28,
            "GFR": 45,
            "AST": 68,
            "ALT": 72,
            "bilirubin": 1.4,
            "albumin": 3.1,
            "alkaline_phosphatase": 185,
            "INR": 1.4,
            "systolic_bp": 158,
            "diastolic_bp": 96,
            "heart_rate": 92,
            "temperature": 37.4,
            "corticosteroid_type": "Dexamethasone",
            "corticosteroid_dose": 8.0,
            "anticoagulants": True,
            "anticonvulsants": True,
            "immunosuppressants": False,
            "chemotherapy_agents": 3,
            "targeted_therapies": 1,
            "other_medications": 12,
            "cardiovascular_disease": True,
            "diabetes": True,
            "hepatic_disease": False,
            "renal_disease": True,
            "pulmonary_disease": False,
            "HIV_immunodeficiency": False,
            "active_infection": False,
            "prior_malignancy": False,
            "psychiatric_conditions": True,
            "substance_abuse": False,
            "karnofsky_performance_status": 50,
            "pregnancy_status": False
        },
        "expected_outcome": "Likely Not Eligible",
        "key_factors": [
            "Prior bevacizumab (exclusion)",
            "Low KPS (50)",
            "Age ≥70 (requires review)",
            "Low platelets",
            "Elevated creatinine",
            "Multiple comorbidities"
        ]
    }


@app.get("/examples/borderline", tags=["Examples"])
async def get_borderline_example():
    """
    Get example patient data that typically produces a borderline prediction.
    
    This is a synthetic example for testing and documentation purposes.
    """
    return {
        "description": "Example patient with mixed factors requiring manual review",
        "patient_data": {
            "age": 68,
            "sex": "M",
            "race": "Black",
            "ethnicity": "Non-Hispanic",
            "IDH_mutation_status": "Wildtype",
            "MGMT_methylation": "Unknown",
            "surgery_type": "Subtotal Resection",
            "radiation_completed": True,
            "temozolomide_received": True,
            "bevacizumab_received": False,
            "time_since_diagnosis_months": 10.0,
            "tumor_size_mm": 38.0,
            "tumor_location": "Temporal lobe",
            "enhancement_pattern": "Ring-enhancing",
            "edema_extent": "Moderate",
            "midline_shift": False,
            "multifocality": False,
            "hemoglobin": 11.5,
            "WBC": 5.5,
            "platelets": 165,
            "creatinine": 1.1,
            "BUN": 22,
            "GFR": 68,
            "AST": 38,
            "ALT": 42,
            "bilirubin": 0.9,
            "albumin": 3.6,
            "alkaline_phosphatase": 95,
            "INR": 1.1,
            "systolic_bp": 142,
            "diastolic_bp": 88,
            "heart_rate": 78,
            "temperature": 36.9,
            "corticosteroid_type": "Dexamethasone",
            "corticosteroid_dose": 4.0,
            "anticoagulants": False,
            "anticonvulsants": True,
            "immunosuppressants": False,
            "chemotherapy_agents": 2,
            "targeted_therapies": 0,
            "other_medications": 6,
            "cardiovascular_disease": False,
            "diabetes": True,
            "hepatic_disease": False,
            "renal_disease": False,
            "pulmonary_disease": False,
            "HIV_immunodeficiency": False,
            "active_infection": False,
            "prior_malignancy": False,
            "psychiatric_conditions": False,
            "substance_abuse": False,
            "karnofsky_performance_status": 70,
            "pregnancy_status": False
        },
        "expected_outcome": "Borderline - Manual Review Required",
        "key_factors": [
            "KPS at minimum threshold (70)",
            "Unknown MGMT status",
            "Borderline age (68)",
            "Diabetes present",
            "Elevated blood pressure"
        ]
    }


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    logger.info("Starting Glioblastoma Eligibility API...")
    
    try:
        predictor = get_predictor(MODEL_PATH)
        logger.info(f"Model loaded successfully. Version: {predictor.model_version}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Don't fail startup - model can be loaded on first request
    
    logger.info("API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Glioblastoma Eligibility API...")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
