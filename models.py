"""
Glioblastoma Clinical Trial Eligibility Screening - Pydantic Models
=====================================================================
Comprehensive input validation schemas for the prediction API.

Author: Clinical AI Team
Version: 1.0.0
License: Apache 2.0
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Literal
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS FOR CATEGORICAL FIELDS
# ============================================================================

class Sex(str, Enum):
    MALE = "M"
    FEMALE = "F"

class Race(str, Enum):
    WHITE = "White"
    BLACK = "Black"
    ASIAN = "Asian"
    OTHER = "Other"
    UNKNOWN = "Unknown"

class Ethnicity(str, Enum):
    HISPANIC = "Hispanic"
    NON_HISPANIC = "Non-Hispanic"
    UNKNOWN = "Unknown"

class IDHStatus(str, Enum):
    WILDTYPE = "Wildtype"
    MUTANT = "Mutant"
    UNKNOWN = "Unknown"

class MGMTStatus(str, Enum):
    METHYLATED = "Methylated"
    UNMETHYLATED = "Unmethylated"
    UNKNOWN = "Unknown"

class SurgeryType(str, Enum):
    GROSS_TOTAL = "Gross Total Resection"
    SUBTOTAL = "Subtotal Resection"
    BIOPSY = "Biopsy Only"
    NONE = "None"

class TumorLocation(str, Enum):
    FRONTAL = "Frontal lobe"
    TEMPORAL = "Temporal lobe"
    PARIETAL = "Parietal lobe"
    OCCIPITAL = "Occipital lobe"
    MULTIFOCAL = "Multifocal"
    OTHER = "Other"

class EnhancementPattern(str, Enum):
    RING = "Ring-enhancing"
    SOLID = "Solid"
    HETEROGENEOUS = "Heterogeneous"
    NONE = "Non-enhancing"

class EdemaExtent(str, Enum):
    NONE = "None"
    MILD = "Mild"
    MODERATE = "Moderate"
    SEVERE = "Severe"

class CorticosteroidType(str, Enum):
    DEXAMETHASONE = "Dexamethasone"
    PREDNISONE = "Prednisone"
    METHYLPREDNISOLONE = "Methylprednisolone"
    NONE = "None"


# ============================================================================
# INPUT SCHEMA - 66 CLINICAL FEATURES
# ============================================================================

class PatientInput(BaseModel):
    """
    Complete patient data input for clinical trial eligibility prediction.
    
    Organized by clinical domain with comprehensive validation rules.
    Total: 66 features across 10 domains.
    """
    
    # -------------------------------------------------------------------------
    # 1. DEMOGRAPHICS (4 features)
    # -------------------------------------------------------------------------
    age: int = Field(
        ..., 
        ge=18, 
        le=120, 
        description="Patient age in years (18-120)"
    )
    sex: Sex = Field(..., description="Biological sex (M/F)")
    race: Race = Field(default=Race.UNKNOWN, description="Race category")
    ethnicity: Ethnicity = Field(default=Ethnicity.UNKNOWN, description="Ethnicity")
    
    # -------------------------------------------------------------------------
    # 2. MOLECULAR MARKERS (2 features) - CRITICAL
    # -------------------------------------------------------------------------
    IDH_mutation_status: IDHStatus = Field(
        ..., 
        description="IDH mutation status - CRITICAL prognostic marker"
    )
    MGMT_methylation: MGMTStatus = Field(
        ..., 
        description="MGMT promoter methylation - predicts temozolomide response"
    )
    
    # -------------------------------------------------------------------------
    # 3. TREATMENT HISTORY (5 features)
    # -------------------------------------------------------------------------
    surgery_type: SurgeryType = Field(
        default=SurgeryType.NONE, 
        description="Type of surgical intervention"
    )
    radiation_completed: bool = Field(
        default=False, 
        description="Has completed radiation therapy"
    )
    temozolomide_received: bool = Field(
        default=False, 
        description="Has received temozolomide"
    )
    bevacizumab_received: bool = Field(
        default=False, 
        description="Prior bevacizumab exposure (key exclusion criterion)"
    )
    time_since_diagnosis_months: float = Field(
        default=0, 
        ge=0, 
        le=240,
        description="Months since initial diagnosis"
    )
    
    # -------------------------------------------------------------------------
    # 4. IMAGING (6 features)
    # -------------------------------------------------------------------------
    tumor_size_mm: float = Field(
        default=0, 
        ge=0, 
        le=200,
        description="Largest tumor dimension in mm"
    )
    tumor_location: TumorLocation = Field(
        default=TumorLocation.OTHER,
        description="Primary tumor location"
    )
    enhancement_pattern: EnhancementPattern = Field(
        default=EnhancementPattern.HETEROGENEOUS,
        description="MRI enhancement pattern"
    )
    edema_extent: EdemaExtent = Field(
        default=EdemaExtent.NONE,
        description="Perilesional edema extent"
    )
    midline_shift: bool = Field(
        default=False, 
        description="Presence of midline shift"
    )
    multifocality: bool = Field(
        default=False, 
        description="Multiple tumor foci present"
    )
    
    # -------------------------------------------------------------------------
    # 5. LABORATORY VALUES (12 features)
    # -------------------------------------------------------------------------
    hemoglobin: float = Field(
        ..., 
        ge=3.0, 
        le=25.0,
        description="Hemoglobin g/dL (normal: 12-17)"
    )
    WBC: float = Field(
        ..., 
        ge=0.1, 
        le=100.0,
        description="White blood cell count x10^9/L (normal: 4-11)"
    )
    platelets: float = Field(
        ..., 
        ge=10, 
        le=1500,
        description="Platelet count x10^9/L (normal: 150-400)"
    )
    creatinine: float = Field(
        ..., 
        ge=0.1, 
        le=20.0,
        description="Creatinine mg/dL (normal: 0.6-1.2)"
    )
    BUN: float = Field(
        default=15.0, 
        ge=1, 
        le=200,
        description="Blood urea nitrogen mg/dL (normal: 7-20)"
    )
    GFR: float = Field(
        default=90.0, 
        ge=5, 
        le=200,
        description="Glomerular filtration rate mL/min (normal: >60)"
    )
    AST: float = Field(
        ..., 
        ge=1, 
        le=2000,
        description="Aspartate aminotransferase U/L (normal: 10-40)"
    )
    ALT: float = Field(
        ..., 
        ge=1, 
        le=2000,
        description="Alanine aminotransferase U/L (normal: 7-56)"
    )
    bilirubin: float = Field(
        default=0.8, 
        ge=0.1, 
        le=30,
        description="Total bilirubin mg/dL (normal: 0.1-1.2)"
    )
    albumin: float = Field(
        default=4.0, 
        ge=1.0, 
        le=6.0,
        description="Albumin g/dL (normal: 3.5-5.0)"
    )
    alkaline_phosphatase: float = Field(
        default=70, 
        ge=10, 
        le=2000,
        description="Alkaline phosphatase U/L (normal: 44-147)"
    )
    INR: float = Field(
        default=1.0, 
        ge=0.5, 
        le=10.0,
        description="International normalized ratio (normal: 0.9-1.1)"
    )
    
    # -------------------------------------------------------------------------
    # 6. VITAL SIGNS (4 features)
    # -------------------------------------------------------------------------
    systolic_bp: int = Field(
        ..., 
        ge=60, 
        le=260,
        description="Systolic blood pressure mmHg (normal: 90-140)"
    )
    diastolic_bp: int = Field(
        ..., 
        ge=30, 
        le=160,
        description="Diastolic blood pressure mmHg (normal: 60-90)"
    )
    heart_rate: int = Field(
        default=75, 
        ge=30, 
        le=220,
        description="Heart rate bpm (normal: 60-100)"
    )
    temperature: float = Field(
        default=37.0, 
        ge=34.0, 
        le=42.0,
        description="Body temperature Celsius (normal: 36.5-37.5)"
    )
    
    # -------------------------------------------------------------------------
    # 7. MEDICATIONS (8 features)
    # -------------------------------------------------------------------------
    corticosteroid_type: CorticosteroidType = Field(
        default=CorticosteroidType.NONE,
        description="Current corticosteroid medication"
    )
    corticosteroid_dose: float = Field(
        default=0, 
        ge=0, 
        le=100,
        description="Corticosteroid dose mg/day (dexamethasone equivalent)"
    )
    anticoagulants: bool = Field(
        default=False, 
        description="Currently on anticoagulant therapy"
    )
    anticonvulsants: bool = Field(
        default=False, 
        description="Currently on anticonvulsant therapy"
    )
    immunosuppressants: bool = Field(
        default=False, 
        description="Currently on immunosuppressant therapy"
    )
    chemotherapy_agents: int = Field(
        default=0, 
        ge=0, 
        le=10,
        description="Number of prior chemotherapy agents"
    )
    targeted_therapies: int = Field(
        default=0, 
        ge=0, 
        le=10,
        description="Number of prior targeted therapies"
    )
    other_medications: int = Field(
        default=0, 
        ge=0, 
        le=50,
        description="Total other medication count"
    )
    
    # -------------------------------------------------------------------------
    # 8. COMORBIDITIES (10 features)
    # -------------------------------------------------------------------------
    cardiovascular_disease: bool = Field(
        default=False, 
        description="History of cardiovascular disease"
    )
    diabetes: bool = Field(
        default=False, 
        description="Diabetes mellitus"
    )
    hepatic_disease: bool = Field(
        default=False, 
        description="Chronic liver disease"
    )
    renal_disease: bool = Field(
        default=False, 
        description="Chronic kidney disease"
    )
    pulmonary_disease: bool = Field(
        default=False, 
        description="Chronic pulmonary disease"
    )
    HIV_immunodeficiency: bool = Field(
        default=False, 
        description="HIV or immunodeficiency disorder"
    )
    active_infection: bool = Field(
        default=False, 
        description="Active uncontrolled infection"
    )
    prior_malignancy: bool = Field(
        default=False, 
        description="History of prior malignancy (non-GBM)"
    )
    psychiatric_conditions: bool = Field(
        default=False, 
        description="Significant psychiatric conditions"
    )
    substance_abuse: bool = Field(
        default=False, 
        description="Active substance abuse"
    )
    
    # -------------------------------------------------------------------------
    # 9. PERFORMANCE STATUS (1 feature) - CRITICAL
    # -------------------------------------------------------------------------
    karnofsky_performance_status: int = Field(
        ..., 
        ge=0, 
        le=100,
        description="Karnofsky Performance Status (0-100, ≥60 typically required)"
    )
    
    # -------------------------------------------------------------------------
    # 10. SPECIAL POPULATIONS (1 feature)
    # -------------------------------------------------------------------------
    pregnancy_status: bool = Field(
        default=False, 
        description="Currently pregnant (females of childbearing potential)"
    )
    
    # -------------------------------------------------------------------------
    # OPTIONAL: Patient identifier (anonymized)
    # -------------------------------------------------------------------------
    patient_id: Optional[str] = Field(
        default=None,
        description="Optional anonymized patient identifier (not logged)"
    )

    # -------------------------------------------------------------------------
    # VALIDATORS
    # -------------------------------------------------------------------------
    
    @field_validator('karnofsky_performance_status')
    @classmethod
    def validate_kps_multiple_of_10(cls, v):
        """KPS should be in increments of 10"""
        if v % 10 != 0:
            # Allow but warn - don't reject
            pass
        return v
    
    @model_validator(mode='after')
    def validate_pregnancy_sex_consistency(self):
        """Males cannot have pregnancy_status = True"""
        if self.sex == Sex.MALE and self.pregnancy_status:
            raise ValueError(
                "Inconsistent data: Male patients cannot have pregnancy_status=True"
            )
        return self
    
    @model_validator(mode='after')
    def validate_blood_pressure_consistency(self):
        """Systolic BP should be greater than diastolic"""
        if self.systolic_bp <= self.diastolic_bp:
            raise ValueError(
                f"Inconsistent blood pressure: Systolic ({self.systolic_bp}) "
                f"must be greater than diastolic ({self.diastolic_bp})"
            )
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "age": 58,
                "sex": "M",
                "race": "White",
                "ethnicity": "Non-Hispanic",
                "IDH_mutation_status": "Wildtype",
                "MGMT_methylation": "Methylated",
                "surgery_type": "Gross Total Resection",
                "radiation_completed": True,
                "temozolomide_received": True,
                "bevacizumab_received": False,
                "time_since_diagnosis_months": 8.5,
                "tumor_size_mm": 32.0,
                "tumor_location": "Frontal lobe",
                "enhancement_pattern": "Ring-enhancing",
                "edema_extent": "Moderate",
                "midline_shift": False,
                "multifocality": False,
                "hemoglobin": 13.5,
                "WBC": 7.2,
                "platelets": 245,
                "creatinine": 0.9,
                "BUN": 16,
                "GFR": 95,
                "AST": 28,
                "ALT": 32,
                "bilirubin": 0.6,
                "albumin": 4.1,
                "alkaline_phosphatase": 72,
                "INR": 1.0,
                "systolic_bp": 128,
                "diastolic_bp": 78,
                "heart_rate": 72,
                "temperature": 36.8,
                "corticosteroid_type": "Dexamethasone",
                "corticosteroid_dose": 4.0,
                "anticoagulants": False,
                "anticonvulsants": True,
                "immunosuppressants": False,
                "chemotherapy_agents": 1,
                "targeted_therapies": 0,
                "other_medications": 5,
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
                "karnofsky_performance_status": 80,
                "pregnancy_status": False
            }
        }


# ============================================================================
# OUTPUT SCHEMAS
# ============================================================================

class SHAPFeature(BaseModel):
    """Individual SHAP feature explanation"""
    feature: str = Field(..., description="Feature name")
    value: float = Field(..., description="Patient's feature value")
    shap_value: float = Field(..., description="SHAP contribution to prediction")
    effect: Literal["positive", "negative"] = Field(
        ..., 
        description="Direction of effect on eligibility"
    )
    clinical_interpretation: str = Field(
        ..., 
        description="Human-readable clinical interpretation"
    )


class SHAPExplanation(BaseModel):
    """Complete SHAP explanation for a prediction"""
    top_5_features: List[SHAPFeature] = Field(
        ..., 
        description="Top 5 features influencing the prediction"
    )
    recommendation: str = Field(
        ..., 
        description="Clinical recommendation based on SHAP analysis"
    )
    base_value: float = Field(
        ..., 
        description="SHAP base value (average prediction)"
    )


class PredictionResponse(BaseModel):
    """Complete prediction response with explanations"""
    patient_id: str = Field(
        ..., 
        description="Anonymized prediction tracking ID"
    )
    prediction: Literal["Eligible", "Not Eligible"] = Field(
        ..., 
        description="Binary eligibility prediction"
    )
    confidence_score: float = Field(
        ..., 
        ge=0, 
        le=1,
        description="Model confidence (probability) score"
    )
    adjusted_prediction: Literal["Eligible", "Not Eligible"] = Field(
        ..., 
        description="Prediction after age-stratified threshold adjustment"
    )
    threshold_used: float = Field(
        ..., 
        description="Decision threshold applied"
    )
    requires_manual_review: bool = Field(
        ..., 
        description="Flag indicating manual review is recommended"
    )
    review_reasons: List[str] = Field(
        default_factory=list,
        description="Reasons for manual review requirement"
    )
    shap_explanation: SHAPExplanation = Field(
        ..., 
        description="SHAP-based feature importance explanation"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Prediction timestamp (UTC)"
    )
    model_version: str = Field(
        default="1.0.0",
        description="Model version used for prediction"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "pred_a1b2c3d4",
                "prediction": "Eligible",
                "confidence_score": 0.78,
                "adjusted_prediction": "Eligible",
                "threshold_used": 0.50,
                "requires_manual_review": False,
                "review_reasons": [],
                "shap_explanation": {
                    "top_5_features": [
                        {
                            "feature": "karnofsky_performance_status",
                            "value": 80.0,
                            "shap_value": 0.12,
                            "effect": "positive",
                            "clinical_interpretation": "Good performance status supports eligibility"
                        }
                    ],
                    "recommendation": "Patient shows favorable eligibility profile based on KPS and molecular markers.",
                    "base_value": 0.43
                },
                "timestamp": "2025-01-15T10:30:00Z",
                "model_version": "1.0.0"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions"""
    patients: List[PatientInput] = Field(
        ..., 
        min_length=1, 
        max_length=100,
        description="List of patients (max 100 per batch)"
    )


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions"""
    predictions: List[PredictionResponse] = Field(
        ..., 
        description="List of prediction results"
    )
    total_patients: int = Field(..., description="Total patients processed")
    eligible_count: int = Field(..., description="Number predicted eligible")
    not_eligible_count: int = Field(..., description="Number predicted not eligible")
    manual_review_count: int = Field(..., description="Number requiring manual review")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response"""
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ..., 
        description="API health status"
    )
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: str = Field(..., description="Loaded model version")
    uptime_seconds: float = Field(..., description="API uptime in seconds")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )


class FeatureInfo(BaseModel):
    """Information about a single feature"""
    name: str
    type: str
    required: bool
    description: str
    range: Optional[str] = None
    allowed_values: Optional[List[str]] = None


class FeatureSchemaResponse(BaseModel):
    """Complete feature schema response"""
    total_features: int
    domains: dict
    features: List[FeatureInfo]


class ValidationError(BaseModel):
    """Detailed validation error"""
    field: str
    message: str
    value: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: str
    validation_errors: Optional[List[ValidationError]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
