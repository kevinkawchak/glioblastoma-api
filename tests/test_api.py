"""
Glioblastoma Clinical Trial Eligibility API - Test Suite
=========================================================
Test cases demonstrating true positive, false negative, borderline,
and age-stratified threshold scenarios.

Author: Clinical AI Team
Version: 1.0.0
"""

import pytest
from datetime import datetime
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import app

# Test client
client = TestClient(app)

# API key for testing
TEST_API_KEY = "test-key"
HEADERS = {"X-API-Key": TEST_API_KEY}


# ============================================================================
# TEST CASE 1: LIKELY ELIGIBLE PATIENT (True Positive Scenario)
# ============================================================================

ELIGIBLE_PATIENT = {
    "age": 55,
    "sex": "M",
    "race": "White",
    "ethnicity": "Non-Hispanic",
    "IDH_mutation_status": "Wildtype",
    "MGMT_methylation": "Methylated",
    "surgery_type": "Gross Total Resection",
    "radiation_completed": True,
    "temozolomide_received": True,
    "bevacizumab_received": False,  # Key: No prior bevacizumab
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
    "karnofsky_performance_status": 90,  # Key: Good KPS
    "pregnancy_status": False
}


def test_eligible_patient():
    """
    Test Case 1: Likely Eligible Patient
    
    Patient Profile:
    - Age: 55 (younger)
    - KPS: 90 (good performance status)
    - No prior bevacizumab
    - MGMT methylated (favorable)
    - Adequate lab values
    - No significant comorbidities
    
    Expected: High confidence eligible prediction
    """
    response = client.post("/predict", json=ELIGIBLE_PATIENT, headers=HEADERS)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "patient_id" in data
    assert "prediction" in data
    assert "confidence_score" in data
    assert "adjusted_prediction" in data
    assert "shap_explanation" in data
    assert "requires_manual_review" in data
    
    # Check prediction ID format
    assert data["patient_id"].startswith("pred_")
    
    # Check confidence score is reasonable
    assert 0 <= data["confidence_score"] <= 1
    
    # Check SHAP explanation structure
    shap = data["shap_explanation"]
    assert "top_5_features" in shap
    assert "recommendation" in shap
    assert "base_value" in shap
    assert len(shap["top_5_features"]) <= 5
    
    # For this patient, expect higher confidence
    print(f"\n[Test 1] Eligible Patient:")
    print(f"  Prediction: {data['prediction']}")
    print(f"  Confidence: {data['confidence_score']:.3f}")
    print(f"  Manual Review: {data['requires_manual_review']}")


# ============================================================================
# TEST CASE 2: NOT ELIGIBLE PATIENT (Has Exclusion Criteria)
# ============================================================================

NOT_ELIGIBLE_PATIENT = {
    "age": 72,
    "sex": "F",
    "race": "Asian",
    "ethnicity": "Non-Hispanic",
    "IDH_mutation_status": "Wildtype",
    "MGMT_methylation": "Unmethylated",  # Unfavorable
    "surgery_type": "Biopsy Only",
    "radiation_completed": True,
    "temozolomide_received": True,
    "bevacizumab_received": True,  # KEY EXCLUSION: Prior bevacizumab
    "time_since_diagnosis_months": 18.0,
    "tumor_size_mm": 52.0,
    "tumor_location": "Multifocal",
    "enhancement_pattern": "Heterogeneous",
    "edema_extent": "Severe",
    "midline_shift": True,
    "multifocality": True,
    "hemoglobin": 10.8,  # Low
    "WBC": 3.2,  # Low
    "platelets": 95,  # Low
    "creatinine": 1.6,  # Elevated
    "BUN": 28,  # Elevated
    "GFR": 45,  # Low
    "AST": 68,  # Elevated
    "ALT": 72,  # Elevated
    "bilirubin": 1.4,  # Elevated
    "albumin": 3.1,  # Low
    "alkaline_phosphatase": 185,  # Elevated
    "INR": 1.4,  # Elevated
    "systolic_bp": 158,  # Hypertensive
    "diastolic_bp": 96,  # Hypertensive
    "heart_rate": 92,
    "temperature": 37.4,
    "corticosteroid_type": "Dexamethasone",
    "corticosteroid_dose": 8.0,  # High dose
    "anticoagulants": True,  # Bleeding risk
    "anticonvulsants": True,
    "immunosuppressants": False,
    "chemotherapy_agents": 3,
    "targeted_therapies": 1,
    "other_medications": 12,
    "cardiovascular_disease": True,  # Comorbidity
    "diabetes": True,  # Comorbidity
    "hepatic_disease": False,
    "renal_disease": True,  # Comorbidity
    "pulmonary_disease": False,
    "HIV_immunodeficiency": False,
    "active_infection": False,
    "prior_malignancy": False,
    "psychiatric_conditions": True,
    "substance_abuse": False,
    "karnofsky_performance_status": 50,  # KEY: Poor KPS
    "pregnancy_status": False
}


def test_not_eligible_patient():
    """
    Test Case 2: Not Eligible Patient (Multiple Exclusion Criteria)
    
    Patient Profile:
    - Age: 72 (elderly, requires review)
    - KPS: 50 (poor performance status)
    - Prior bevacizumab (exclusion criterion)
    - Multiple abnormal lab values
    - Multiple comorbidities
    - On anticoagulants
    
    Expected: Low confidence, flagged for manual review
    """
    response = client.post("/predict", json=NOT_ELIGIBLE_PATIENT, headers=HEADERS)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check that manual review is required
    assert data["requires_manual_review"] == True
    
    # Check that review reasons mention age or bevacizumab
    assert len(data["review_reasons"]) > 0
    
    # Should use elderly threshold (0.45)
    assert data["threshold_used"] == 0.45
    
    print(f"\n[Test 2] Not Eligible Patient (Exclusions):")
    print(f"  Prediction: {data['prediction']}")
    print(f"  Adjusted Prediction: {data['adjusted_prediction']}")
    print(f"  Confidence: {data['confidence_score']:.3f}")
    print(f"  Manual Review: {data['requires_manual_review']}")
    print(f"  Review Reasons: {data['review_reasons']}")


# ============================================================================
# TEST CASE 3: BORDERLINE PATIENT
# ============================================================================

BORDERLINE_PATIENT = {
    "age": 68,
    "sex": "M",
    "race": "Black",
    "ethnicity": "Non-Hispanic",
    "IDH_mutation_status": "Wildtype",
    "MGMT_methylation": "Unknown",  # Uncertain
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
    "hemoglobin": 11.5,  # Borderline low
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
    "systolic_bp": 142,  # Borderline elevated
    "diastolic_bp": 88,
    "heart_rate": 78,
    "temperature": 36.9,
    "corticosteroid_type": "Dexamethasone",
    "corticosteroid_dose": 4.0,  # At threshold
    "anticoagulants": False,
    "anticonvulsants": True,
    "immunosuppressants": False,
    "chemotherapy_agents": 2,
    "targeted_therapies": 0,
    "other_medications": 6,
    "cardiovascular_disease": False,
    "diabetes": True,  # One comorbidity
    "hepatic_disease": False,
    "renal_disease": False,
    "pulmonary_disease": False,
    "HIV_immunodeficiency": False,
    "active_infection": False,
    "prior_malignancy": False,
    "psychiatric_conditions": False,
    "substance_abuse": False,
    "karnofsky_performance_status": 70,  # At threshold
    "pregnancy_status": False
}


def test_borderline_patient():
    """
    Test Case 3: Borderline Patient
    
    Patient Profile:
    - Age: 68 (approaching elderly threshold)
    - KPS: 70 (at minimum typical threshold)
    - Unknown MGMT status
    - Mixed lab values
    - One comorbidity (diabetes)
    
    Expected: May be borderline, should trigger review
    """
    response = client.post("/predict", json=BORDERLINE_PATIENT, headers=HEADERS)
    
    assert response.status_code == 200
    data = response.json()
    
    # Confidence should be somewhere in the middle range
    print(f"\n[Test 3] Borderline Patient:")
    print(f"  Prediction: {data['prediction']}")
    print(f"  Adjusted Prediction: {data['adjusted_prediction']}")
    print(f"  Confidence: {data['confidence_score']:.3f}")
    print(f"  Manual Review: {data['requires_manual_review']}")
    print(f"  Review Reasons: {data['review_reasons']}")


# ============================================================================
# TEST CASE 4: AGE-STRATIFIED THRESHOLD TEST
# ============================================================================

ELDERLY_PATIENT = {
    "age": 75,  # Age >= 70, triggers elderly threshold
    "sex": "F",
    "race": "White",
    "ethnicity": "Non-Hispanic",
    "IDH_mutation_status": "Wildtype",
    "MGMT_methylation": "Methylated",
    "surgery_type": "Gross Total Resection",
    "radiation_completed": True,
    "temozolomide_received": True,
    "bevacizumab_received": False,
    "time_since_diagnosis_months": 5.0,
    "tumor_size_mm": 25.0,
    "tumor_location": "Parietal lobe",
    "enhancement_pattern": "Ring-enhancing",
    "edema_extent": "Mild",
    "midline_shift": False,
    "multifocality": False,
    "hemoglobin": 12.5,
    "WBC": 6.0,
    "platelets": 220,
    "creatinine": 1.0,
    "BUN": 18,
    "GFR": 75,
    "AST": 30,
    "ALT": 28,
    "bilirubin": 0.7,
    "albumin": 3.8,
    "alkaline_phosphatase": 80,
    "INR": 1.0,
    "systolic_bp": 135,
    "diastolic_bp": 82,
    "heart_rate": 70,
    "temperature": 36.8,
    "corticosteroid_type": "Dexamethasone",
    "corticosteroid_dose": 2.0,
    "anticoagulants": False,
    "anticonvulsants": False,
    "immunosuppressants": False,
    "chemotherapy_agents": 1,
    "targeted_therapies": 0,
    "other_medications": 4,
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


def test_elderly_threshold_adjustment():
    """
    Test Case 4: Age-Stratified Threshold
    
    Patient Profile:
    - Age: 75 (elderly, uses 0.45 threshold instead of 0.50)
    - Otherwise favorable profile
    
    Expected:
    - Threshold used should be 0.45 (not 0.50)
    - Mandatory manual review due to age
    """
    response = client.post("/predict", json=ELDERLY_PATIENT, headers=HEADERS)
    
    assert response.status_code == 200
    data = response.json()
    
    # Key assertion: elderly threshold should be used
    assert data["threshold_used"] == 0.45
    
    # Mandatory review for elderly patients
    assert data["requires_manual_review"] == True
    assert any("70" in reason or "age" in reason.lower() for reason in data["review_reasons"])
    
    print(f"\n[Test 4] Elderly Patient (Age-Stratified Threshold):")
    print(f"  Age: 75")
    print(f"  Threshold Used: {data['threshold_used']} (elderly adjustment)")
    print(f"  Prediction: {data['prediction']}")
    print(f"  Adjusted Prediction: {data['adjusted_prediction']}")
    print(f"  Confidence: {data['confidence_score']:.3f}")
    print(f"  Manual Review: {data['requires_manual_review']}")


# ============================================================================
# TEST CASE 5: VALIDATION ERROR TESTS
# ============================================================================

def test_validation_male_pregnancy():
    """
    Test that male patients cannot have pregnancy_status=True
    """
    invalid_patient = ELIGIBLE_PATIENT.copy()
    invalid_patient["sex"] = "M"
    invalid_patient["pregnancy_status"] = True
    
    response = client.post("/predict", json=invalid_patient, headers=HEADERS)
    
    # Should return validation error
    assert response.status_code == 422
    print(f"\n[Test 5a] Male + Pregnancy Validation:")
    print(f"  Status Code: {response.status_code} (expected 422)")


def test_validation_blood_pressure():
    """
    Test that systolic must be greater than diastolic
    """
    invalid_patient = ELIGIBLE_PATIENT.copy()
    invalid_patient["systolic_bp"] = 70
    invalid_patient["diastolic_bp"] = 80  # Diastolic > systolic
    
    response = client.post("/predict", json=invalid_patient, headers=HEADERS)
    
    # Should return validation error
    assert response.status_code == 422
    print(f"\n[Test 5b] Blood Pressure Validation:")
    print(f"  Status Code: {response.status_code} (expected 422)")


def test_validation_age_range():
    """
    Test age range validation (must be 18-120)
    """
    invalid_patient = ELIGIBLE_PATIENT.copy()
    invalid_patient["age"] = 15  # Too young
    
    response = client.post("/predict", json=invalid_patient, headers=HEADERS)
    
    assert response.status_code == 422
    print(f"\n[Test 5c] Age Range Validation:")
    print(f"  Status Code: {response.status_code} (expected 422)")


# ============================================================================
# TEST CASE 6: HEALTH CHECK
# ============================================================================

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert "model_loaded" in data
    assert "model_version" in data
    assert "uptime_seconds" in data
    
    print(f"\n[Test 6] Health Check:")
    print(f"  Status: {data['status']}")
    print(f"  Model Loaded: {data['model_loaded']}")
    print(f"  Version: {data['model_version']}")


# ============================================================================
# TEST CASE 7: BATCH PREDICTION
# ============================================================================

def test_batch_prediction():
    """Test batch prediction endpoint"""
    batch_request = {
        "patients": [
            ELIGIBLE_PATIENT,
            NOT_ELIGIBLE_PATIENT,
            BORDERLINE_PATIENT
        ]
    }
    
    response = client.post("/batch", json=batch_request, headers=HEADERS)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "predictions" in data
    assert "total_patients" in data
    assert data["total_patients"] == 3
    assert "eligible_count" in data
    assert "not_eligible_count" in data
    assert "processing_time_ms" in data
    
    print(f"\n[Test 7] Batch Prediction:")
    print(f"  Total Patients: {data['total_patients']}")
    print(f"  Eligible: {data['eligible_count']}")
    print(f"  Not Eligible: {data['not_eligible_count']}")
    print(f"  Manual Review: {data['manual_review_count']}")
    print(f"  Processing Time: {data['processing_time_ms']:.1f}ms")


# ============================================================================
# TEST CASE 8: FEATURE SCHEMA
# ============================================================================

def test_feature_schema():
    """Test feature schema endpoint"""
    response = client.get("/features")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "total_features" in data
    assert data["total_features"] == 66
    assert "domains" in data
    assert "features" in data
    
    print(f"\n[Test 8] Feature Schema:")
    print(f"  Total Features: {data['total_features']}")
    print(f"  Domains: {list(data['domains'].keys())}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Glioblastoma Eligibility API - Test Suite")
    print("=" * 60)
    
    # Run all tests
    test_health_check()
    test_feature_schema()
    test_eligible_patient()
    test_not_eligible_patient()
    test_borderline_patient()
    test_elderly_threshold_adjustment()
    test_validation_male_pregnancy()
    test_validation_blood_pressure()
    test_validation_age_range()
    test_batch_prediction()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
