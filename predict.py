"""
Glioblastoma Clinical Trial Eligibility - Prediction Engine
============================================================
Model loading, feature preprocessing, and SHAP explanations.

Author: Clinical AI Team
Version: 1.0.0
License: Apache 2.0
"""

import os
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Age-stratified thresholds for bias mitigation
DEFAULT_THRESHOLD = 0.50
ELDERLY_THRESHOLD = 0.45  # More permissive for age ≥70
ELDERLY_AGE_CUTOFF = 70

# Borderline prediction range requiring manual review
BORDERLINE_LOW = 0.40
BORDERLINE_HIGH = 0.60

# Model version
MODEL_VERSION = "1.0.0"

# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

# These are the 66 features expected by the model, in the correct order
FEATURE_COLUMNS = [
    # Demographics (11 features)
    'age', 'kps_score', 'sex_male', 'sex_female',
    'race_white', 'race_black', 'race_asian', 'race_other',
    'insurance_medicare', 'insurance_medicaid', 'insurance_private',
    
    # Molecular Markers (6 features) - CRITICAL
    'idh_wildtype', 'idh_mutant', 'idh_unknown',
    'mgmt_methylated', 'mgmt_unmethylated', 'mgmt_unknown',
    
    # Tumor Site (5 features)
    'site_Frontal lobe', 'site_Multifocal', 'site_Occipital lobe',
    'site_Parietal lobe', 'site_Temporal lobe',
    
    # Treatment (3 features)
    'prior_bevacizumab', 'max_treatment_line', 'progressive_disease_count',
    
    # Imaging (6 features)
    'largest_lesion_mm', 'sum_target_lesions_mm', 'enhancing_tumor',
    't2_flair_increased', 'rano_progression', 'mri_available',
    
    # Laboratory (8 features)
    'hemoglobin', 'anc', 'platelets', 'wbc', 'creatinine', 'alt', 'ast', 'inr',
    
    # Vitals (8 features)
    'weight_kg', 'height_cm', 'bsa_m2', 'bmi',
    'systolic_bp', 'diastolic_bp', 'hypertension_bp', 'heart_rate',
    
    # Medications (8 features)
    'on_corticosteroid', 'on_antiepileptic', 'on_anticoagulant', 'on_ppi',
    'on_beta_blocker', 'medication_count', 'high_dose_dexamethasone', 'on_contraception',
    
    # Comorbidities (8 features)
    'has_hypertension', 'has_diabetes', 'has_cardiovascular', 'has_dvt',
    'has_depression', 'has_prior_cancer', 'active_comorbidity_count', 'total_comorbidity_count',
    
    # Pregnancy (3 features)
    'pregnancy_test_positive', 'pregnancy_test_negative', 'childbearing_potential'
]

# Feature descriptions for clinical interpretation
FEATURE_DESCRIPTIONS = {
    'age': 'Patient age in years',
    'kps_score': 'Karnofsky Performance Status (0-100)',
    'sex_male': 'Male sex',
    'sex_female': 'Female sex',
    'race_white': 'White race',
    'race_black': 'Black race',
    'race_asian': 'Asian race',
    'race_other': 'Other race',
    'idh_wildtype': 'IDH wildtype (worse prognosis)',
    'idh_mutant': 'IDH mutant (better prognosis)',
    'idh_unknown': 'IDH status unknown',
    'mgmt_methylated': 'MGMT methylated (better TMZ response)',
    'mgmt_unmethylated': 'MGMT unmethylated',
    'mgmt_unknown': 'MGMT status unknown',
    'prior_bevacizumab': 'Prior bevacizumab exposure (exclusion criterion)',
    'largest_lesion_mm': 'Largest tumor dimension (mm)',
    'hemoglobin': 'Hemoglobin level (g/dL)',
    'platelets': 'Platelet count (x10^9/L)',
    'creatinine': 'Creatinine level (mg/dL)',
    'on_corticosteroid': 'Currently on corticosteroids',
    'on_anticoagulant': 'Currently on anticoagulants',
    'high_dose_dexamethasone': 'High-dose dexamethasone (>4mg)',
    'has_cardiovascular': 'Cardiovascular disease history',
    'has_diabetes': 'Diabetes mellitus',
    'has_hypertension': 'Hypertension',
    'rano_progression': 'RANO progression criteria met',
    'hypertension_bp': 'Elevated blood pressure',
    'systolic_bp': 'Systolic blood pressure (mmHg)',
    'diastolic_bp': 'Diastolic blood pressure (mmHg)',
}


# ============================================================================
# PREDICTION ENGINE CLASS
# ============================================================================

class GlioblastomaPredictor:
    """
    Main prediction engine for glioblastoma clinical trial eligibility.
    
    Uses TabPFN v2 model with SHAP explanations and age-stratified thresholds.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to saved model file. If None, creates synthetic model.
        """
        self.model = None
        self.scaler = None
        self.is_loaded = False
        self.model_version = MODEL_VERSION
        self.shap_explainer = None
        self._training_data = None  # For SHAP background
        
        # Initialize model
        self._load_or_create_model(model_path)
        
    def _load_or_create_model(self, model_path: Optional[str] = None):
        """Load model from file or create a new TabPFN model."""
        try:
            if model_path and os.path.exists(model_path):
                logger.info(f"Loading model from {model_path}")
                with open(model_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.model = saved_data['model']
                    self.scaler = saved_data['scaler']
                    self._training_data = saved_data.get('training_data')
                self.is_loaded = True
                logger.info("Model loaded successfully")
            else:
                logger.info("Creating new TabPFN model")
                self._create_tabpfn_model()
                self.is_loaded = True
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Falling back to synthetic model")
            self._create_synthetic_model()
            self.is_loaded = True
    
    def _create_tabpfn_model(self):
        """Create and train a TabPFN v2 model with synthetic data."""
        try:
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
            
            # Generate synthetic training data
            np.random.seed(42)
            n_samples = 500
            n_features = len(FEATURE_COLUMNS)
            
            X_train = np.random.randn(n_samples, n_features)
            # Create realistic eligibility based on key features
            y_train = self._generate_synthetic_labels(X_train)
            
            # Create and fit scaler
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Create TabPFN v2 model (Apache 2.0 license, no auth required)
            self.model = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device='cpu')
            self.model.fit(X_train_scaled, y_train)
            
            self._training_data = X_train_scaled[:100]  # Save subset for SHAP
            logger.info("TabPFN v2 model created successfully")
            
        except ImportError:
            logger.warning("TabPFN not available, using synthetic model")
            self._create_synthetic_model()
            
    def _create_synthetic_model(self):
        """Create a synthetic model for testing when TabPFN is not available."""
        from sklearn.ensemble import GradientBoostingClassifier
        
        np.random.seed(42)
        n_samples = 500
        n_features = len(FEATURE_COLUMNS)
        
        X_train = np.random.randn(n_samples, n_features)
        y_train = self._generate_synthetic_labels(X_train)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        self._training_data = X_train_scaled[:100]
        logger.info("Synthetic model created successfully")
    
    def _generate_synthetic_labels(self, X: np.ndarray) -> np.ndarray:
        """Generate synthetic eligibility labels based on clinical logic."""
        n_samples = X.shape[0]
        
        # Key feature indices (based on FEATURE_COLUMNS)
        age_idx = 0
        kps_idx = 1
        prior_bev_idx = FEATURE_COLUMNS.index('prior_bevacizumab')
        
        # Initialize as eligible
        y = np.ones(n_samples)
        
        # Exclusion criteria
        # 1. Prior bevacizumab (strong exclusion)
        y[X[:, prior_bev_idx] > 0.5] = 0
        
        # 2. Low KPS (poor performance status)
        y[X[:, kps_idx] < -1.5] = 0
        
        # 3. Very old age with poor status
        y[(X[:, age_idx] > 1.5) & (X[:, kps_idx] < 0)] = 0
        
        # Add some randomness
        noise_mask = np.random.random(n_samples) < 0.1
        y[noise_mask] = 1 - y[noise_mask]
        
        return y.astype(int)
    
    def preprocess_patient(self, patient_data: Dict[str, Any]) -> np.ndarray:
        """
        Convert patient input to model feature vector.
        
        Args:
            patient_data: Dictionary of patient features from API input
            
        Returns:
            Numpy array of processed features
        """
        features = {}
        
        # Demographics
        features['age'] = patient_data['age']
        features['kps_score'] = patient_data['karnofsky_performance_status']
        features['sex_male'] = 1 if patient_data['sex'] == 'M' else 0
        features['sex_female'] = 1 if patient_data['sex'] == 'F' else 0
        
        # Race encoding
        race = patient_data.get('race', 'Unknown')
        features['race_white'] = 1 if race == 'White' else 0
        features['race_black'] = 1 if race == 'Black' else 0
        features['race_asian'] = 1 if race == 'Asian' else 0
        features['race_other'] = 1 if race not in ['White', 'Black', 'Asian'] else 0
        
        # Insurance (default to private if not specified)
        features['insurance_medicare'] = 0
        features['insurance_medicaid'] = 0
        features['insurance_private'] = 1
        
        # Molecular markers - CRITICAL
        idh = patient_data['IDH_mutation_status']
        features['idh_wildtype'] = 1 if idh == 'Wildtype' else 0
        features['idh_mutant'] = 1 if idh == 'Mutant' else 0
        features['idh_unknown'] = 1 if idh == 'Unknown' else 0
        
        mgmt = patient_data['MGMT_methylation']
        features['mgmt_methylated'] = 1 if mgmt == 'Methylated' else 0
        features['mgmt_unmethylated'] = 1 if mgmt == 'Unmethylated' else 0
        features['mgmt_unknown'] = 1 if mgmt == 'Unknown' else 0
        
        # Tumor site
        location = patient_data.get('tumor_location', 'Other')
        features['site_Frontal lobe'] = 1 if location == 'Frontal lobe' else 0
        features['site_Multifocal'] = 1 if location == 'Multifocal' or patient_data.get('multifocality', False) else 0
        features['site_Occipital lobe'] = 1 if location == 'Occipital lobe' else 0
        features['site_Parietal lobe'] = 1 if location == 'Parietal lobe' else 0
        features['site_Temporal lobe'] = 1 if location == 'Temporal lobe' else 0
        
        # Treatment history
        features['prior_bevacizumab'] = 1 if patient_data.get('bevacizumab_received', False) else 0
        features['max_treatment_line'] = patient_data.get('chemotherapy_agents', 0) + patient_data.get('targeted_therapies', 0)
        features['progressive_disease_count'] = 1 if patient_data.get('time_since_diagnosis_months', 0) > 6 else 0
        
        # Imaging
        features['largest_lesion_mm'] = patient_data.get('tumor_size_mm', 30)
        features['sum_target_lesions_mm'] = patient_data.get('tumor_size_mm', 30) * 1.5
        features['enhancing_tumor'] = 1 if patient_data.get('enhancement_pattern', '') != 'Non-enhancing' else 0
        features['t2_flair_increased'] = 1 if patient_data.get('edema_extent', 'None') != 'None' else 0
        features['rano_progression'] = 1  # Assume progression for trial
        features['mri_available'] = 1
        
        # Laboratory values
        features['hemoglobin'] = patient_data['hemoglobin']
        features['anc'] = patient_data.get('WBC', 7) * 0.6  # Approximate ANC from WBC
        features['platelets'] = patient_data['platelets']
        features['wbc'] = patient_data['WBC']
        features['creatinine'] = patient_data['creatinine']
        features['alt'] = patient_data['ALT']
        features['ast'] = patient_data['AST']
        features['inr'] = patient_data.get('INR', 1.0)
        
        # Vitals
        features['weight_kg'] = 75  # Default if not provided
        features['height_cm'] = 170  # Default if not provided
        features['bsa_m2'] = 1.9  # Default BSA
        features['bmi'] = 26  # Default BMI
        features['systolic_bp'] = patient_data['systolic_bp']
        features['diastolic_bp'] = patient_data['diastolic_bp']
        features['hypertension_bp'] = 1 if patient_data['systolic_bp'] > 140 or patient_data['diastolic_bp'] > 90 else 0
        features['heart_rate'] = patient_data.get('heart_rate', 75)
        
        # Medications
        features['on_corticosteroid'] = 1 if patient_data.get('corticosteroid_type', 'None') != 'None' else 0
        features['on_antiepileptic'] = 1 if patient_data.get('anticonvulsants', False) else 0
        features['on_anticoagulant'] = 1 if patient_data.get('anticoagulants', False) else 0
        features['on_ppi'] = 0  # Not directly captured
        features['on_beta_blocker'] = 0  # Not directly captured
        features['medication_count'] = patient_data.get('other_medications', 0)
        features['high_dose_dexamethasone'] = 1 if patient_data.get('corticosteroid_dose', 0) > 4 else 0
        features['on_contraception'] = 0  # Default
        
        # Comorbidities
        features['has_hypertension'] = features['hypertension_bp']
        features['has_diabetes'] = 1 if patient_data.get('diabetes', False) else 0
        features['has_cardiovascular'] = 1 if patient_data.get('cardiovascular_disease', False) else 0
        features['has_dvt'] = 0  # Not directly captured
        features['has_depression'] = 1 if patient_data.get('psychiatric_conditions', False) else 0
        features['has_prior_cancer'] = 1 if patient_data.get('prior_malignancy', False) else 0
        
        # Count active comorbidities
        comorbidity_fields = ['cardiovascular_disease', 'diabetes', 'hepatic_disease', 
                            'renal_disease', 'pulmonary_disease', 'HIV_immunodeficiency',
                            'active_infection', 'prior_malignancy', 'psychiatric_conditions',
                            'substance_abuse']
        active_count = sum(1 for f in comorbidity_fields if patient_data.get(f, False))
        features['active_comorbidity_count'] = active_count
        features['total_comorbidity_count'] = active_count
        
        # Pregnancy
        features['pregnancy_test_positive'] = 1 if patient_data.get('pregnancy_status', False) else 0
        features['pregnancy_test_negative'] = 0 if patient_data.get('pregnancy_status', False) else 1
        features['childbearing_potential'] = 1 if patient_data['sex'] == 'F' and patient_data['age'] < 50 else 0
        
        # Create feature vector in correct order
        feature_vector = np.array([features[col] for col in FEATURE_COLUMNS]).reshape(1, -1)
        
        return feature_vector
    
    def predict(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make eligibility prediction for a single patient.
        
        Args:
            patient_data: Dictionary of patient features
            
        Returns:
            Dictionary with prediction, confidence, and explanations
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Preprocess patient data
        X = self.preprocess_patient(patient_data)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get prediction probability
        proba = self.model.predict_proba(X_scaled)[0, 1]
        
        # Determine age-stratified threshold
        age = patient_data['age']
        threshold = ELDERLY_THRESHOLD if age >= ELDERLY_AGE_CUTOFF else DEFAULT_THRESHOLD
        
        # Make predictions
        prediction = "Eligible" if proba >= 0.5 else "Not Eligible"
        adjusted_prediction = "Eligible" if proba >= threshold else "Not Eligible"
        
        # Determine if manual review is needed
        requires_review = False
        review_reasons = []
        
        if BORDERLINE_LOW <= proba <= BORDERLINE_HIGH:
            requires_review = True
            review_reasons.append(f"Borderline prediction score ({proba:.2f})")
        
        if age >= ELDERLY_AGE_CUTOFF:
            requires_review = True
            review_reasons.append(f"Patient age ≥{ELDERLY_AGE_CUTOFF} requires mandatory review")
        
        if patient_data.get('bevacizumab_received', False):
            requires_review = True
            review_reasons.append("Prior bevacizumab exposure flagged for review")
        
        # Generate SHAP explanations
        shap_explanation = self._generate_shap_explanation(X_scaled, patient_data)
        
        return {
            'confidence_score': float(proba),
            'prediction': prediction,
            'adjusted_prediction': adjusted_prediction,
            'threshold_used': threshold,
            'requires_manual_review': requires_review,
            'review_reasons': review_reasons,
            'shap_explanation': shap_explanation
        }
    
    def _generate_shap_explanation(
        self, 
        X_scaled: np.ndarray, 
        patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for a prediction.
        
        Args:
            X_scaled: Scaled feature vector
            patient_data: Original patient data
            
        Returns:
            Dictionary with SHAP explanations
        """
        try:
            import shap
            
            # Create explainer if not exists
            if self.shap_explainer is None:
                background = self._training_data if self._training_data is not None else X_scaled
                
                # Use appropriate explainer based on model type
                try:
                    self.shap_explainer = shap.TreeExplainer(self.model)
                except:
                    self.shap_explainer = shap.KernelExplainer(
                        self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                        background[:50]
                    )
            
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(X_scaled)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification, use class 1
            
            shap_values = shap_values.flatten()
            
            # Get base value
            base_value = float(self.shap_explainer.expected_value)
            if isinstance(base_value, (list, np.ndarray)):
                base_value = float(base_value[1]) if len(base_value) > 1 else float(base_value[0])
                
        except Exception as e:
            logger.warning(f"SHAP calculation failed: {e}, using fallback")
            # Fallback: Use feature importance approximation
            shap_values = self._approximate_shap_values(X_scaled)
            base_value = 0.43
        
        # Get feature values
        feature_values = X_scaled.flatten()
        
        # Create feature importance ranking
        importance_df = pd.DataFrame({
            'feature': FEATURE_COLUMNS,
            'shap_value': shap_values,
            'feature_value': feature_values,
            'abs_shap': np.abs(shap_values)
        }).sort_values('abs_shap', ascending=False)
        
        # Get top 5 features
        top_5 = []
        for _, row in importance_df.head(5).iterrows():
            feature_name = row['feature']
            shap_val = row['shap_value']
            feat_val = row['feature_value']
            
            # Get clinical interpretation
            interpretation = self._get_clinical_interpretation(
                feature_name, feat_val, shap_val, patient_data
            )
            
            top_5.append({
                'feature': feature_name,
                'value': float(feat_val),
                'shap_value': float(shap_val),
                'effect': 'positive' if shap_val > 0 else 'negative',
                'clinical_interpretation': interpretation
            })
        
        # Generate recommendation
        recommendation = self._generate_recommendation(top_5, patient_data)
        
        return {
            'top_5_features': top_5,
            'recommendation': recommendation,
            'base_value': float(base_value)
        }
    
    def _approximate_shap_values(self, X_scaled: np.ndarray) -> np.ndarray:
        """Approximate SHAP values using feature perturbation."""
        base_proba = self.model.predict_proba(X_scaled)[0, 1]
        shap_approx = np.zeros(len(FEATURE_COLUMNS))
        
        for i in range(len(FEATURE_COLUMNS)):
            X_perturbed = X_scaled.copy()
            X_perturbed[0, i] = 0  # Set to mean (scaled)
            perturbed_proba = self.model.predict_proba(X_perturbed)[0, 1]
            shap_approx[i] = base_proba - perturbed_proba
            
        return shap_approx
    
    def _get_clinical_interpretation(
        self,
        feature_name: str,
        feature_value: float,
        shap_value: float,
        patient_data: Dict[str, Any]
    ) -> str:
        """Generate human-readable clinical interpretation for a feature."""
        
        effect = "increases" if shap_value > 0 else "decreases"
        
        # Feature-specific interpretations
        interpretations = {
            'kps_score': f"Karnofsky Performance Status {effect} eligibility likelihood",
            'age': f"Patient age {effect} eligibility based on protocol criteria",
            'prior_bevacizumab': f"Prior bevacizumab exposure {effect} eligibility (exclusion criterion)",
            'idh_wildtype': f"IDH wildtype status {effect} eligibility consideration",
            'idh_mutant': f"IDH mutant status {effect} eligibility consideration",
            'mgmt_methylated': f"MGMT methylation {effect} predicted treatment response",
            'hemoglobin': f"Hemoglobin level {effect} eligibility based on lab criteria",
            'platelets': f"Platelet count {effect} eligibility for treatment tolerance",
            'creatinine': f"Renal function {effect} eligibility for drug clearance",
            'on_anticoagulant': f"Anticoagulant use {effect} eligibility due to bleeding risk",
            'high_dose_dexamethasone': f"High-dose steroids {effect} eligibility per protocol",
            'hypertension_bp': f"Blood pressure status {effect} eligibility for bevacizumab trials",
            'has_cardiovascular': f"Cardiovascular history {effect} eligibility assessment",
            'rano_progression': f"Disease progression status {effect} trial entry criteria",
            'largest_lesion_mm': f"Tumor size {effect} measurable disease requirement",
        }
        
        if feature_name in interpretations:
            return interpretations[feature_name]
        
        # Generic interpretation
        description = FEATURE_DESCRIPTIONS.get(feature_name, feature_name)
        return f"{description} {effect} eligibility likelihood"
    
    def _generate_recommendation(
        self,
        top_features: List[Dict],
        patient_data: Dict[str, Any]
    ) -> str:
        """Generate clinical recommendation based on SHAP analysis."""
        
        positive_features = [f for f in top_features if f['effect'] == 'positive']
        negative_features = [f for f in top_features if f['effect'] == 'negative']
        
        recommendations = []
        
        # Check for key exclusion criteria
        if patient_data.get('bevacizumab_received', False):
            recommendations.append(
                "⚠️ Prior bevacizumab exposure is typically an exclusion criterion for "
                "bevacizumab-containing trials. Consider alternative trial designs."
            )
        
        if patient_data.get('karnofsky_performance_status', 0) < 60:
            recommendations.append(
                "⚠️ Low Karnofsky Performance Status (<60) may not meet eligibility "
                "requirements for most glioblastoma trials."
            )
        
        # Positive factors
        if positive_features:
            pos_names = [f['feature'].replace('_', ' ') for f in positive_features[:3]]
            recommendations.append(
                f"Favorable factors: {', '.join(pos_names)} support eligibility."
            )
        
        # Negative factors
        if negative_features:
            neg_names = [f['feature'].replace('_', ' ') for f in negative_features[:3]]
            recommendations.append(
                f"Consider reviewing: {', '.join(neg_names)} may impact eligibility."
            )
        
        # Age-specific recommendations
        if patient_data['age'] >= ELDERLY_AGE_CUTOFF:
            recommendations.append(
                f"Patient is ≥{ELDERLY_AGE_CUTOFF} years old. Age-adjusted threshold applied. "
                "Mandatory clinical review recommended."
            )
        
        if not recommendations:
            recommendations.append(
                "Patient profile shows balanced eligibility factors. "
                "Clinical review recommended for final determination."
            )
        
        return " ".join(recommendations)
    
    def save_model(self, path: str):
        """Save the model and scaler to a file."""
        save_data = {
            'model': self.model,
            'scaler': self.scaler,
            'training_data': self._training_data,
            'version': self.model_version,
            'timestamp': datetime.utcnow().isoformat()
        }
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        logger.info(f"Model saved to {path}")


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

# Global predictor instance (initialized on first use)
_predictor_instance: Optional[GlioblastomaPredictor] = None


def get_predictor(model_path: Optional[str] = None) -> GlioblastomaPredictor:
    """
    Get or create the global predictor instance.
    
    Args:
        model_path: Optional path to model file
        
    Returns:
        GlioblastomaPredictor instance
    """
    global _predictor_instance
    
    if _predictor_instance is None:
        _predictor_instance = GlioblastomaPredictor(model_path)
    
    return _predictor_instance


def reset_predictor():
    """Reset the global predictor instance (for testing)."""
    global _predictor_instance
    _predictor_instance = None
