import os
from typing import Dict, List, Any

import joblib
import numpy as np
import pandas as pd
from career_ai.src.llm_enricher import LLMEnricher

class CareerPredictor:
    """Career prediction using trained ML models + LLM enrichment."""

    def __init__(self, models_dir: str = "models"):
        # Resolve models dir relative to repo no matter where run from
        here = os.path.dirname(__file__)
        candidate = os.path.normpath(os.path.join(here, "..", models_dir))
        self.models_dir = candidate if os.path.isdir(candidate) else models_dir
        self.preprocessor = None
        self.label_encoder = None
        self.rf_model = None
        self.xgb_model = None
        self.llm_enricher = LLMEnricher(provider="gemini")
        
        self.load_models()
    
    def load_models(self):
        """Load all trained models and preprocessors"""
        try:
            # Load preprocessor and label encoder
            self.preprocessor = joblib.load(f"{self.models_dir}/preprocessor.pkl")
            self.label_encoder = joblib.load(f"{self.models_dir}/label_encoder.pkl")
            
            print(f"Loaded preprocessor and label encoder")
            print(f"Available career classes: {len(self.label_encoder.classes_)}")
            
            # Load Random Forest model
            self.rf_model = joblib.load(f"{self.models_dir}/model_rf.pkl")
            print("Random Forest model loaded")
            
            # Load XGBoost model (optional)
            try:
                self.xgb_model = joblib.load(f"{self.models_dir}/model_xgb.pkl")
                print("XGBoost model loaded")
            except Exception as ex:
                self.xgb_model = None
                print(f"XGBoost model unavailable ({ex}); proceeding with Random Forest only")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def preprocess_user_input(self, user_data: Dict) -> pd.DataFrame:
        """
        Convert user input to the format expected by the model
        """
        # Define the expected columns based on the dataset structure
        expected_columns = [
            'Course', 'Linguistic', 'Musical', 'Bodily', 'Logical - Mathematical',
            'Spatial-Visualization', 'Interpersonal', 'Intrapersonal', 'Naturalist',
            's/p', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8'
        ]
        
        # Create DataFrame with expected structure
        df_input = pd.DataFrame([user_data])
        
        # Fill missing columns with default values
        for col in expected_columns:
            if col not in df_input.columns:
                if col == 'Course':
                    df_input[col] = 'Unknown'
                elif col == 's/p':
                    df_input[col] = 'Unknown'
                elif col.startswith('P'):
                    df_input[col] = 'AVG'  # Default performance rating
                else:
                    df_input[col] = 10  # Default numerical value
        
        # Reorder columns to match training data
        df_input = df_input[expected_columns]
        
        return df_input
    
    def predict_career(self, user_data: Dict, model_type: str = "xgb") -> Dict:
        """
        Predict career for a single user using the specified model only.
        """
        df_input = self.preprocess_user_input(user_data)
        X_processed = self.preprocessor.transform(df_input)
        model_used = None
        try:
            if model_type == "xgb":
                if self.xgb_model is None:
                    raise Exception("XGBoost model not available")
                prediction = self.xgb_model.predict(X_processed)[0]
                probabilities = self.xgb_model.predict_proba(X_processed)[0]
                model_used = "XGBoost"
            elif model_type == "rf":
                if self.rf_model is None:
                    raise Exception("RandomForest model not available")
                prediction = self.rf_model.predict(X_processed)[0]
                probabilities = self.rf_model.predict_proba(X_processed)[0]
                model_used = "RandomForest"
            else:
                raise Exception(f"Unknown model_type: {model_type}")
        except Exception as e:
            print(f"Error making prediction with {model_type}: {e}")
            return {"error": str(e)}
        predicted_career = self.label_encoder.inverse_transform([prediction])[0]
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_careers = []
        for idx in top_indices:
            career = self.label_encoder.inverse_transform([idx])[0]
            prob = probabilities[idx]
            top_careers.append({
                "career": career.strip(),
                "probability": float(prob),
                "confidence": "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
            })
        return {
            "primary_prediction": predicted_career.strip(),
            "confidence": float(max(probabilities)),
            "top_predictions": top_careers,
            "model_used": model_used
        }
    
    def get_comprehensive_guidance(self, user_data: Dict, model_type: str = "rf") -> Dict:
        """
        Get ML prediction + LLM-enhanced career guidance (improved JSON output)
        """
        try:
            # Get ML prediction
            ml_prediction = self.predict_career(user_data, model_type)
            if "error" in ml_prediction:
                return ml_prediction

            # Prepare user profile for LLM
            user_profile = {
                "skills": user_data.get("skills", []),
                "interests": user_data.get("interests", []),
                "education": user_data.get("Course", "Unknown"),
                "experience": user_data.get("experience", "Not specified"),
                "GPA": user_data.get("GPA", None),
                "Major": user_data.get("Major", None),
                "Personality": user_data.get("Personality", {}),
            }

            # Get top ML predictions for LLM context
            ml_top_predictions = ml_prediction.get("top_predictions", [])
            primary_career = ml_prediction["primary_prediction"]

            # Get improved LLM guidance (summary, recommendations, next_steps)
            llm_guidance = self.llm_enricher.get_career_guidance(primary_career, user_profile, ml_top_predictions)

            # Get skill recommendations
            current_skills = user_data.get("skills", [])
            skill_recommendations = self.llm_enricher.get_skill_recommendations(
                current_skills, primary_career
            )

            # Combine results
            comprehensive_result = {
                "ml_prediction": ml_prediction,
                "llm_guidance": llm_guidance,
                "skill_recommendations": skill_recommendations,
                "timestamp": pd.Timestamp.now().isoformat(),
                "user_profile": user_profile
            }
            return comprehensive_result
        except Exception as e:
            print(f"Error generating comprehensive guidance: {e}")
            return {"error": str(e)}

    def batch_predict(self, users_data: List[Dict], model_type: str = "rf") -> List[Dict]:
        """Predict for a list of user dicts."""
        results: List[Dict] = []
        for u in users_data:
            try:
                results.append(self.predict_career(u, model_type))
            except Exception as e:
                results.append({"error": str(e)})
        return results

def test_prediction():
    """Test the prediction system with sample data."""
    try:
        predictor = CareerPredictor()

        # Sample user data matching dataset structure
        sample_user = {
            "Course": "Computer Science",
            "Linguistic": 12,
            "Musical": 8,
            "Bodily": 10,
            "Logical - Mathematical": 18,
            "Spatial-Visualization": 16,
            "Interpersonal": 14,
            "Intrapersonal": 15,
            "Naturalist": 9,
            "s/p": "s1",
            "P1": "AVG",
            "P2": "GOOD",
            "P3": "AVG",
            "P4": "BEST",
            "P5": "BEST",
            "P6": "AVG",
            "P7": "GOOD",
            "P8": "AVG",
            # Additional profile info for LLM
            "skills": ["Python", "Machine Learning", "Data Analysis"],
            "interests": ["Technology", "Problem Solving"],
            "experience": "2 years",
        }

        print("Testing career prediction...")
        result = predictor.predict_career(sample_user)
        print("Primary:", result.get("primary_prediction"), "Conf:", f"{result.get('confidence', 0):.2%}")

        comprehensive = predictor.get_comprehensive_guidance(sample_user)
        if "error" not in comprehensive:
            print("Comprehensive guidance generated.")
        else:
            print("LLM guidance error:", comprehensive.get("error"))

    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_prediction()