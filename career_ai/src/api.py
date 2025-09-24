from fastapi.responses import Response
# FastAPI backend for frontend calls
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn
from career_ai.src.inference import CareerPredictor
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
import json

# Initialize FastAPI app
app = FastAPI(
    title="Career AI API",
    description="AI-powered career prediction and guidance API",
    version="1.0.0"
)

# Dummy favicon route to suppress 404 errors
@app.get("/favicon.ico")
async def favicon():
    return Response(content=b"", media_type="image/x-icon")

# Add CORS middleware for React Native integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = None

try:
    predictor = CareerPredictor()
except Exception as e:
    print(f"Warning: Could not initialize predictor: {e}")

# Pydantic models for request/response
class UserProfile(BaseModel):
    # Dataset-specific fields
    Course: Optional[str] = "Unknown"
    Linguistic: Optional[int] = 10
    Musical: Optional[int] = 10
    Bodily: Optional[int] = 10
    Logical_Mathematical: Optional[int] = 10  # Will be mapped to "Logical - Mathematical"
    Spatial_Visualization: Optional[int] = 10  # Will be mapped to "Spatial-Visualization"
    Interpersonal: Optional[int] = 10
    Intrapersonal: Optional[int] = 10
    Naturalist: Optional[int] = 10
    sp: Optional[str] = "s1"  # Will be mapped to "s/p"
    P1: Optional[str] = "AVG"
    P2: Optional[str] = "AVG"
    P3: Optional[str] = "AVG"
    P4: Optional[str] = "AVG"
    P5: Optional[str] = "AVG"
    P6: Optional[str] = "AVG"
    P7: Optional[str] = "AVG"
    P8: Optional[str] = "AVG"
    
    # Additional profile fields for LLM
    skills: Optional[List[str]] = []
    interests: Optional[List[str]] = []
    experience: Optional[str] = "Not specified"
    age: Optional[int] = None
    location: Optional[str] = None

class PredictionRequest(BaseModel):
    user_profile: UserProfile
    model_type: Optional[str] = "rf"
    include_llm_guidance: Optional[bool] = True

    class Config:
        protected_namespaces = ()

class BatchPredictionRequest(BaseModel):
    users: List[UserProfile]
    model_type: Optional[str] = "rf"

    class Config:
        protected_namespaces = ()

"""Helper function to convert API input to model input."""
def convert_user_profile(user_profile: UserProfile) -> Dict:
    """Convert API user profile to model-expected format"""
    data = user_profile.dict()
    
    # Map field names to match dataset structure
    model_data = {
        "Course": data.get("Course", "Unknown"),
        "Linguistic": data.get("Linguistic", 10),
        "Musical": data.get("Musical", 10),
        "Bodily": data.get("Bodily", 10),
        "Logical - Mathematical": data.get("Logical_Mathematical", 10),
        "Spatial-Visualization": data.get("Spatial_Visualization", 10),
        "Interpersonal": data.get("Interpersonal", 10),
        "Intrapersonal": data.get("Intrapersonal", 10),
        "Naturalist": data.get("Naturalist", 10),
        "s/p": data.get("sp", "s1"),
        "P1": data.get("P1", "AVG"),
        "P2": data.get("P2", "AVG"),
        "P3": data.get("P3", "AVG"),
        "P4": data.get("P4", "AVG"),
        "P5": data.get("P5", "AVG"),
        "P6": data.get("P6", "AVG"),
        "P7": data.get("P7", "AVG"),
        "P8": data.get("P8", "AVG"),
        
        # Additional fields
        "skills": data.get("skills", []),
        "interests": data.get("interests", []),
        "experience": data.get("experience", "Not specified"),
        "age": data.get("age"),
        "location": data.get("location")
    }
    
    # Drop None values for cleanliness
    return {k: v for k, v in model_data.items() if v is not None}

# API Routes
@app.get("/")
async def root():
    """Health check endpoint"""
    return JSONResponse(content={
        "message": "Career AI API is running",
        "status": "healthy",
        "predictor_loaded": predictor is not None
    })

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return JSONResponse(content={
        "status": "healthy",
        "predictor_status": "loaded" if predictor else "not_loaded",
        "available_endpoints": [
            "/predict",
            "/predict/batch",
            "/predict/simple",
            "/careers",
            "/skills"
        ]
    })

@app.post("/predict")
async def predict_career(request: PredictionRequest):
    """
    Get comprehensive career prediction with LLM guidance
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        # Convert Pydantic model to model-expected format
        user_data = convert_user_profile(request.user_profile)

        # Get both model responses
        xgb_result = predictor.predict_career(user_data, model_type="xgb")
        rf_result = predictor.predict_career(user_data, model_type="rf")

        if request.include_llm_guidance:
            llm_guidance_xgb = predictor.get_comprehensive_guidance(user_data, model_type="xgb")
            llm_guidance_rf = predictor.get_comprehensive_guidance(user_data, model_type="rf")
        else:
            llm_guidance_xgb = None
            llm_guidance_rf = None

        return JSONResponse(content={
            "xgboost": xgb_result,
            "randomforest": rf_result,
            "llm_guidance_xgboost": llm_guidance_xgb,
            "llm_guidance_randomforest": llm_guidance_rf
        })
    except Exception as e:
        import traceback
        return JSONResponse(content={
            "error": "Prediction failed",
            "detail": str(e),
            "traceback": traceback.format_exc()
        }, status_code=500)

@app.post("/predict/simple")
async def predict_career_simple(request: PredictionRequest):
    """
    Get simple career prediction without LLM guidance
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        user_data = convert_user_profile(request.user_profile)
        xgb_result = predictor.predict_career(user_data, model_type="xgb")
        rf_result = predictor.predict_career(user_data, model_type="rf")
        return JSONResponse(content={
            "xgboost": xgb_result,
            "randomforest": rf_result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_careers_batch(request: BatchPredictionRequest):
    """
    Batch career prediction for multiple users
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        users_data = []
        for user_profile in request.users:
            user_data = convert_user_profile(user_profile)
            users_data.append(user_data)
        
        results = predictor.batch_predict(users_data, request.model_type)
        return JSONResponse(content={"predictions": results, "total_processed": len(results)})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/careers")
async def get_available_careers():
    """
    Get list of all available career predictions
    """
    if not predictor or not predictor.label_encoder:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        careers = predictor.label_encoder.classes_.tolist()
        return JSONResponse(content={"careers": careers, "total_count": len(careers)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get careers: {str(e)}")

@app.get("/skills")
async def get_skill_recommendations():
    """
    Get general skill recommendations (placeholder)
    """
    return {
        "technical_skills": [
            "Python Programming",
            "Data Analysis",
            "Machine Learning",
            "SQL",
            "JavaScript",
            "React/React Native",
            "Cloud Computing (AWS/Azure/GCP)",
            "Git/Version Control"
        ],
        "soft_skills": [
            "Communication",
            "Problem Solving",
            "Leadership",
            "Time Management",
            "Critical Thinking",
            "Adaptability",
            "Teamwork",
            "Project Management"
        ]
    }

@app.post("/feedback")
async def submit_feedback(feedback: Dict[str, Any]):
    """
    Endpoint for collecting user feedback (placeholder)
    """
    # In a real application, you would save this to a database
    return {
        "message": "Feedback received successfully",
        "feedback_id": "placeholder_id"
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "status_code": 404}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "status_code": 500}

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )