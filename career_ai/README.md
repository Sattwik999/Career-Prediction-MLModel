# Career AI

A comprehensive machine learning project for career prediction and guidance, integrating ML models with LLM-powered insights.

## Features
- **ML-based Career Prediction**: RandomForest and XGBoost models for accurate career predictions
- **LLM Integration**: Gemini/OpenAI APIs for detailed career guidance and skill recommendations
- **FastAPI Backend**: RESTful API for frontend integration
- **Comprehensive Preprocessing**: Automated data preprocessing pipeline
- **Batch Processing**: Support for multiple user predictions

## Project Structure
```
career_ai/
├─ data/
│  └─ career_dataset.csv            # Raw dataset
├─ models/
│  ├─ preprocessor.pkl              # Scikit-learn preprocessing pipeline
│  ├─ label_encoder.pkl             # Label encoder for career/job roles
│  ├─ model_rf.pkl                  # Trained RandomForest model
│  └─ model_xgb.pkl                 # Trained XGBoost model (optional)
├─ src/
│  ├─ preprocess.py                 # Data preprocessing & pipeline creation
│  ├─ train.py                      # Model training & saving
│  ├─ inference.py                  # ML inference + LLM integration
│  ├─ llm_enricher.py               # Calls LLM (Gemini/OpenAI) for guidance
│  └─ api.py                        # FastAPI backend for frontend calls
├─ requirements.txt                 # Python dependencies
└─ README.md
```

## Setup & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file or set environment variables:
```bash
# For Gemini API
export GEMINI_API_KEY="your_gemini_api_key"

# For OpenAI API (alternative)
export OPENAI_API_KEY="your_openai_api_key"
```

### 3. Prepare Dataset
- Place your career dataset in `data/career_dataset.csv`
- Ensure it has appropriate features and a `career` target column

## Usage

### 1. Data Preprocessing
```bash
cd src
python preprocess.py
```
This will:
- Load and clean the dataset
- Create preprocessing pipelines
- Split data for training
- Save preprocessors and processed data

### 2. Model Training
```bash
python train.py
```
This will:
- Train RandomForest and XGBoost models
- Evaluate model performance
- Save trained models
- Generate training summary

### 3. Running Inference
```bash
python inference.py
```
Test single-user prediction with comprehensive LLM guidance.

### 4. Start API Server
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Core Prediction Endpoints
- `GET /` - Health check
- `POST /predict` - Comprehensive prediction with LLM guidance
- `POST /predict/simple` - ML prediction only
- `POST /predict/batch` - Batch predictions for multiple users

### Utility Endpoints
- `GET /careers` - List all available career options
- `GET /skills` - Get skill recommendations
- `POST /feedback` - Submit user feedback

## Dataset Structure

The `career_dataset.csv` contains the following columns:
- **Sr.No.**: Serial number
- **Course**: Academic course/field of study
- **Job profession**: Target variable (career to predict)
- **Student**: Student identifier
- **Intelligence Scores** (Numerical, 1-20):
  - `Linguistic`: Language and verbal skills
  - `Musical`: Musical and rhythmic abilities  
  - `Bodily`: Physical and kinesthetic intelligence
  - `Logical - Mathematical`: Logic and mathematical reasoning
  - `Spatial-Visualization`: Spatial and visual processing
  - `Interpersonal`: Social and people skills
  - `Intrapersonal`: Self-awareness and introspection
  - `Naturalist`: Nature and environmental awareness
- **Performance Ratings** (Categorical: POOR, AVG, GOOD, BEST):
  - `P1` to `P8`: Various performance indicators
- **s/p**: Student performance category

## API Usage Examples

### Single Prediction
```json
POST /predict
{
  "user_profile": {
    "Course": "Computer Science",
    "Linguistic": 15,
    "Musical": 8,
    "Bodily": 12,
    "Logical_Mathematical": 18,
    "Spatial_Visualization": 16,
    "Interpersonal": 14,
    "Intrapersonal": 15,
    "Naturalist": 10,
    "sp": "s1",
    "P1": "GOOD",
    "P2": "AVG",
    "P3": "BEST",
    "P4": "BEST",
    "P5": "GOOD",
    "P6": "AVG",
    "P7": "GOOD",
    "P8": "AVG",
    "skills": ["Python", "Machine Learning"],
    "interests": ["Technology", "Data Science"]
  },
  "model_type": "rf",
  "include_llm_guidance": true
}
```

### Response Format
```json
{
  "ml_prediction": {
    "primary_prediction": "Data Scientist",
    "confidence": 0.87,
    "top_predictions": [
      {
        "career": "Data Scientist",
        "probability": 0.87,
        "confidence": "High"
      },
      {
        "career": "Software Engineer", 
        "probability": 0.72,
        "confidence": "High"
      },
      {
        "career": "Machine Learning Engineer",
        "probability": 0.65,
        "confidence": "Medium"
      }
    ],
    "model_used": "RF"
  },
  "llm_guidance": {
    "career_path": "Detailed career guidance...",
    "skills_to_develop": ["Advanced ML", "Deep Learning"],
    "certifications": ["AWS ML Specialty", "Google ML Engineer"],
    "industry_insights": "Growing field with high demand...",
    "salary_range": "$80,000 - $150,000 annually",
    "growth_opportunities": "Senior DS, ML Architect, Head of AI..."
  }
}
```

## Model Features

### ML Models
- **RandomForest**: Ensemble method for robust predictions
- **XGBoost**: Gradient boosting for high performance
- **Automated Model Selection**: Choose best performing model

### LLM Integration
- **Career Guidance**: Detailed career path recommendations
- **Skill Analysis**: Gap analysis and learning recommendations
- **Industry Insights**: Market trends and salary expectations
- **Personalized Advice**: Tailored guidance based on user profile

## Development

### Running Tests
```bash
# Run individual components
python src/preprocess.py
python src/train.py
python src/inference.py
```

### API Testing
Use tools like Postman or curl:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"user_profile": {"age": 25, "skills": ["Python"]}}'
```

## Deployment

### Local Development
```bash
uvicorn src.api:app --reload --port 8000
```

### Production Deployment
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
```

## Configuration

### Model Parameters
Adjust model parameters in `train.py`:
- RandomForest: `n_estimators`, `max_depth`, etc.
- XGBoost: `learning_rate`, `max_depth`, etc.

### API Configuration
Modify CORS settings and other API configurations in `api.py`.

## Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## License
This project is licensed under the MIT License.

## Support
For issues and questions, please open an issue in the repository.
