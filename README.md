# Career Prediction Hybrid Model (ML + LLM/Gemini)

>This project is a hybrid system that combines traditional Machine Learning (ML) models with a Large Language Model (LLM, now Gemini) to predict and guide users toward suitable career paths. It leverages both structured data and generative AI for comprehensive, personalized career recommendations.

---

## Features
- **Hybrid ML + LLM**: Combines RandomForest/XGBoost predictions with Gemini (LLM) for rich, actionable guidance
- **API-Ready**: FastAPI backend for easy integration
- **Automated Preprocessing**: Robust data cleaning and feature engineering
- **Batch & Single Prediction**: Supports both single and multiple user predictions
- **LLM Guidance**: Skill gap analysis, career path suggestions, industry insights

## Project Structure
```
career_ai/
├─ data/
│  └─ career_dataset.csv            # Raw dataset
├─ models/
│  ├─ preprocessor.pkl              # Preprocessing pipeline
│  ├─ label_encoder.pkl             # Label encoder for job roles (It will appear after you Train the Model)
│  ├─ model_rf.pkl                  # Trained RandomForest model (It will appear after you Train the Model)
│  └─ model_xgb.pkl                 # Trained XGBoost model (It will appear after you Train the Model)
├─ src/
│  ├─ preprocess.py                 # Data preprocessing
│  ├─ train.py                      # Model training
│  ├─ inference.py                  # ML+LLM inference
│  ├─ llm_enricher.py               # LLM (Gemini) integration
│  └─ api.py                        # FastAPI backend
├─ requirements.txt                 # Python dependencies
└─ README.md
```

---

## Step-by-Step Setup & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/Sattwik999/Career-Prediction-MLModel.git
cd Career-Prediction-MLModel
```

### 2. Create a Python Environment
```bash
# Using venv
python -m venv venv
venv\Scripts\activate  # On Windows
# Or
source venv/bin/activate  # On Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file or set these variables:
```
GEMINI_API_KEY=your_gemini_api_key
# or for OpenAI
OPENAI_API_KEY=your_openai_api_key
```

### 5. Prepare the Dataset
- Place your dataset as `career_ai/data/career_dataset.csv`
- Ensure it has the required columns (see below)

### 6. Run Preprocessing
```bash
cd career_ai/src
python preprocess.py
```

### 7. Train the Model
```bash
python train.py
```

### 8. Run Inference (ML + LLM)
```bash
python inference.py
```

### 9. Start the API Server
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

---

## Dataset Overview
The `career_dataset.csv` should include:
- **Sr.No.**: Serial number
- **Course**: Academic field
- **Job profession**: Target variable (career)
- **Student**: Identifier
- **Intelligence Scores** (Numerical, 1-20):
	- Linguistic, Musical, Bodily, Logical-Mathematical, Spatial-Visualization, Interpersonal, Intrapersonal, Naturalist
- **Performance Ratings** (POOR, AVG, GOOD, BEST):
	- P1 to P8
- **s/p**: Student performance category

---

## API Endpoints
- `GET /` — Health check
- `POST /predict` — ML + LLM prediction
- `POST /predict/simple` — ML only
- `POST /predict/batch` — Batch predictions
- `GET /careers` — List careers
- `GET /skills` — Skill recommendations
- `POST /feedback` — User feedback

---

## Example Usage

### Single Prediction Request (Note this is the Format of Input that is needed to pushed to this)
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

### Response Example
```json
{
	"ml_prediction": {
		"primary_prediction": "Data Scientist",
		"confidence": 0.87,
		"top_predictions": [
			{"career": "Data Scientist", "probability": 0.87, "confidence": "High"},
			{"career": "Software Engineer", "probability": 0.72, "confidence": "High"},
			{"career": "Machine Learning Engineer", "probability": 0.65, "confidence": "Medium"}
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

---

## Training & Customization
- Edit `src/train.py` to adjust model parameters (RandomForest, XGBoost)
- Add new features or preprocessing steps in `src/preprocess.py`
- Update LLM prompts or integration in `src/llm_enricher.py`
---

## Expose API Publicly with ngrok

You can make your local API accessible over the internet using ngrok. This is useful for sharing your running app for demos or testing.

### 1. Install ngrok
- Download from https://ngrok.com/download
- Or install via command line:
	- On Windows (using Chocolatey):
		```bash
		choco install ngrok
		```
	- On Linux/Mac:
		```bash
		brew install ngrok/ngrok/ngrok
		# or
		sudo snap install ngrok
		```

### 2. Start Your API Locally
```bash
uvicorn career_ai.src.api:app --host 0.0.0.0 --port 8000
# or if using Docker:
docker run --rm -p 8000:8000 career-hybrid-ai
```

### 3. Expose with ngrok
```bash
ngrok http 8000
```

ngrok will provide a public URL (like https://xxxx.ngrok.io) that you can share for external access to your API.
---

## Notes
- Large model files are ignored by git (see `.gitignore`). Retrain or download as needed.
- For API testing, use Postman or curl.

---

## License
MIT License

---


---

## Docker Usage

You can run this project easily using Docker. This is useful for deployment or to avoid local environment issues.

### 1. Build the Docker Image
```bash
docker build -t career-hybrid-ai .
```

### 2. Run the Docker Container
```bash
docker run --rm -p 8000:8000 career-hybrid-ai
```

The API will be available at http://localhost:8000

### 3. Environment Variables
To use Gemini or OpenAI, pass your API keys as environment variables:
```bash
docker run --rm -p 8000:8000 -e GEMINI_API_KEY=your_gemini_api_key career-hybrid-ai
# or
docker run --rm -p 8000:8000 -e OPENAI_API_KEY=your_openai_api_key career-hybrid-ai
```

---

Created by Sattwik Sarkar
