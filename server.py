from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from typing import List
import json

# Import our modules
from models import (
    Recommendation, RecommendationCreate, Feedback, FeedbackCreate,
    Benchmark, NormalizationResult, SimulationResult
)
from normalization import normalize_csv
from rules import generate_recommendations, find_relevant_benchmarks
from simulation import simulate_counterfactual

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Load benchmarks from JSON file
BENCHMARKS_FILE = ROOT_DIR / 'data' / 'benchmarks.json'
with open(BENCHMARKS_FILE, 'r') as f:
    BENCHMARKS_DATA = json.load(f)

# API Routes
@api_router.get("/")
async def root():
    return {"message": "GrowthAI API - D2C Ad Spend Optimization"}

@api_router.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload and normalize CSV file"""
    try:
        # Read file content
        content = await file.read()
        
        # Normalize the CSV
        result = normalize_csv(content)
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result.get('error', 'Failed to process CSV'))
        
        # Store campaigns in database
        if result['data']:
            # Clear existing campaigns for demo purposes
            await db.campaigns.delete_many({})
            
            # Make a copy to insert (MongoDB will add _id)
            docs_to_insert = [dict(doc) for doc in result['data']]
            
            # Insert new campaigns
            _ = await db.campaigns.insert_many(docs_to_insert)
        
        # Return only what we need (not the full MongoDB response)
        return JSONResponse(content={
            'success': True,
            'cleaned_rows': result['cleaned_rows'],
            'dropped_rows': result['dropped_rows'],
            'duplicates_merged': result['duplicates_merged'],
            'warnings': result['warnings'],
            'preview': result['data'][:20]  # First 20 rows
        })
    
    except Exception as e:
        logging.error(f"Error processing CSV: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

@api_router.get("/campaigns")
async def get_campaigns():
    """Get all stored campaigns"""
    campaigns = await db.campaigns.find({}, {"_id": 0}).to_list(10000)
    return campaigns

@api_router.post("/analyze")
async def analyze_campaigns():
    """Generate recommendations based on uploaded campaigns"""
    try:
        # Get all campaigns
        campaigns_data = await db.campaigns.find({}, {"_id": 0}).to_list(10000)
        
        if not campaigns_data:
            raise HTTPException(status_code=400, detail="No campaign data found. Please upload a CSV first.")
        
        # Generate recommendations
        recommendations = generate_recommendations(campaigns_data, BENCHMARKS_DATA)
        
        # Clear old recommendations
        await db.recommendations.delete_many({})
        
        # Store recommendations
        if recommendations:
            recs_to_store = []
            for rec in recommendations:
                rec_obj = Recommendation(**rec)
                rec_dict = rec_obj.model_dump()
                rec_dict['created_at'] = rec_dict['created_at'].isoformat()
                recs_to_store.append(rec_dict)
            
            await db.recommendations.insert_many(recs_to_store)
        
        return {
            'success': True,
            'recommendations_count': len(recommendations),
            'recommendations': recommendations
        }
    
    except Exception as e:
        logging.error(f"Error analyzing campaigns: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing campaigns: {str(e)}")

@api_router.get("/recommendations")
async def get_recommendations():
    """Get all recommendations"""
    recommendations = await db.recommendations.find({}, {"_id": 0}).to_list(1000)
    return recommendations

@api_router.get("/benchmarks")
async def get_benchmarks(platform: str = None, metric_type: str = None):
    """Get benchmarks, optionally filtered by platform or metric type"""
    benchmarks = BENCHMARKS_DATA
    
    if platform:
        benchmarks = [b for b in benchmarks if b.get('platform') == platform or b.get('platform') == 'all']
    
    if metric_type:
        benchmarks = [b for b in benchmarks if b.get('metric_type') == metric_type or b.get('metric_type') == 'general']
    
    return benchmarks

@api_router.get("/benchmarks/{benchmark_id}")
async def get_benchmark_by_id(benchmark_id: str):
    """Get a specific benchmark by ID"""
    benchmark = next((b for b in BENCHMARKS_DATA if b['id'] == benchmark_id), None)
    
    if not benchmark:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    
    return benchmark

@api_router.post("/feedback", response_model=Feedback)
async def submit_feedback(feedback: FeedbackCreate):
    """Submit feedback on a recommendation"""
    try:
        feedback_obj = Feedback(**feedback.model_dump())
        feedback_dict = feedback_obj.model_dump()
        feedback_dict['created_at'] = feedback_dict['created_at'].isoformat()
        
        await db.feedback.insert_one(feedback_dict)
        
        return feedback_obj
    
    except Exception as e:
        logging.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error submitting feedback: {str(e)}")

@api_router.get("/feedback")
async def get_all_feedback():
    """Get all feedback"""
    feedback_list = await db.feedback.find({}, {"_id": 0}).to_list(1000)
    return feedback_list

@api_router.get("/evaluation/metrics")
async def get_evaluation_metrics():
    """Calculate precision metrics from feedback"""
    try:
        feedback_list = await db.feedback.find({}, {"_id": 0}).to_list(1000)
        
        if not feedback_list:
            return {
                'overall_precision': 0,
                'total_feedback': 0,
                'useful_count': 0,
                'by_type': {}
            }
        
        total = len(feedback_list)
        useful = len([f for f in feedback_list if f['is_useful']])
        overall_precision = useful / total if total > 0 else 0
        
        # Calculate by type
        by_type = {}
        for rec_type in ['fatigue', 'reallocation']:
            type_feedback = [f for f in feedback_list if f['recommendation_type'] == rec_type]
            type_total = len(type_feedback)
            type_useful = len([f for f in type_feedback if f['is_useful']])
            type_precision = type_useful / type_total if type_total > 0 else 0
            
            by_type[rec_type] = {
                'precision': round(type_precision, 2),
                'useful': type_useful,
                'total': type_total
            }
        
        return {
            'overall_precision': round(overall_precision, 2),
            'total_feedback': total,
            'useful_count': useful,
            'by_type': by_type
        }
    
    except Exception as e:
        logging.error(f"Error calculating metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating metrics: {str(e)}")

@api_router.post("/simulate")
async def simulate_impact(campaign_name: str, action: str):
    """Simulate counterfactual impact of an action"""
    try:
        # Get campaign history
        campaign_history = await db.campaigns.find(
            {"campaign_name": campaign_name},
            {"_id": 0}
        ).sort("date", 1).to_list(1000)
        
        if not campaign_history:
            raise HTTPException(status_code=404, detail=f"Campaign '{campaign_name}' not found")
        
        # Get action parameters from recommendation if applicable
        action_params = {}
        if action == 'reallocate_budget':
            # Find related recommendation
            rec = await db.recommendations.find_one(
                {"details.from_campaign": campaign_name},
                {"_id": 0}
            )
            if rec and rec.get('details'):
                action_params['target_roas'] = rec['details'].get('to_roas', 0)
        
        # Run simulation
        result = simulate_counterfactual(campaign_history, action, action_params)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        result['campaign_name'] = campaign_name
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error simulating impact: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error simulating impact: {str(e)}")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()