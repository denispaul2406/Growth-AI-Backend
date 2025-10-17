from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

# Campaign Models
class CampaignRow(BaseModel):
    date: str
    campaign_name: str
    platform: str  # 'meta' or 'google'
    spend: float
    impressions: int
    clicks: int
    conversions: int
    revenue: float
    ctr: Optional[float] = None  # Click-through rate
    cpa: Optional[float] = None  # Cost per acquisition
    roas: Optional[float] = None  # Return on ad spend

class NormalizationResult(BaseModel):
    cleaned_rows: int
    dropped_rows: int
    duplicates_merged: int
    warnings: List[str]
    data: List[CampaignRow]

# Recommendation Models
class FatigueDetection(BaseModel):
    campaign_name: str
    platform: str
    ctr_drop_pct: float
    ctr_last_7d: float
    ctr_prev_7d: float
    cpa_rise_pct: float
    cpa_last_7d: float
    cpa_prev_7d: float
    confidence: float
    action: str = "Pause or refresh creative"

class ReallocationPlan(BaseModel):
    from_campaign: str
    to_campaign: str
    current_daily_budget: float
    suggested_shift_amount: float
    suggested_shift_pct: float
    from_roas: float
    to_roas: float
    confidence: float
    action: str

class Recommendation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str  # 'fatigue' or 'reallocation'
    title: str
    description: str
    why_fired: str
    trigger_metrics: Dict[str, Any]
    thresholds: Dict[str, Any]
    projected_impact: str
    confidence: float
    source_ids: List[str]  # IDs from RAG benchmarks
    details: Dict[str, Any]  # FatigueDetection or ReallocationPlan as dict
    created_at: datetime = Field(default_factory=datetime.utcnow)

class RecommendationCreate(BaseModel):
    type: str
    title: str
    description: str
    why_fired: str
    trigger_metrics: Dict[str, Any]
    thresholds: Dict[str, Any]
    projected_impact: str
    confidence: float
    source_ids: List[str]
    details: Dict[str, Any]

# Benchmark Models
class Benchmark(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    source: str
    source_url: str
    platform: Optional[str] = None  # 'meta', 'google', 'all'
    vertical: Optional[str] = None  # 'fashion', 'beauty', 'home', 'all'
    metric_type: str  # 'ctr', 'cpa', 'roas', 'fatigue', 'general'
    key_finding: str
    notes: str
    year: int

# Feedback Models
class Feedback(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    recommendation_id: str
    recommendation_type: str
    is_useful: bool
    created_at: datetime = Field(default_factory=datetime.utcnow)

class FeedbackCreate(BaseModel):
    recommendation_id: str
    recommendation_type: str
    is_useful: bool

# Simulation Models
class SimulationResult(BaseModel):
    action: str
    campaign_name: str
    current_metrics: Dict[str, float]
    projected_metrics: Dict[str, Any]  # Contains median, p5, p95
    confidence_interval: str
    impact_summary: str
