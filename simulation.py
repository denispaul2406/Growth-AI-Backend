import random
import numpy as np
from typing import Dict, List, Any

def simulate_counterfactual(campaign_history: List[Dict], action: str, action_params: Dict) -> Dict:
    """Bootstrap simulation for counterfactual impact"""
    
    # Extract last 28 days of metrics
    if len(campaign_history) < 7:
        return {
            'error': 'Insufficient data for simulation (need at least 7 days)'
        }
    
    recent_data = campaign_history[-28:] if len(campaign_history) >= 28 else campaign_history
    
    # Extract baseline metrics
    daily_roas = [day['roas'] for day in recent_data if day['roas'] > 0]
    daily_spend = [day['spend'] for day in recent_data if day['spend'] > 0]
    daily_cpa = [day['cpa'] for day in recent_data if day['cpa'] > 0]
    
    if not daily_roas or not daily_spend:
        return {
            'error': 'Insufficient non-zero metrics for simulation'
        }
    
    # Define uplift ranges based on action type
    if action == 'refresh_creative':
        # Creative refresh typically improves CTR, reduces CPA
        cpa_reduction_range = (0.15, 0.25)  # 15-25% reduction
        roas_uplift_range = (1.10, 1.20)  # 10-20% improvement
    elif action == 'reallocate_budget':
        # Budget reallocation to higher ROAS campaign
        target_roas = action_params.get('target_roas', np.mean(daily_roas))
        current_roas = np.mean(daily_roas)
        expected_improvement = target_roas / current_roas if current_roas > 0 else 1.05
        roas_uplift_range = (expected_improvement * 0.9, expected_improvement * 1.1)
        cpa_reduction_range = (0.0, 0.05)  # Minimal CPA impact
    else:
        roas_uplift_range = (1.03, 1.08)
        cpa_reduction_range = (0.05, 0.15)
    
    # Bootstrap simulation
    n_iterations = 1000
    simulated_roas = []
    simulated_cpa = []
    simulated_revenue_lift = []
    
    avg_daily_spend = np.mean(daily_spend)
    avg_current_roas = np.mean(daily_roas)
    avg_current_cpa = np.mean(daily_cpa) if daily_cpa else 0
    
    for _ in range(n_iterations):
        # Resample with replacement
        sample_indices = np.random.choice(len(daily_roas), size=len(daily_roas), replace=True)
        sample_roas = [daily_roas[i] for i in sample_indices]
        
        # Apply random uplift from range
        uplift = random.uniform(*roas_uplift_range)
        simulated_sample_roas = [r * uplift for r in sample_roas]
        avg_sim_roas = np.mean(simulated_sample_roas)
        
        simulated_roas.append(avg_sim_roas)
        
        # CPA simulation
        if daily_cpa and avg_current_cpa > 0:
            cpa_reduction = random.uniform(*cpa_reduction_range)
            simulated_sample_cpa = avg_current_cpa * (1 - cpa_reduction)
            simulated_cpa.append(simulated_sample_cpa)
        
        # Revenue lift
        revenue_lift = avg_daily_spend * (avg_sim_roas - avg_current_roas)
        simulated_revenue_lift.append(revenue_lift)
    
    # Calculate percentiles
    roas_median = np.median(simulated_roas)
    roas_p5 = np.percentile(simulated_roas, 5)
    roas_p95 = np.percentile(simulated_roas, 95)
    
    revenue_median = np.median(simulated_revenue_lift)
    revenue_p5 = np.percentile(simulated_revenue_lift, 5)
    revenue_p95 = np.percentile(simulated_revenue_lift, 95)
    
    result = {
        'action': action,
        'current_metrics': {
            'avg_daily_spend': round(avg_daily_spend, 2),
            'avg_roas': round(avg_current_roas, 2),
            'avg_cpa': round(avg_current_cpa, 2) if avg_current_cpa > 0 else 'N/A'
        },
        'projected_metrics': {
            'roas': {
                'median': round(roas_median, 2),
                'p5': round(roas_p5, 2),
                'p95': round(roas_p95, 2)
            },
            'daily_revenue_lift': {
                'median': round(revenue_median, 2),
                'p5': round(revenue_p5, 2),
                'p95': round(revenue_p95, 2)
            }
        },
        'confidence_interval': f"₹{round(revenue_p5, 0)} to ₹{round(revenue_p95, 0)} additional daily revenue (90% CI)",
        'impact_summary': f"Expected daily revenue lift: ₹{round(revenue_median, 0)} (ROAS: {round(avg_current_roas, 2)} → {round(roas_median, 2)})"
    }
    
    if simulated_cpa:
        cpa_median = np.median(simulated_cpa)
        result['projected_metrics']['cpa'] = {
            'median': round(cpa_median, 2),
            'reduction_pct': round((avg_current_cpa - cpa_median) / avg_current_cpa * 100, 1) if avg_current_cpa > 0 else 0
        }
    
    return result
