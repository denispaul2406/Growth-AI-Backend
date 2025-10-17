from typing import List, Dict, Any
import pandas as pd
from datetime import datetime, timedelta
import statistics

def detect_creative_fatigue(campaigns_data: List[Dict]) -> List[Dict]:
    """Detect creative fatigue using WoW CTR drop and CPA rise"""
    df = pd.DataFrame(campaigns_data)
    df['date'] = pd.to_datetime(df['date'])
    
    fatigue_alerts = []
    
    # Group by campaign
    for campaign_name in df['campaign_name'].unique():
        campaign_df = df[df['campaign_name'] == campaign_name].sort_values('date')
        
        if len(campaign_df) < 14:
            continue  # Need at least 14 days
        
        # Get last 14 days
        last_14_days = campaign_df.tail(14)
        
        # Split into two weeks
        last_7_days = last_14_days.tail(7)
        prev_7_days = last_14_days.head(7)
        
        # Calculate averages
        avg_ctr_last = last_7_days['ctr'].mean()
        avg_ctr_prev = prev_7_days['ctr'].mean()
        avg_cpa_last = last_7_days['cpa'].mean()
        avg_cpa_prev = prev_7_days['cpa'].mean()
        
        if avg_ctr_prev == 0 or avg_cpa_prev == 0:
            continue
        
        # Calculate percentage changes
        ctr_drop_pct = ((avg_ctr_prev - avg_ctr_last) / avg_ctr_prev * 100)
        cpa_rise_pct = ((avg_cpa_last - avg_cpa_prev) / avg_cpa_prev * 100)
        
        # Thresholds: CTR drop >= 20% AND CPA rise >= 15%
        if ctr_drop_pct >= 20 and cpa_rise_pct >= 15:
            platform = campaign_df['platform'].iloc[0]
            
            # Calculate confidence based on data quality
            confidence = min(0.95, 0.70 + (len(campaign_df) / 100))
            
            fatigue_alerts.append({
                'campaign_name': campaign_name,
                'platform': platform,
                'ctr_drop_pct': round(ctr_drop_pct, 1),
                'ctr_last_7d': round(avg_ctr_last, 2),
                'ctr_prev_7d': round(avg_ctr_prev, 2),
                'cpa_rise_pct': round(cpa_rise_pct, 1),
                'cpa_last_7d': round(avg_cpa_last, 2),
                'cpa_prev_7d': round(avg_cpa_prev, 2),
                'confidence': round(confidence, 2),
                'action': 'Pause or refresh creative'
            })
    
    return fatigue_alerts

def detect_budget_reallocation(campaigns_data: List[Dict]) -> List[Dict]:
    """Suggest budget reallocation from low ROAS to high ROAS campaigns"""
    df = pd.DataFrame(campaigns_data)
    
    # Calculate average ROAS per campaign over last 14 days
    df['date'] = pd.to_datetime(df['date'])
    recent_df = df.sort_values('date').groupby('campaign_name').tail(14)
    
    campaign_stats = recent_df.groupby('campaign_name').agg({
        'roas': 'mean',
        'spend': 'mean',  # Average daily spend
        'platform': 'first'
    }).reset_index()
    
    campaign_stats = campaign_stats[campaign_stats['spend'] > 0]
    
    if len(campaign_stats) < 2:
        return []  # Need at least 2 campaigns
    
    # Calculate quartiles
    roas_q1 = campaign_stats['roas'].quantile(0.25)
    roas_q3 = campaign_stats['roas'].quantile(0.75)
    spend_q3 = campaign_stats['spend'].quantile(0.75)
    
    low_performers = campaign_stats[campaign_stats['roas'] <= roas_q1]
    high_performers = campaign_stats[
        (campaign_stats['roas'] >= roas_q3) & 
        (campaign_stats['spend'] < spend_q3)  # Has headroom
    ]
    
    reallocation_plans = []
    
    for _, low_camp in low_performers.iterrows():
        for _, high_camp in high_performers.iterrows():
            # Don't reallocate within the same campaign
            if low_camp['campaign_name'] == high_camp['campaign_name']:
                continue
            
            # Suggest 15% shift
            shift_pct = 15
            shift_amount = low_camp['spend'] * (shift_pct / 100)
            
            confidence = 0.75 if high_camp['roas'] > low_camp['roas'] * 1.5 else 0.65
            
            reallocation_plans.append({
                'from_campaign': low_camp['campaign_name'],
                'to_campaign': high_camp['campaign_name'],
                'current_daily_budget': round(low_camp['spend'], 2),
                'suggested_shift_amount': round(shift_amount, 2),
                'suggested_shift_pct': shift_pct,
                'from_roas': round(low_camp['roas'], 2),
                'to_roas': round(high_camp['roas'], 2),
                'confidence': round(confidence, 2),
                'action': f'Shift ₹{round(shift_amount, 0)}/day from low ROAS to high ROAS campaign'
            })
    
    return reallocation_plans[:5]  # Return top 5

def generate_recommendations(campaigns_data: List[Dict], benchmarks: List[Dict]) -> List[Dict]:
    """Generate all recommendations with benchmark linkage"""
    recommendations = []
    
    # Detect fatigue
    fatigue_alerts = detect_creative_fatigue(campaigns_data)
    
    for alert in fatigue_alerts:
        # Find relevant benchmarks
        source_ids = find_relevant_benchmarks(
            benchmarks,
            alert['platform'],
            'fatigue'
        )
        
        rec = {
            'type': 'fatigue',
            'title': f"Creative Fatigue Detected: {alert['campaign_name']}",
            'description': f"CTR dropped {alert['ctr_drop_pct']}% and CPA rose {alert['cpa_rise_pct']}% week-over-week.",
            'why_fired': f"This recommendation fired because:\n• CTR dropped {alert['ctr_drop_pct']}% week-over-week (threshold: ≥20%)\n• CPA rose {alert['cpa_rise_pct']}% week-over-week (threshold: ≥15%)\n• Both signals indicate audience fatigue with current creative",
            'trigger_metrics': {
                'ctr_last_7d': alert['ctr_last_7d'],
                'ctr_prev_7d': alert['ctr_prev_7d'],
                'cpa_last_7d': alert['cpa_last_7d'],
                'cpa_prev_7d': alert['cpa_prev_7d']
            },
            'thresholds': {
                'ctr_drop_threshold': 20,
                'cpa_rise_threshold': 15
            },
            'projected_impact': f"Refreshing creative could reduce CPA by 15-25% (₹{round(alert['cpa_last_7d'] * 0.15, 0)}-₹{round(alert['cpa_last_7d'] * 0.25, 0)})",
            'confidence': alert['confidence'],
            'source_ids': source_ids[:3],
            'details': alert
        }
        recommendations.append(rec)
    
    # Detect reallocation opportunities
    realloc_plans = detect_budget_reallocation(campaigns_data)
    
    for plan in realloc_plans:
        source_ids = find_relevant_benchmarks(
            benchmarks,
            'all',
            'roas'
        )
        
        roas_improvement = plan['to_roas'] - plan['from_roas']
        projected_revenue = plan['suggested_shift_amount'] * plan['to_roas']
        
        rec = {
            'type': 'reallocation',
            'title': f"Budget Reallocation Opportunity",
            'description': f"Shift ₹{plan['suggested_shift_amount']}/day from '{plan['from_campaign']}' to '{plan['to_campaign']}'",
            'why_fired': f"This recommendation fired because:\n• '{plan['from_campaign']}' has ROAS of {plan['from_roas']} (bottom quartile)\n• '{plan['to_campaign']}' has ROAS of {plan['to_roas']} (top quartile)\n• ROAS difference: {round(roas_improvement, 2)}x\n• Target campaign has budget headroom for scaling",
            'trigger_metrics': {
                'from_roas': plan['from_roas'],
                'to_roas': plan['to_roas'],
                'roas_improvement': round(roas_improvement, 2)
            },
            'thresholds': {
                'low_roas_threshold': 'bottom_quartile',
                'high_roas_threshold': 'top_quartile'
            },
            'projected_impact': f"Estimated additional revenue: ₹{round(projected_revenue, 0)}/day (confidence: {int(plan['confidence']*100)}%)",
            'confidence': plan['confidence'],
            'source_ids': source_ids[:3],
            'details': plan
        }
        recommendations.append(rec)
    
    return recommendations

def find_relevant_benchmarks(benchmarks: List[Dict], platform: str, metric_type: str, top_k: int = 3) -> List[str]:
    """Find most relevant benchmark IDs using keyword matching"""
    scored_benchmarks = []
    
    for bench in benchmarks:
        score = 0
        
        # Platform match
        if bench.get('platform') == platform or bench.get('platform') == 'all':
            score += 10
        
        # Metric type match
        if bench.get('metric_type') == metric_type or bench.get('metric_type') == 'general':
            score += 8
        
        # Keyword match in title/notes
        keywords = [metric_type, platform, 'optimization', 'performance']
        text = f"{bench.get('title', '')} {bench.get('notes', '')}".lower()
        
        for keyword in keywords:
            if keyword.lower() in text:
                score += 2
        
        scored_benchmarks.append((score, bench['id']))
    
    # Sort by score and return top k IDs
    scored_benchmarks.sort(reverse=True, key=lambda x: x[0])
    return [bench_id for score, bench_id in scored_benchmarks[:top_k]]
