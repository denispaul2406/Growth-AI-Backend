import pandas as pd
import io
from datetime import datetime
from typing import Dict, List, Tuple
import re

def clean_currency(value) -> float:
    """Remove currency symbols and convert to float"""
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    # Remove currency symbols, commas, and 'INR' prefix
    cleaned = str(value).replace('₹', '').replace('$', '').replace('INR', '').replace(',', '').strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0

def parse_date(date_str) -> str:
    """Parse various date formats and return YYYY-MM-DD"""
    if pd.isna(date_str):
        return None
    
    date_str = str(date_str).strip()
    
    # Try various formats
    formats = [
        '%Y-%m-%d',
        '%d-%b-%Y',  # 15-Jan-2024
        '%d/%m/%Y',
        '%m/%d/%Y',
        '%Y/%m/%d',
        '%d-%m-%Y',
        '%m-%d-%Y'
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            continue
    
    return None

def normalize_csv(file_content: bytes) -> Dict:
    """Normalize CSV with intelligent cleaning"""
    try:
        # Try reading CSV
        df = pd.read_csv(io.BytesIO(file_content))
    except Exception as e:
        return {
            'success': False,
            'error': f'Failed to parse CSV: {str(e)}',
            'cleaned_rows': 0,
            'dropped_rows': 0,
            'duplicates_merged': 0,
            'warnings': [],
            'data': []
        }
    
    warnings = []
    initial_rows = len(df)
    
    # Step 1: Clean headers
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Step 2: Detect platform from columns or data
    platform = 'unknown'
    if 'platform' in df.columns:
        platform = df['platform'].iloc[0] if len(df) > 0 else 'unknown'
    else:
        # Auto-detect from column names
        if 'ad_set' in df.columns or 'adset' in df.columns:
            platform = 'meta'
            df['platform'] = 'meta'
        elif 'ad_group' in df.columns or 'adgroup' in df.columns:
            platform = 'google'
            df['platform'] = 'google'
        else:
            df['platform'] = 'unknown'
    
    # Step 3: Normalize column names to standard schema
    column_mapping = {
        'campaign': 'campaign_name',
        'campaign_id': 'campaign_name',
        'ad_set': 'campaign_name',
        'adset': 'campaign_name',
        'ad_group': 'campaign_name',
        'adgroup': 'campaign_name',
        'cost': 'spend',
        'amount_spent': 'spend',
        'impressions': 'impressions',
        'impr': 'impressions',
        'clicks': 'clicks',
        'link_clicks': 'clicks',
        'conversions': 'conversions',
        'conv': 'conversions',
        'purchases': 'conversions',
        'revenue': 'revenue',
        'conversion_value': 'revenue',
        'date': 'date',
        'day': 'date',
        'date_start': 'date'
    }
    
    df.rename(columns=column_mapping, inplace=True)
    
    # Step 4: Ensure required columns exist
    required_cols = ['date', 'campaign_name', 'spend']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return {
            'success': False,
            'error': f'Missing required columns: {", ".join(missing_cols)}',
            'cleaned_rows': 0,
            'dropped_rows': initial_rows,
            'duplicates_merged': 0,
            'warnings': warnings,
            'data': []
        }
    
    # Step 5: Fill missing optional columns with defaults
    if 'impressions' not in df.columns:
        df['impressions'] = 0
    if 'clicks' not in df.columns:
        df['clicks'] = 0
    if 'conversions' not in df.columns:
        df['conversions'] = 0
    if 'revenue' not in df.columns:
        df['revenue'] = 0.0
    
    # Step 6: Clean currency fields
    df['spend'] = df['spend'].apply(clean_currency)
    df['revenue'] = df['revenue'].apply(clean_currency)
    
    # Step 7: Clean numeric fields
    df['impressions'] = pd.to_numeric(df['impressions'], errors='coerce').fillna(0).astype(int)
    df['clicks'] = pd.to_numeric(df['clicks'], errors='coerce').fillna(0).astype(int)
    df['conversions'] = pd.to_numeric(df['conversions'], errors='coerce').fillna(0).astype(int)
    
    # Step 8: Parse dates
    df['date'] = df['date'].apply(parse_date)
    
    # Step 9: Drop rows with missing critical data
    before_drop = len(df)
    df = df.dropna(subset=['date', 'campaign_name'])
    df = df[df['spend'] > 0]  # Drop zero spend rows
    dropped = before_drop - len(df)
    
    if dropped > 0:
        warnings.append(f'{dropped} rows dropped (missing date/campaign or zero spend)')
    
    # Step 10: Deduplicate by composite key (date + campaign + platform)
    before_dedup = len(df)
    df = df.groupby(['date', 'campaign_name', 'platform'], as_index=False).agg({
        'spend': 'sum',
        'impressions': 'sum',
        'clicks': 'sum',
        'conversions': 'sum',
        'revenue': 'sum'
    })
    duplicates_merged = before_dedup - len(df)
    
    if duplicates_merged > 0:
        warnings.append(f'{duplicates_merged} duplicate rows merged')
    
    # Step 11: Calculate derived metrics
    df['ctr'] = (df['clicks'] / df['impressions'] * 100).round(2)
    df['ctr'] = df['ctr'].replace([float('inf'), -float('inf')], 0)
    
    df['cpa'] = (df['spend'] / df['conversions']).round(2)
    df['cpa'] = df['cpa'].replace([float('inf'), -float('inf')], 0)
    
    df['roas'] = (df['revenue'] / df['spend']).round(2)
    df['roas'] = df['roas'].replace([float('inf'), -float('inf')], 0)
    
    # Convert to list of dicts and ensure JSON serializable types
    data_records = []
    for _, row in df.iterrows():
        record = {}
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                record[col] = 0 if col in ['impressions', 'clicks', 'conversions'] else 0.0
            elif isinstance(val, (int, bool)):
                record[col] = int(val)
            elif isinstance(val, float):
                record[col] = float(val)
            else:
                record[col] = str(val)
        data_records.append(record)
    
    return {
        'success': True,
        'cleaned_rows': len(df),
        'dropped_rows': initial_rows - len(df),
        'duplicates_merged': duplicates_merged,
        'warnings': warnings,
        'data': data_records
    }
