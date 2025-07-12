# tools/logger.py

import json
from config.settings import Settings

REDACTED_FIELDS = set(Settings.REDACTED_FIELDS)

def redact_phis(data: dict) -> dict:
    """
    Recursively redacts PHI fields based on settings.
    """
    if isinstance(data, dict):
        return {
            key: ("[REDACTED]" if key in REDACTED_FIELDS else redact_phis(value))
            for key, value in data.items()
        }
    elif isinstance(data, list):
        return [redact_phis(item) for item in data]
    return data

def clean_text(text):
    """Remove problematic Unicode characters"""
    if not isinstance(text, str):
        return text
    # Replace problematic characters with ASCII equivalents
    replacements = {
        '\u2013': '-',  # en dash
        '\u2014': '--', # em dash
        '\u2018': "'",  # left single quote
        '\u2019': "'",  # right single quote
        '\u201C': '"',  # left double quote
        '\u201D': '"',  # right double quote
        '\u2026': '...', # ellipsis
        '\u2713': 'v',  # check mark
        '\u26a0': '!',  # warning sign
        '\u274c': 'X',  # cross mark
        '\u2705': 'v',  # check mark button
        '\ud83e\udde0': 'BRAIN',  # brain emoji
        '\u2192': '->',  # right arrow
        '\u2190': '<-',  # left arrow
        '\u2191': '^',   # up arrow
        '\u2193': 'v',   # down arrow
    }
    result = text
    for old, new in replacements.items():
        result = result.replace(old, new)
    
    # Remove any remaining non-ASCII characters as fallback
    result = ''.join(char if ord(char) < 128 else '?' for char in result)
    return result

def secure_log(agent_name: str, state: dict):
    """Log agent activities with comprehensive state information"""
    from datetime import datetime
    from pathlib import Path
    
    try:
        # Deep clean the entire state to remove any Unicode issues
        cleaned_state = {}
        for key, value in state.items():
            if isinstance(value, str):
                cleaned_state[key] = clean_text(value)
            elif isinstance(value, list):
                cleaned_state[key] = [clean_text(item) if isinstance(item, str) else item for item in value]
            elif isinstance(value, dict):
                cleaned_state[key] = {k: clean_text(v) if isinstance(v, str) else v for k, v in value.items()}
            else:
                cleaned_state[key] = value
        
        # Create comprehensive state snapshot
        state_snapshot = {
            'agent': clean_text(agent_name),
            'timestamp': datetime.now().isoformat(),
            'patient_id': clean_text(str(cleaned_state.get('patient_id', 'N/A'))),
            'claim_id': clean_text(str(cleaned_state.get('claim_id', 'N/A'))),
            'risk_score': cleaned_state.get('risk_score', 0.0),
            'issues': cleaned_state.get('issues', []),
            'resolved_issues': cleaned_state.get('resolved_issues', []),
            'remaining_issues': cleaned_state.get('remaining_issues', []),
            'data_quality_score': cleaned_state.get('data_quality_score', 0.0),
            'final_status': clean_text(str(cleaned_state.get('final_status', 'pending'))),
            'submission_successful': cleaned_state.get('submission_successful', False),
            'policy_compliant': cleaned_state.get('policy_compliant', False),
            'documentation_complete': cleaned_state.get('documentation_complete', False),
            'estimated_approval_rate': cleaned_state.get('estimated_approval_rate', 0.0)
        }
        
        # Create log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'agent': clean_text(agent_name),
            'action': clean_text(f'{agent_name} processing'),
            'state_snapshot': state_snapshot,
            'details': clean_text(f'Processing claim {cleaned_state.get("claim_id", "N/A")} with risk score {cleaned_state.get("risk_score", 0.0)}')
        }
        
        # Store in logs directory
        logs_dir = Path(__file__).parent.parent / 'data' / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = logs_dir / f'{clean_text(agent_name)}_log.jsonl'
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=True) + '\n')
        
        # Console output
        print(f"[{clean_text(agent_name)}] State snapshot logged - Risk: {cleaned_state.get('risk_score', 0.0)}, "
              f"Issues: {len(cleaned_state.get('issues', []))}, "
              f"Data Quality: {cleaned_state.get('data_quality_score', 0.0):.1f}%, "
              f"Status: {clean_text(str(cleaned_state.get('final_status', 'pending')))}")
        
    except Exception as e:
        clean_agent_name = clean_text(agent_name)
        error_msg = clean_text(str(e))
        print(f"[{clean_agent_name}] Error logging state: {error_msg}")
        # Fallback to simple console logging
        print(f"[{clean_agent_name}] Processing claim {clean_text(str(state.get('claim_id', 'N/A')))} - "
              f"Risk: {state.get('risk_score', 0.0):.2f}, "
              f"Issues: {len(state.get('issues', []))}")

    return log_entry
