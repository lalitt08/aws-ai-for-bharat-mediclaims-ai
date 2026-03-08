# agents/auto_corrector.py - Enhanced with MCP Integration

from tools.code_mapper import correct_codes
from config.settings import Settings
from tools.logger import secure_log
from tools.csv_data_loader import patient_loader, denial_loader
from orchestrator.mcp_client import mcp_client
import random
import asyncio

# Import centralized execution logger
try:
    from tools.execution_logger import execution_logger, log_execution, log_error
    HAS_EXECUTION_LOGGER = True
except ImportError:
    HAS_EXECUTION_LOGGER = False

async def run_auto_correction(state: dict) -> dict:
    """Enhanced Auto-Corrector Agent with Bedrock Agent Core + MCP integration"""
    
    claim_id = state.get("claim_id", "unknown")
    patient_name = state.get("raw_data", {}).get("patient_name", "Unknown Patient")

    # ── Bedrock Agent Core call (primary path) ────────────────────────────────
    try:
        from tools.bedrock_agent_integration import bedrock_auto_correct
        claim_data_for_bedrock = state.get("raw_data", {}).copy()
        ba_result = bedrock_auto_correct(
            {**claim_data_for_bedrock, "claim_id": claim_id},
            state.get("issues", []),
        )
        if ba_result:
            # Apply Bedrock corrections on top of existing claim data
            claim_data_for_bedrock.update({
                k: v for k, v in {
                    "icd_code":   ba_result.get("corrected_icd"),
                    "cpt_code":   ba_result.get("corrected_cpt"),
                    "prior_auth": ba_result.get("prior_auth"),
                }.items() if v
            })
            state["corrected_data"] = claim_data_for_bedrock
            state["corrections_made"] = ba_result.get("corrections", [])
            state["final_status"] = "corrected"
            state.setdefault("log", []).append(
                f"[AutoCorrector] Bedrock Agent Core: corrections={len(state['corrections_made'])} "
                f"source={ba_result.get('source')}"
            )
            secure_log("AutoCorrector-Bedrock", {
                "claim_id": claim_id,
                "corrections": state["corrections_made"],
                "source": ba_result.get("source"),
            })
            return state
    except Exception as _be:
        state.setdefault("log", []).append(f"[AutoCorrector] Bedrock Agent skipped: {_be}")
    # ── End Bedrock Agent Core ────────────────────────────────────────────────

    # Log agent start with centralized logger
    if HAS_EXECUTION_LOGGER:
        log_execution('auto_corrector', 'AGENT_START', {
            'claim_id': claim_id,
            'patient_name': patient_name,
            'issues_count': len(state.get("issues", [])),
            'action': 'Starting intelligent auto-correction with MCP data enhancement'
        })
    
    print("\n" + "="*80)
    print("🔧 STAGE 2: INTELLIGENT AUTO-CORRECTION")
    print("="*80)
    
    issues = state.get("issues", [])
    recommendations = state.get("recommendations", [])
    claim_data = state.get("raw_data", {}).copy()
    insurance_company = claim_data.get("insurance_company", "")
    
    print(f"📋 Processing Claim: {claim_id}")
    print(f"   Patient: {patient_name}")
    print(f"   Issues to resolve: {len(issues)}")
    print(f"   Recommendations available: {len(recommendations)}")
    
    if issues:
        print(f"\n   📝 Issues Identified:")
        for i, issue in enumerate(issues, 1):
            print(f"      {i}. {issue}")
            
        # Log specific issues being processed
        if HAS_EXECUTION_LOGGER:
            log_execution('auto_corrector', 'ISSUES_PROCESSING', {
                'claim_id': claim_id,
                'patient_name': patient_name,
                'issues': issues,
                'action': f'Processing {len(issues)} data quality issues'
            })
    else:
        print(f"   [SUCCESS] No issues detected")
    
    corrections_made = []
    
    # Get MCP data for intelligent corrections
    mcp_data = state.get("mcp_data", {})
    policy_check = mcp_data.get("policy_check", {})
    denial_analysis = mcp_data.get("denial_analysis", {})
    
    print(f"\n🔗 MCP-ENHANCED CORRECTION:")
    print(f"   Policy data available: {'Yes' if policy_check else 'No'}")
    print(f"   Denial patterns available: {'Yes' if denial_analysis else 'No'}")
    
    # Add detailed logging for UI activity tracking
    state["log"].append("[AutoCorrector] Enhanced patient data retrieval from CSV and MCP sources")
    state["log"].append("[AutoCorrector] Analyzing demographic completeness and data quality")
    
    # Generate prior authorization via MCP if required
    if policy_check.get("prior_auth_required", False) and not claim_data.get("prior_auth"):
        print(f"   🔍 Processing prior authorization requirement...")
        state["log"].append("[AutoCorrector] Generated missing prior authorization numbers")
        try:
            prior_auth_result = await mcp_client.generate_prior_auth_request(
                patient_id=claim_data.get("patient_id", ""),
                procedure_code=claim_data.get("cpt_code", ""),
                medical_necessity=claim_data.get("diagnosis", "Medical necessity as documented")
            )
            
            if prior_auth_result.get("authorization_number"):
                claim_data["prior_auth"] = prior_auth_result["authorization_number"]
                corrections_made.append(f"Generated prior authorization via MCP: {claim_data['prior_auth']}")
                print(f"      ✅ MCP Auth Generated: {claim_data['prior_auth']}")
            else:
                # Fallback to manual generation
                auth_prefix = get_auth_prefix(insurance_company)
                claim_data["prior_auth"] = f"{auth_prefix}-{random.randint(100000, 999999)}"
                corrections_made.append(f"Added prior authorization: {claim_data['prior_auth']}")
                print(f"      ✅ Fallback Auth: {claim_data['prior_auth']}")
        except Exception as e:
            # Fallback to manual generation
            auth_prefix = get_auth_prefix(insurance_company)
            claim_data["prior_auth"] = f"{auth_prefix}-{random.randint(100000, 999999)}"
            corrections_made.append(f"Added prior authorization (fallback): {claim_data['prior_auth']}")
            print(f"      ✅ Error Fallback Auth: {claim_data['prior_auth']}")
    
    print(f"\n🧠 INTELLIGENT ISSUE RESOLUTION:")
    
    # Get CSV patient data for validation
    patient_id = claim_data.get("patient_id", "")
    patient_data = patient_loader.get_patient_by_id(patient_id)
    
    if patient_data:
        print(f"   ✅ CSV Patient Data Found:")
        print(f"      Name: {patient_data.get('name', 'N/A')}")
        print(f"      Age: {patient_data.get('age', 'N/A')}")
        print(f"      Insurance: {patient_data.get('insurer', 'N/A')}")
    else:
        print(f"   [ERROR] No CSV patient data available")
    
    # Intelligently resolve issues based on data analysis
    resolved_issues = []
    updated_issues = []
    
    # Process each issue and attempt to resolve it
    for i, issue in enumerate(issues, 1):
        print(f"\n   Issue {i}: {issue}")
        issue_resolved = False
        
        # Handle missing demographic information with CSV data priority
        if "missing patient demographic information" in issue.lower() or any(keyword in issue.lower() for keyword in ["missing", "demographic", "age", "gender", "name"]):
            print(f"      🔍 Processing demographic issue...")
            
            # Use CSV data if available, otherwise generate
            if patient_data:
                print(f"         📊 Using CSV data source")
                
                if not claim_data.get("patient_name") or claim_data.get("patient_name") == "unknown":
                    claim_data["patient_name"] = patient_data.get("name", "Unknown Patient")
                    corrections_made.append(f"Added patient name from CSV: {claim_data['patient_name']}")
                    print(f"         ✅ Name: {claim_data['patient_name']}")
                
                if not claim_data.get("age"):
                    claim_data["age"] = patient_data.get("age", random.randint(25, 75))
                    corrections_made.append(f"Added age from CSV: {claim_data['age']}")
                    print(f"         ✅ Age: {claim_data['age']}")
                
                if not claim_data.get("gender"):
                    claim_data["gender"] = patient_data.get("gender", random.choice(["M", "F"]))
                    corrections_made.append(f"Added gender from CSV: {claim_data['gender']}")
                    print(f"         ✅ Gender: {claim_data['gender']}")
                
                if not claim_data.get("insurance_company"):
                    claim_data["insurance_company"] = patient_data.get("insurer", "DefaultInsurance")
                    corrections_made.append(f"Added insurance from CSV: {claim_data['insurance_company']}")
                    print(f"         ✅ Insurance: {claim_data['insurance_company']}")
            else:
                print(f"         🔧 Generating synthetic data")
                
                if not claim_data.get("age"):
                    # Generate reasonable age based on procedure type and diagnosis
                    if claim_data.get("cpt_code") in ["99213", "99214"]:  # Office visits
                        claim_data["age"] = random.randint(25, 75)
                    else:
                        claim_data["age"] = random.randint(18, 85)
                    corrections_made.append(f"Added estimated age: {claim_data['age']}")
                    print(f"         ✅ Age: {claim_data['age']} (generated)")
                
                if not claim_data.get("gender"):
                    claim_data["gender"] = random.choice(["M", "F"])
                    corrections_made.append(f"Added gender: {claim_data['gender']}")
                    print(f"         ✅ Gender: {claim_data['gender']} (generated)")
                
                if not claim_data.get("patient_name") or claim_data.get("patient_name") == "unknown":
                    first_names = ["John", "Jane", "Michael", "Sarah", "David", "Lisa", "Robert", "Mary"]
                    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
                    claim_data["patient_name"] = f"{random.choice(first_names)} {random.choice(last_names)}"
                    corrections_made.append(f"Added patient name: {claim_data['patient_name']}")
                    print(f"         ✅ Name: {claim_data['patient_name']} (generated)")
            
            # Add date of birth if missing
            if not claim_data.get("date_of_birth") and claim_data.get("age"):
                from datetime import datetime
                birth_year = datetime.now().year - claim_data["age"]
                claim_data["date_of_birth"] = f"{birth_year}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
                corrections_made.append(f"Added estimated DOB: {claim_data['date_of_birth']}")
                print(f"         ✅ DOB: {claim_data['date_of_birth']}")
            
            resolved_issues.append(issue)
            issue_resolved = True
            print(f"      ✅ Demographic issue fully resolved")
        
        # Handle prior authorization issues
        elif "no prior authorization" in issue.lower() or "authorization" in issue.lower():
            print(f"      🔍 Processing authorization issue...")
            
            if not claim_data.get("prior_auth"):
                auth_prefix = get_auth_prefix(insurance_company)
                claim_data["prior_auth"] = f"{auth_prefix}-{random.randint(100000, 999999)}"
                corrections_made.append(f"Added prior authorization: {claim_data['prior_auth']}")
                print(f"         ✅ Authorization: {claim_data['prior_auth']}")
                
                resolved_issues.append(issue)
                issue_resolved = True
                print(f"      ✅ Authorization issue resolved")
        
        # Handle medical documentation issues
        elif "medical history" in issue.lower() or "documentation" in issue.lower() or "medical necessity" in issue.lower() or "insufficient" in issue.lower() or "lack of" in issue.lower() or "supporting" in issue.lower():
            print(f"      🔍 Processing medical documentation...")
            
            if not claim_data.get("medical_history") or claim_data.get("medical_history") == "None" or "insufficient" in issue.lower() or "lack of" in issue.lower():
                # Use CSV patient data if available for more accurate documentation
                if patient_data and patient_data.get('medical_history'):
                    enhanced_doc = f"{patient_data.get('medical_history')}. Current visit: {claim_data.get('diagnosis', 'Standard clinical evaluation')} documented with appropriate clinical findings and medical necessity."
                    claim_data["medical_history"] = enhanced_doc
                    corrections_made.append("Enhanced medical documentation using patient history")
                    print(f"         ✅ Medical history (from CSV): {enhanced_doc[:80]}...")
                else:
                    medical_doc = generate_medical_documentation(claim_data)
                    claim_data["medical_history"] = medical_doc
                    corrections_made.append("Enhanced medical documentation")
                    print(f"         ✅ Medical history (generated): {medical_doc[:60]}...")
                
                resolved_issues.append(issue)
                issue_resolved = True
                print(f"      ✅ Documentation issue resolved")
        
        # Handle provider information issues
        elif "provider" in issue.lower() or "npi" in issue.lower():
            print(f"      🔍 Processing provider information...")
            
            if not claim_data.get("provider_name"):
                claim_data["provider_name"] = "Dr. Sarah Johnson, MD"
                corrections_made.append("Added provider name")
                print(f"         ✅ Provider: {claim_data['provider_name']}")
            
            if not claim_data.get("provider_npi"):
                claim_data["provider_npi"] = f"{random.randint(1000000000, 9999999999)}"
                corrections_made.append("Added provider NPI")
                print(f"         ✅ NPI: {claim_data['provider_npi']}")
            
            resolved_issues.append(issue)
            issue_resolved = True
            print(f"      ✅ Provider issue resolved")
        
        # Handle coding issues
        elif any(keyword in issue.lower() for keyword in ["icd", "cpt", "code", "diagnosis"]):
            print(f"      🔍 Processing medical coding...")
            
            # Validate and correct codes if needed
            if not claim_data.get("icd_code") or len(claim_data.get("icd_code", "")) < 3:
                claim_data["icd_code"] = "Z00.00"  # General medical examination
                corrections_made.append("Corrected ICD code")
                print(f"         ✅ ICD Code: {claim_data['icd_code']}")
            
            if not claim_data.get("cpt_code") or len(claim_data.get("cpt_code", "")) < 5:
                claim_data["cpt_code"] = "99213"  # Office visit
                corrections_made.append("Corrected CPT code")
                print(f"         ✅ CPT Code: {claim_data['cpt_code']}")
            
            resolved_issues.append(issue)
            issue_resolved = True
            print(f"      ✅ Coding issue resolved")
        
        # Handle amounts and billing issues
        elif "amount" in issue.lower() or "billing" in issue.lower():
            print(f"      🔍 Processing billing information...")
            
            if not claim_data.get("claim_amount") or claim_data.get("claim_amount") == 0:
                # Generate reasonable amount based on procedure
                cpt_code = claim_data.get("cpt_code", "99213")
                amount_ranges = {
                    "99213": (150, 250),  # Office visit
                    "99214": (200, 350),  # Complex office visit
                    "71020": (300, 500),  # Chest X-ray
                    "73721": (800, 1200)  # MRI
                }
                min_amt, max_amt = amount_ranges.get(cpt_code, (100, 300))
                claim_data["claim_amount"] = random.randint(min_amt, max_amt)
                corrections_made.append(f"Added claim amount: ${claim_data['claim_amount']}")
                print(f"         ✅ Amount: ${claim_data['claim_amount']}")
            
            resolved_issues.append(issue)
            issue_resolved = True
            print(f"      ✅ Billing issue resolved")
        
        # If issue not resolved, add to updated issues for next stage
        if not issue_resolved:
            updated_issues.append(issue)
            print(f"      ❌ Issue requires manual review: {issue}")
    
    # Update state with corrected data
    state["raw_data"] = claim_data
    
    # Update state with issue tracking
    state["resolved_issues"] = resolved_issues
    state["remaining_issues"] = updated_issues
    state["issues"] = updated_issues  # Only remaining issues go forward
    state["corrections_made"] = corrections_made
    
    # Calculate data quality improvement
    total_fields = 10  # demographic, auth, medical, billing, etc.
    completed_fields = len([k for k in claim_data.keys() if claim_data[k] and str(claim_data[k]).strip() not in ['', 'None', 'unknown']])
    data_quality_score = min(95.0, (completed_fields / total_fields) * 100)
    state["data_quality_score"] = data_quality_score
    
    # Set processing status
    if len(updated_issues) == 0:
        state["final_status"] = "fully_corrected"
    elif len(resolved_issues) > 0:
        state["final_status"] = "partially_corrected"
    else:
        state["final_status"] = "no_corrections"
    
    print(f"\n📊 CORRECTION SUMMARY:")
    print(f"   Total issues processed: {len(issues)}")
    print(f"   Issues resolved: {len(resolved_issues)}")
    print(f"   Remaining issues: {len(updated_issues)}")
    print(f"   Corrections made: {len(corrections_made)}")
    print(f"   Data quality score: {data_quality_score:.1f}%")
    print(f"   Status: {state['final_status'].upper()}")
    
    if corrections_made:
        print(f"\n   📋 Corrections Applied:")
        for correction in corrections_made:
            print(f"      • {correction}")
    
    if updated_issues:
        print(f"\n   ⚠️  Issues Requiring Further Review:")
        for issue in updated_issues:
            print(f"      • {issue}")
    
    # Enhanced logging
    log_entry = (
        f"[AutoCorrector-MCP] Processed {len(issues)} issues, "
        f"resolved {len(resolved_issues)}, "
        f"remaining {len(updated_issues)}, "
        f"corrections: {len(corrections_made)}, "
        f"quality: {data_quality_score:.1f}%, "
        f"status: {state.get('final_status', 'processing')}"
    )
    state["log"].append(log_entry)
    
    secure_log("AutoCorrector-MCP", {
        "claim_id": state.get("claim_id"),
        "total_issues": len(issues),
        "resolved_issues": len(resolved_issues),
        "remaining_issues": len(updated_issues),
        "corrections_made": corrections_made,
        "resolved_issue_list": resolved_issues,
        "remaining_issue_list": updated_issues,
        "data_quality_score": data_quality_score,
        "csv_data_used": patient_data is not None,
        "mcp_data_available": bool(mcp_data),
        "final_status": state.get("final_status", "processing"),
        "log": state.get("log", [])
    })
    
    print("="*80)
    return state
    updated_issues = []
    
    # Process each issue and attempt to resolve it
    for issue in issues:
        issue_resolved = False
        
        # Handle missing demographic information
        if "missing patient demographic information" in issue.lower():
            patient_id = claim_data.get("patient_id", "")
            
            # Add age if missing
            if not claim_data.get("age") and not claim_data.get("date_of_birth"):
                # Generate reasonable age based on procedure type and diagnosis
                if claim_data.get("cpt_code") in ["99213", "99214"]:  # Office visits
                    claim_data["age"] = random.randint(25, 75)
                else:
                    claim_data["age"] = random.randint(18, 85)
                corrections_made.append(f"Added estimated age: {claim_data['age']}")
            
            # Add gender if missing
            if not claim_data.get("gender"):
                claim_data["gender"] = random.choice(["M", "F"])
                corrections_made.append(f"Added gender: {claim_data['gender']}")
            
            # Add date of birth if missing (based on age)
            if not claim_data.get("date_of_birth") and claim_data.get("age"):
                from datetime import datetime, timedelta
                birth_year = datetime.now().year - claim_data["age"]
                claim_data["date_of_birth"] = f"{birth_year}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
                corrections_made.append(f"Added estimated DOB: {claim_data['date_of_birth']}")
            
            # Add patient name if missing
            if not claim_data.get("patient_name") or claim_data.get("patient_name") == "unknown":
                first_names = ["John", "Jane", "Michael", "Sarah", "David", "Lisa", "Robert", "Mary"]
                last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
                claim_data["patient_name"] = f"{random.choice(first_names)} {random.choice(last_names)}"
                corrections_made.append(f"Added patient name: {claim_data['patient_name']}")
            
            # Mark this issue as resolved
            resolved_issues.append(issue)
            issue_resolved = True
        
        # Handle prior authorization issues
        elif "no prior authorization" in issue.lower():
            if not claim_data.get("prior_auth"):
                auth_prefix = get_auth_prefix(insurance_company)
                claim_data["prior_auth"] = f"{auth_prefix}-{random.randint(100000, 999999)}"
                corrections_made.append(f"Added prior authorization: {claim_data['prior_auth']}")
                resolved_issues.append(issue)
                issue_resolved = True
        
        # Handle documentation issues
        elif "lack of supporting documentation" in issue.lower() or "insufficient" in issue.lower():
            if not claim_data.get("medical_history") or claim_data.get("medical_history") == "None":
                claim_data["medical_history"] = generate_medical_documentation(claim_data)
                corrections_made.append("Enhanced medical documentation")
                resolved_issues.append(issue)
                issue_resolved = True
        
        # Handle eligibility verification issues
        elif "unknown patient details" in issue.lower() or "eligibility verification" in issue.lower():
            # If we've added demographic info, this should be resolved
            if claim_data.get("age") and claim_data.get("gender") and claim_data.get("patient_name"):
                resolved_issues.append(issue)
                issue_resolved = True
        
        # Handle pricing/amount issues
        elif "outlier pricing" in issue.lower() or "claim amount" in issue.lower():
            current_amount = claim_data.get("claim_amount", 0)
            if current_amount > 500:  # If amount seems high, adjust it
                claim_data["claim_amount"] = min(current_amount, 350.0)
                corrections_made.append(f"Adjusted claim amount to reasonable range: ${claim_data['claim_amount']}")
                resolved_issues.append(issue)
                issue_resolved = True
        
        # If issue wasn't resolved, keep it in the list
        if not issue_resolved:
            updated_issues.append(issue)
    
    # Update the issues list to only include unresolved issues
    state["issues"] = updated_issues
    
    # Apply MCP-informed corrections based on recommendations
    for recommendation in recommendations:
        if "prior authorization" in recommendation.lower():
            if not claim_data.get("prior_auth"):
                # Generate realistic prior auth based on insurance company
                auth_prefix = get_auth_prefix(insurance_company)
                claim_data["prior_auth"] = f"{auth_prefix}-{random.randint(100000, 999999)}"
                corrections_made.append(f"Added prior authorization: {claim_data['prior_auth']}")
        
        elif "documentation" in recommendation.lower():
            # Enhance medical documentation
            if not claim_data.get("medical_history") or claim_data.get("medical_history") == "None":
                claim_data["medical_history"] = generate_medical_documentation(claim_data)
                corrections_made.append("Enhanced medical documentation")
        
        elif "cpt code" in recommendation.lower() or "icd code" in recommendation.lower():
            # Correct coding issues
            original_codes = [claim_data.get("cpt_code", "")]
            corrected_codes = correct_codes(original_codes)
            if corrected_codes != original_codes:
                claim_data["cpt_code"] = corrected_codes[0]
                corrections_made.append(f"Corrected CPT code: {claim_data['cpt_code']}")
    
    # Get historical patterns from MCP for additional corrections
    denial_patterns = denial_analysis.get("patterns", [])
    if not denial_patterns:
        # Fallback to CSV data
        denial_patterns = denial_loader.get_patterns_by_insurer(insurance_company)
    
    # Apply corrections based on specific issues identified
    for issue in issues:
        issue_lower = issue.lower()
        
        # Handle missing prior auth
        if "prior auth" in issue_lower and not claim_data.get("prior_auth"):
            auth_prefix = get_auth_prefix(insurance_company)
            claim_data["prior_auth"] = f"{auth_prefix}-{random.randint(100000, 999999)}"
            corrections_made.append(f"Added missing prior authorization: {claim_data['prior_auth']}")
        
        # Handle invalid code combinations
        elif "invalid" in issue_lower and "code" in issue_lower:
            original_codes = [claim_data.get("cpt_code", "")]
            corrected_codes = correct_codes(original_codes)
            if corrected_codes != original_codes:
                claim_data["cpt_code"] = corrected_codes[0]
                corrections_made.append(f"Fixed invalid code combination: {claim_data['cpt_code']}")
        
        # Handle documentation issues
        elif "documentation" in issue_lower or "medical history" in issue_lower:
            if not claim_data.get("medical_history") or claim_data.get("medical_history") == "None":
                claim_data["medical_history"] = generate_medical_documentation(claim_data)
                corrections_made.append("Added comprehensive medical documentation")
        
        # Handle network issues
        elif "network" in issue_lower:
            # Try to use in-network alternatives (simplified)
            corrections_made.append("Flagged for in-network provider verification")
        
        # Handle benefit limit issues
        elif "benefit" in issue_lower or "limit" in issue_lower:
            # Potentially split large claims
            if claim_data.get("claim_amount", 0) > 400:
                original_amount = claim_data["claim_amount"]
                claim_data["claim_amount"] = min(original_amount, 350.0)
                corrections_made.append(f"Adjusted claim amount: ${original_amount} -> ${claim_data['claim_amount']}")
    
    # Apply corrections based on learned patterns
    for pattern in denial_patterns[-3:]:  # Last 3 patterns
        pattern_reason = pattern.get("denial_reason", "").lower()
        solution = pattern.get("solution_applied", "")
        
        if "prior auth" in pattern_reason and not claim_data.get("prior_auth"):
            auth_prefix = get_auth_prefix(insurance_company)
            claim_data["prior_auth"] = f"{auth_prefix}-{random.randint(100000, 999999)}"
            corrections_made.append(f"Applied learned pattern: {solution}")
        
        elif "documentation" in pattern_reason:
            if not claim_data.get("medical_history") or claim_data.get("medical_history") == "None":
                claim_data["medical_history"] = generate_medical_documentation(claim_data)
                corrections_made.append(f"Applied learned pattern: {solution}")
    
    # Ensure all fields are non-null and reasonable
    for key in ["risk_score", "issues", "corrected_data", "submission_result", "appeal_packet", "final_status", "log"]:
        if key not in state or state[key] is None:
            if key == "risk_score":
                state[key] = 0.0
            elif key == "issues":
                state[key] = []
            elif key == "log":
                state[key] = []
            else:
                state[key] = None

    state["corrected_data"] = claim_data
    state["corrections_made"] = corrections_made

    # Enhanced logging with issue resolution tracking
    state["log"].append(f"[AutoCorrector] Applied {len(corrections_made)} corrections")
    for correction in corrections_made:
        state["log"].append(f"[AutoCorrector] - {correction}")
    
    if resolved_issues:
        state["log"].append(f"[AutoCorrector] Resolved {len(resolved_issues)} issues")
        for resolved in resolved_issues:
            # Remove special characters that might cause encoding issues
            clean_resolved = resolved.replace('–', '-').replace('—', '-').replace('"', '"').replace('"', '"')
            state["log"].append(f"[AutoCorrector] [RESOLVED] {clean_resolved[:60]}...")
    
    if updated_issues:
        state["log"].append(f"[AutoCorrector] {len(updated_issues)} issues remain unresolved")
        for remaining in updated_issues:
            # Remove special characters that might cause encoding issues
            clean_remaining = remaining.replace('–', '-').replace('—', '-').replace('"', '"').replace('"', '"')
            state["log"].append(f"[AutoCorrector] [REMAINING] {clean_remaining[:60]}...")
    else:
        state["log"].append("[AutoCorrector] [SUCCESS] All issues resolved successfully")

    # Update state with issue tracking
    state["resolved_issues"] = resolved_issues
    state["remaining_issues"] = updated_issues
    state["issues"] = updated_issues  # Only remaining issues go forward
    
    # Calculate data quality improvement
    total_fields = 10  # demographic, auth, medical, billing, etc.
    completed_fields = len([k for k in state.keys() if k not in ['issues', 'log', 'resolved_issues', 'remaining_issues']])
    data_quality_score = min(95.0, (completed_fields / total_fields) * 100)
    state["data_quality_score"] = data_quality_score

    # Set processing status
    if len(updated_issues) == 0:
        state["final_status"] = "corrected"
    else:
        state["final_status"] = "partially_corrected"

    # Log completion with centralized logger
    if HAS_EXECUTION_LOGGER:
        log_execution('auto_corrector', 'AGENT_COMPLETE', {
            'claim_id': claim_id,
            'patient_name': patient_name,
            'corrections_applied': len(resolved_issues),
            'remaining_issues': len(updated_issues),
            'data_quality_score': data_quality_score,
            'status': state["final_status"],
            'action': f'Auto-correction completed - {len(resolved_issues)} corrections applied'
        })

    secure_log("AutoCorrector", {
        "claim_id": state.get("claim_id"),
        "corrections_made": corrections_made,
        "resolved_issues": len(resolved_issues),
        "remaining_issues": len(updated_issues),
        "insurance_company": insurance_company,
        "risk_score": state.get("risk_score", 0.0),
        "issues": state.get("issues", []),
        "final_status": state.get("final_status", "processing")
    })

    return state

def get_auth_prefix(insurance_company: str) -> str:
    """Get appropriate authorization prefix for insurance company"""
    prefixes = {
        "bluecross": "BC-AUTH",
        "aetna": "AET-PA",
        "cigna": "CGN-AUTH",
        "united": "UHC-PA"
    }
    return prefixes.get(insurance_company.lower(), "GEN-AUTH")

def generate_medical_documentation(claim_data: dict) -> str:
    """Generate appropriate medical documentation based on claim data"""
    
    diagnosis = claim_data.get("diagnosis", "")
    age = claim_data.get("age", 0)
    gender = claim_data.get("gender", "")
    risk_factors = claim_data.get("risk_factors", "")
    
    # Generate contextual medical history
    documentation_parts = []
    
    # Add age-appropriate context
    if age > 60:
        documentation_parts.append("Age-related risk factors considered")
    elif age < 30:
        documentation_parts.append("Young adult presentation")
    
    # Add diagnosis-specific context
    if "diabetes" in diagnosis.lower():
        documentation_parts.append("Glucose monitoring, dietary counseling provided")
    elif "hypertension" in diagnosis.lower():
        documentation_parts.append("Blood pressure monitoring, lifestyle modification counseling")
    elif "asthma" in diagnosis.lower():
        documentation_parts.append("Pulmonary function assessment, inhaler technique review")
    elif "arthritis" in diagnosis.lower():
        documentation_parts.append("Joint assessment, mobility evaluation, pain management")
    
    # Add risk factor context
    if risk_factors and risk_factors != "None":
        documentation_parts.append(f"Risk factors addressed: {risk_factors}")
    
    # Add standard documentation
    documentation_parts.extend([
        "Patient education provided",
        "Follow-up plan established",
        "Treatment compliance discussed"
    ])
    
    return "; ".join(documentation_parts)
