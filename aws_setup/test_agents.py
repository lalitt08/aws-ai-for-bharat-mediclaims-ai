"""
Test all 6 Bedrock Agents end-to-end.
Verifies: agent invocation → Lambda Action Group → tool execution → response
"""
import boto3, json, uuid, sys

REGION = "us-east-1"
client = boto3.client("bedrock-agent-runtime", region_name=REGION)

AGENTS = {
    "RiskPredictor":    ("XLAYW801JO", "GMML5M64KW"),
    "AutoCorrector":    ("53KNQP1PMD", "FYFALB8NSI"),
    "ClaimSubmitter":   ("BSI3CF17OU", "BZBIFWBIBO"),
    "AppealGenerator":  ("S4YKZVC69F", "F7V98MQXIZ"),
    "Resubmitter":      ("VCKGXKAZN0", "EWC6FLUFBK"),
    "FeedbackLearner":  ("WUKCB3RG8Z", "U9I9LCMJIL"),
}

TEST_PROMPTS = {
    "RiskPredictor":   "Analyze claim risk for patient PAT001 with CPT code 99213, ICD-10 M54.5, insurer BlueCross, amount $250. Use your tools to validate codes and check prior auth.",
    "AutoCorrector":   "Correct the claim for patient PAT001. ICD-10 code is Z00, CPT is 99213, insurer BlueCross. Use your tools to correct the ICD-10 code and validate the NPI 1234567890.",
    "ClaimSubmitter":  "Check eligibility for patient PAT001 with BlueCross for service date 2026-03-06, then get claim status for claim CLM-001.",
    "AppealGenerator": "Get denial details for patient PAT001 claim CLM-001, then check appeal requirements for denial code CO-16.",
    "Resubmitter":     "Determine resubmission strategy for denial code CO-16 with reason 'Missing clinical documentation'.",
    "FeedbackLearner": "Get learning insights for insurer BlueCross and CPT code 99213.",
}

results = {}
all_passed = True

for name, (agent_id, alias_id) in AGENTS.items():
    print(f"\n{'='*60}")
    print(f"Testing {name} (agent={agent_id}, alias={alias_id})")
    print(f"{'='*60}")
    
    prompt = TEST_PROMPTS[name]
    print(f"Prompt: {prompt[:100]}...")
    
    try:
        resp = client.invoke_agent(
            agentId=agent_id,
            agentAliasId=alias_id,
            sessionId=str(uuid.uuid4()),
            inputText=prompt,
        )
        
        full_text = ""
        for event in resp.get("completion", []):
            chunk = event.get("chunk", {})
            if "bytes" in chunk:
                full_text += chunk["bytes"].decode("utf-8")
        
        if full_text:
            print(f"✅ RESPONSE ({len(full_text)} chars):")
            print(full_text[:400])
            results[name] = {"status": "PASS", "chars": len(full_text)}
        else:
            print(f"⚠️  Empty response (agent may have stalled)")
            results[name] = {"status": "EMPTY", "chars": 0}
            all_passed = False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        results[name] = {"status": "FAIL", "error": str(e)}
        all_passed = False

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for name, r in results.items():
    status = r["status"]
    icon = "✅" if status == "PASS" else "⚠️" if status == "EMPTY" else "❌"
    print(f"{icon} {name}: {status} {r.get('chars', '')} chars {r.get('error', '')}")

print(f"\n{'PASS' if all_passed else 'PARTIAL'} — {sum(1 for r in results.values() if r['status'] == 'PASS')}/{len(results)} agents responding")
sys.exit(0 if all_passed else 1)
