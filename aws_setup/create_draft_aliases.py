"""
Create 'live' aliases pointing to DRAFT for all 6 agents.
DRAFT has Claude 3.5 Haiku + Action Groups — this is the correct target.
Version 1 and old aliases have already been deleted by fix_aliases.py.
"""
import boto3
import time

REGION = "us-east-1"
bc = boto3.client("bedrock-agent", region_name=REGION)

AGENTS = {
    "RiskPredictor":   "XLAYW801JO",
    "AutoCorrector":   "53KNQP1PMD",
    "ClaimSubmitter":  "BSI3CF17OU",
    "AppealGenerator": "S4YKZVC69F",
    "Resubmitter":     "VCKGXKAZN0",
    "FeedbackLearner": "WUKCB3RG8Z",
}

new_aliases = {}

for name, agent_id in AGENTS.items():
    print(f"\n[{name}] agent={agent_id}")

    # Verify DRAFT state
    a = bc.get_agent(agentId=agent_id)["agent"]
    model = a.get("foundationModel", "N/A")
    status = a.get("agentStatus", "N/A")
    print(f"  DRAFT: model={model}, status={status}")

    ags = bc.list_agent_action_groups(agentId=agent_id, agentVersion="DRAFT").get("actionGroupSummaries", [])
    print(f"  DRAFT AGs: {[ag['actionGroupName'] for ag in ags]}")

    # Create alias pointing to DRAFT
    try:
        resp = bc.create_agent_alias(
            agentId=agent_id,
            agentAliasName="live",
            routingConfiguration=[{"agentVersion": "DRAFT"}],
        )
        alias_id = resp["agentAlias"]["agentAliasId"]
        new_aliases[name] = (agent_id, alias_id)
        print(f"  ✓ Created alias 'live': {alias_id} → DRAFT ({model}, {len(ags)} AGs)")
        time.sleep(1)
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "="*60)
print("NEW ALIAS IDs")
print("="*60)
for name, (agent_id, alias_id) in new_aliases.items():
    print(f"{name}: {alias_id}")

print("\n.env entries to add/update:")
env_keys = {
    "RiskPredictor":   "BEDROCK_AGENT_ALIAS_RISK",
    "AutoCorrector":   "BEDROCK_AGENT_ALIAS_CORRECTOR",
    "ClaimSubmitter":  "BEDROCK_AGENT_ALIAS_SUBMITTER",
    "AppealGenerator": "BEDROCK_AGENT_ALIAS_APPEAL",
    "Resubmitter":     "BEDROCK_AGENT_ALIAS_RESUBMITTER",
    "FeedbackLearner": "BEDROCK_AGENT_ALIAS_FEEDBACK",
}
for name, (agent_id, alias_id) in new_aliases.items():
    print(f"{env_keys[name]}={alias_id}")
