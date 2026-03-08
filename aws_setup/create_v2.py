"""
Create new 'live' aliases WITHOUT routingConfiguration.
Per AWS docs: omitting routingConfiguration causes Bedrock to auto-create
a new version from DRAFT and point the alias to it.
DRAFT has Claude 3.5 Haiku + Action Groups — this is what we want.
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

    # Verify DRAFT state first
    a = bc.get_agent(agentId=agent_id)["agent"]
    model = a.get("foundationModel", "N/A")
    status = a.get("agentStatus", "N/A")
    ags = bc.list_agent_action_groups(agentId=agent_id, agentVersion="DRAFT").get("actionGroupSummaries", [])
    print(f"  DRAFT: model={model}, status={status}, AGs={len(ags)}")

    # Create alias WITHOUT routingConfiguration → auto-creates new version from DRAFT
    try:
        resp = bc.create_agent_alias(
            agentId=agent_id,
            agentAliasName="live",
            # No routingConfiguration — Bedrock creates a new version from DRAFT
        )
        alias = resp["agentAlias"]
        alias_id = alias["agentAliasId"]
        routing = alias.get("routingConfiguration", [])
        new_aliases[name] = (agent_id, alias_id)
        print(f"  ✓ Alias created: {alias_id}, routing={routing}")
        time.sleep(2)
    except Exception as e:
        print(f"  ERROR: {e}")

# Wait for aliases to be PREPARED
print("\nWaiting for aliases to be ready...")
time.sleep(10)

print("\n" + "="*60)
print("RESULTS — verify version and model:")
print("="*60)
for name, (agent_id, alias_id) in new_aliases.items():
    try:
        alias_info = bc.get_agent_alias(agentId=agent_id, agentAliasId=alias_id)["agentAlias"]
        routing = alias_info.get("routingConfiguration", [])
        status = alias_info.get("agentAliasStatus", "?")
        print(f"\n{name}: alias={alias_id}, status={status}")
        for r in routing:
            ver = r.get("agentVersion", "?")
            print(f"  → version {ver}")
            if ver != "DRAFT":
                try:
                    ver_info = bc.get_agent_version(agentId=agent_id, agentVersion=ver)["agentVersion"]
                    ver_model = ver_info.get("foundationModel", "?")
                    ver_ags = bc.list_agent_action_groups(agentId=agent_id, agentVersion=ver).get("actionGroupSummaries", [])
                    print(f"    model={ver_model}, AGs={[ag['actionGroupName'] for ag in ver_ags]}")
                except Exception as e:
                    print(f"    (could not get version details: {e})")
    except Exception as e:
        print(f"{name}: ERROR {e}")

print("\n" + "="*60)
print(".env entries (add these):")
print("="*60)
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
