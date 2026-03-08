"""
Delete v2 aliases + create new 'live' aliases from DRAFT (Nova Micro + Action Groups).
"""
import boto3, time

REGION = "us-east-1"
bc = boto3.client("bedrock-agent", region_name=REGION)

# Current v2 aliases (from create_v2.py)
AGENTS = {
    "RiskPredictor":   ("XLAYW801JO", "LV9RD6DF2S"),
    "AutoCorrector":   ("53KNQP1PMD", "VUEBZQUUMH"),
    "ClaimSubmitter":  ("BSI3CF17OU", "IF4URG0JBQ"),
    "AppealGenerator": ("S4YKZVC69F", "ACNFYPPJ9X"),
    "Resubmitter":     ("VCKGXKAZN0", "Q6HRVPUVUC"),
    "FeedbackLearner": ("WUKCB3RG8Z", "BI8VHL8Y3X"),
}

new_aliases = {}

for name, (agent_id, old_alias_id) in AGENTS.items():
    print(f"\n[{name}]")

    # Delete old alias
    try:
        bc.delete_agent_alias(agentId=agent_id, agentAliasId=old_alias_id)
        print(f"  ✓ Deleted old alias {old_alias_id}")
        time.sleep(1)
    except Exception as e:
        print(f"  Delete alias: {e}")

    # Create new alias (no routingConfiguration → auto-creates version from DRAFT)
    try:
        resp = bc.create_agent_alias(agentId=agent_id, agentAliasName="live")
        alias_id = resp["agentAlias"]["agentAliasId"]
        new_aliases[name] = (agent_id, alias_id)
        print(f"  ✓ New alias: {alias_id}")
        time.sleep(1)
    except Exception as e:
        print(f"  ERROR: {e}")

time.sleep(10)

print("\n" + "="*60)
print("Verifying new aliases:")
print("="*60)
for name, (agent_id, alias_id) in new_aliases.items():
    try:
        info = bc.get_agent_alias(agentId=agent_id, agentAliasId=alias_id)["agentAlias"]
        routing = info.get("routingConfiguration", [])
        status = info.get("agentAliasStatus", "?")
        for r in routing:
            ver = r.get("agentVersion", "?")
            ver_info = bc.get_agent_version(agentId=agent_id, agentVersion=ver)["agentVersion"]
            ver_model = ver_info.get("foundationModel", "?")
            ags = bc.list_agent_action_groups(agentId=agent_id, agentVersion=ver).get("actionGroupSummaries", [])
            print(f"{name}: alias={alias_id} → v{ver}, model={ver_model}, AGs={len(ags)}, status={status}")
    except Exception as e:
        print(f"{name}: ERROR {e}")

print("\n.env entries:")
env_keys = {
    "RiskPredictor":   "BEDROCK_AGENT_ALIAS_RISK_PREDICTOR",
    "AutoCorrector":   "BEDROCK_AGENT_ALIAS_AUTO_CORRECTOR",
    "ClaimSubmitter":  "BEDROCK_AGENT_ALIAS_CLAIM_SUBMITTER",
    "AppealGenerator": "BEDROCK_AGENT_ALIAS_APPEAL_GENERATOR",
    "Resubmitter":     "BEDROCK_AGENT_ALIAS_RESUBMITTER",
    "FeedbackLearner": "BEDROCK_AGENT_ALIAS_FEEDBACK_LEARNER",
}
for name, (agent_id, alias_id) in new_aliases.items():
    print(f"{env_keys[name]}={alias_id}")
