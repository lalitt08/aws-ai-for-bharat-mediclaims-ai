"""
Fix aliases: delete alias → delete version 1 → create version from DRAFT → recreate alias.
This is the correct flow to get Claude 3.5 Haiku + Action Groups into a versioned alias.
"""
import boto3
import time

REGION = "us-east-1"
bc = boto3.client("bedrock-agent", region_name=REGION)

AGENTS = {
    "RiskPredictor":   ("XLAYW801JO", "EBAYQHOO1R"),
    "AutoCorrector":   ("53KNQP1PMD", "S8BEFA916D"),
    "ClaimSubmitter":  ("BSI3CF17OU", "QRWLSK1XYI"),
    "AppealGenerator": ("S4YKZVC69F", "3YXIDH087L"),
    "Resubmitter":     ("VCKGXKAZN0", "5MLEANLSDT"),
    "FeedbackLearner": ("WUKCB3RG8Z", "F0RW6RSPUL"),
}

new_alias_ids = {}

for name, (agent_id, alias_id) in AGENTS.items():
    print(f"\n[{name}] agent={agent_id}")

    # Step 1: Delete the existing alias (unblocks version deletion)
    print("  Step 1: Deleting alias...")
    try:
        bc.delete_agent_alias(agentId=agent_id, agentAliasId=alias_id)
        print(f"  ✓ Alias {alias_id} deleted")
        time.sleep(2)
    except Exception as e:
        print(f"  Alias delete: {e}")

    # Step 2: Delete version 1
    print("  Step 2: Deleting version 1...")
    try:
        bc.delete_agent_version(agentId=agent_id, agentVersion="1")
        print("  ✓ Version 1 deleted")
        time.sleep(3)
    except Exception as e:
        print(f"  Version delete: {e}")

    # Step 3: Create a new version from DRAFT (snapshots current DRAFT state)
    print("  Step 3: Creating new version from DRAFT...")
    try:
        resp = bc.create_agent_version(agentId=agent_id)
        new_ver = resp["agentVersion"]["agentVersion"]
        ver_model = resp["agentVersion"].get("foundationModel", "")
        print(f"  ✓ Created version {new_ver} (model: {ver_model})")
        time.sleep(3)
    except Exception as e:
        print(f"  ERROR creating version: {e}")
        continue

    # Step 4: Verify action groups on new version
    ags = bc.list_agent_action_groups(agentId=agent_id, agentVersion=new_ver).get("actionGroupSummaries", [])
    print(f"  Action groups on v{new_ver}: {[ag['actionGroupName'] for ag in ags]}")

    # Step 5: Create new alias pointing to new version
    print(f"  Step 5: Creating alias 'live' → version {new_ver}...")
    try:
        alias_resp = bc.create_agent_alias(
            agentId=agent_id,
            agentAliasName="live",
            routingConfiguration=[{"agentVersion": new_ver}],
        )
        new_alias_id = alias_resp["agentAlias"]["agentAliasId"]
        new_alias_ids[name] = (agent_id, new_alias_id, new_ver)
        print(f"  ✓ New alias: {new_alias_id} → v{new_ver} ({ver_model}, {len(ags)} AGs)")
    except Exception as e:
        print(f"  ERROR creating alias: {e}")

print("\n" + "="*60)
print("NEW ALIAS IDs (update .env with these):")
print("="*60)
for name, (agent_id, alias_id, ver) in new_alias_ids.items():
    key = name.upper().replace("PREDICTOR", "_PREDICTOR").replace("CORRECTOR", "_CORRECTOR") \
               .replace("SUBMITTER", "_SUBMITTER").replace("GENERATOR", "_GENERATOR") \
               .replace("RESUBMITTER", "_RESUBMITTER").replace("LEARNER", "_LEARNER")
    print(f"{name}: agent={agent_id}, alias={alias_id}, version={ver}")

print("\n.env entries:")
env_keys = {
    "RiskPredictor":   "BEDROCK_AGENT_ALIAS_RISK",
    "AutoCorrector":   "BEDROCK_AGENT_ALIAS_CORRECTOR",
    "ClaimSubmitter":  "BEDROCK_AGENT_ALIAS_SUBMITTER",
    "AppealGenerator": "BEDROCK_AGENT_ALIAS_APPEAL",
    "Resubmitter":     "BEDROCK_AGENT_ALIAS_RESUBMITTER",
    "FeedbackLearner": "BEDROCK_AGENT_ALIAS_FEEDBACK",
}
for name, (agent_id, alias_id, ver) in new_alias_ids.items():
    env_key = env_keys.get(name, f"BEDROCK_AGENT_ALIAS_{name.upper()}")
    print(f"{env_key}={alias_id}")
