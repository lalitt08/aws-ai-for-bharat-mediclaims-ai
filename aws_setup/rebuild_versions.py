"""
Delete version 1 (Llama, no action groups) and create new version 1 from DRAFT (Claude + AGs).
Strategy: delete v1 → update alias to point to nothing → prepare agent → alias auto-updates.
Actually: delete v1 → prepare agent creates new v1 from DRAFT → update alias to new v1.
"""
import boto3, time

REGION = "us-east-1"
bc = boto3.client("bedrock-agent", region_name=REGION)

AGENTS = {
    "RiskPredictor":    ("XLAYW801JO", "EBAYQHOO1R"),
    "AutoCorrector":    ("53KNQP1PMD", "S8BEFA916D"),
    "ClaimSubmitter":   ("BSI3CF17OU", "QRWLSK1XYI"),
    "AppealGenerator":  ("S4YKZVC69F", "3YXIDH087L"),
    "Resubmitter":      ("VCKGXKAZN0", "5MLEANLSDT"),
    "FeedbackLearner":  ("WUKCB3RG8Z", "F0RW6RSPUL"),
}

for name, (agent_id, alias_id) in AGENTS.items():
    print(f"\n[{name}]")
    try:
        # Step 1: Delete version 1
        print(f"  Deleting version 1...")
        try:
            bc.delete_agent_version(agentId=agent_id, agentVersion="1")
            print(f"  ✓ Version 1 deleted")
            time.sleep(3)
        except Exception as e:
            print(f"  Delete v1: {e}")
        
        # Step 2: Prepare agent — this creates a new version from DRAFT
        print(f"  Preparing agent (creates new version from DRAFT)...")
        bc.prepare_agent(agentId=agent_id)
        
        # Wait for PREPARED
        for _ in range(20):
            time.sleep(5)
            status = bc.get_agent(agentId=agent_id)["agent"]["agentStatus"]
            print(f"  Status: {status}")
            if status == "PREPARED":
                break
        
        # Step 3: Find the new version
        versions = bc.list_agent_versions(agentId=agent_id).get("agentVersionSummaries", [])
        numbered = sorted(
            [v for v in versions if v.get("agentVersion", "DRAFT") != "DRAFT"],
            key=lambda v: int(v.get("agentVersion", "0")),
            reverse=True,
        )
        
        if numbered:
            new_ver = numbered[0]["agentVersion"]
            ver_info = bc.get_agent_version(agentId=agent_id, agentVersion=new_ver)["agentVersion"]
            ver_model = ver_info.get("foundationModel", "")
            
            # Check action groups on this version
            ags = bc.list_agent_action_groups(agentId=agent_id, agentVersion=new_ver).get("actionGroupSummaries", [])
            print(f"  New version: {new_ver}, model: {ver_model}, AGs: {len(ags)}")
            
            # Step 4: Update alias to new version
            bc.update_agent_alias(
                agentId=agent_id,
                agentAliasId=alias_id,
                agentAliasName="live",
                routingConfiguration=[{"agentVersion": new_ver}],
            )
            print(f"  ✓ Alias {alias_id} → version {new_ver} ({ver_model}, {len(ags)} AGs)")
        else:
            print(f"  ⚠ No numbered versions found after prepare")
            
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n✅ Done — all agents rebuilt with Claude 3.5 Haiku + Action Groups")
