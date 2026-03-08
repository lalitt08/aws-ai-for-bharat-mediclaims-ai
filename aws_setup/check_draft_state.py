"""Check DRAFT model and action groups for all 6 agents."""
import boto3

bc = boto3.client("bedrock-agent", region_name="us-east-1")

AGENTS = {
    "RiskPredictor":   "XLAYW801JO",
    "AutoCorrector":   "53KNQP1PMD",
    "ClaimSubmitter":  "BSI3CF17OU",
    "AppealGenerator": "S4YKZVC69F",
    "Resubmitter":     "VCKGXKAZN0",
    "FeedbackLearner": "WUKCB3RG8Z",
}

for name, aid in AGENTS.items():
    a = bc.get_agent(agentId=aid)["agent"]
    model = a.get("foundationModel", "N/A")
    status = a.get("agentStatus", "N/A")
    print(f"{name}: model={model}, status={status}")
    ags = bc.list_agent_action_groups(agentId=aid, agentVersion="DRAFT").get("actionGroupSummaries", [])
    for ag in ags:
        print(f"  AG: {ag['actionGroupName']} state={ag.get('actionGroupState','?')}")
    if not ags:
        print("  NO ACTION GROUPS on DRAFT")
