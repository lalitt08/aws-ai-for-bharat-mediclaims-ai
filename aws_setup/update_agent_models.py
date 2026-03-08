"""
Update all 6 Bedrock Agents to use Amazon Nova Micro (us.amazon.nova-micro-v1:0).
Nova Micro: works with IAM keys, supports tool use in Agent Core, no marketplace subscription needed.
"""
import boto3
import time

REGION = "us-east-1"
bc = boto3.client("bedrock-agent", region_name=REGION)

MODEL = "us.amazon.nova-micro-v1:0"

AGENTS = {
    "RiskPredictor":   "XLAYW801JO",
    "AutoCorrector":   "53KNQP1PMD",
    "ClaimSubmitter":  "BSI3CF17OU",
    "AppealGenerator": "S4YKZVC69F",
    "Resubmitter":     "VCKGXKAZN0",
    "FeedbackLearner": "WUKCB3RG8Z",
}

for name, agent_id in AGENTS.items():
    print(f"\n[{name}] agent={agent_id}")

    # Get current agent config
    agent = bc.get_agent(agentId=agent_id)["agent"]
    current_model = agent.get("foundationModel", "")
    print(f"  Current model: {current_model}")

    if current_model == MODEL:
        print(f"  Already using {MODEL}, skipping update")
        continue

    # Update agent model
    try:
        bc.update_agent(
            agentId=agent_id,
            agentName=agent["agentName"],
            agentResourceRoleArn=agent["agentResourceRoleArn"],
            foundationModel=MODEL,
            instruction=agent.get("instruction", "You are a helpful medical claims processing agent."),
        )
        print(f"  ✓ Updated to {MODEL}")
        time.sleep(2)
    except Exception as e:
        print(f"  ERROR updating: {e}")
        continue

    # Prepare agent
    print(f"  Preparing agent...")
    try:
        bc.prepare_agent(agentId=agent_id)
        for _ in range(20):
            time.sleep(5)
            status = bc.get_agent(agentId=agent_id)["agent"]["agentStatus"]
            print(f"  Status: {status}")
            if status == "PREPARED":
                break
        print(f"  ✓ PREPARED")
    except Exception as e:
        print(f"  ERROR preparing: {e}")

print("\n" + "="*60)
print("Verifying final state:")
print("="*60)
for name, agent_id in AGENTS.items():
    a = bc.get_agent(agentId=agent_id)["agent"]
    model = a.get("foundationModel", "N/A")
    status = a.get("agentStatus", "N/A")
    ags = bc.list_agent_action_groups(agentId=agent_id, agentVersion="DRAFT").get("actionGroupSummaries", [])
    print(f"{name}: model={model}, status={status}, AGs={len(ags)}")
