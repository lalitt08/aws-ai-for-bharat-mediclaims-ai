"""Debug agent invocation issues."""
import boto3, json, uuid

REGION = "us-east-1"
bc = boto3.client("bedrock-agent", region_name=REGION)
rt = boto3.client("bedrock-agent-runtime", region_name=REGION)

agent_id = "XLAYW801JO"
alias_id  = "EBAYQHOO1R"  # custom 'live' alias

# Check agent state
agent = bc.get_agent(agentId=agent_id)["agent"]
print(f"Agent: {agent['agentName']}")
print(f"DRAFT model: {agent.get('foundationModel')}")
print(f"DRAFT status: {agent.get('agentStatus')}")

# Check alias
alias = bc.get_agent_alias(agentId=agent_id, agentAliasId=alias_id)["agentAlias"]
print(f"\nAlias: {alias['agentAliasName']} ({alias_id})")
print(f"Alias status: {alias.get('agentAliasStatus')}")
print(f"Routing: {alias.get('routingConfiguration')}")

# Check version 1
try:
    v1 = bc.get_agent_version(agentId=agent_id, agentVersion="1")["agentVersion"]
    print(f"\nVersion 1 model: {v1.get('foundationModel')}")
    print(f"Version 1 status: {v1.get('agentStatus')}")
except Exception as e:
    print(f"Version 1: {e}")

# Check action groups on DRAFT
ags = bc.list_agent_action_groups(agentId=agent_id, agentVersion="DRAFT").get("actionGroupSummaries", [])
print(f"\nDRAFT Action Groups: {[(ag['actionGroupName'], ag['actionGroupState']) for ag in ags]}")

# Check action groups on version 1
try:
    ags_v1 = bc.list_agent_action_groups(agentId=agent_id, agentVersion="1").get("actionGroupSummaries", [])
    print(f"Version 1 Action Groups: {[(ag['actionGroupName'], ag['actionGroupState']) for ag in ags_v1]}")
except Exception as e:
    print(f"Version 1 AGs: {e}")

# Try invoking with custom alias
print(f"\nTrying invoke with alias {alias_id}...")
try:
    resp = rt.invoke_agent(
        agentId=agent_id,
        agentAliasId=alias_id,
        sessionId=str(uuid.uuid4()),
        inputText="What is 2+2?",
    )
    full_text = ""
    for event in resp.get("completion", []):
        chunk = event.get("chunk", {})
        if "bytes" in chunk:
            full_text += chunk["bytes"].decode("utf-8")
    print(f"Response: {full_text[:200]}")
except Exception as e:
    print(f"Error with custom alias: {e}")
