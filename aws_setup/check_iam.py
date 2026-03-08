"""Check IAM permissions for admin user and BedrockAgentsClaimsRole."""
import boto3, json

iam = boto3.client('iam', region_name='us-east-1')

print("=== admin user policies ===")
attached = iam.list_attached_user_policies(UserName='admin')['AttachedPolicies']
print("Attached:", [p['PolicyName'] for p in attached])

inline = iam.list_user_policies(UserName='admin')['PolicyNames']
print("Inline:", inline)

groups = iam.list_groups_for_user(UserName='admin')['Groups']
print("Groups:", [g['GroupName'] for g in groups])
for g in groups:
    gname = g['GroupName']
    gp = iam.list_attached_group_policies(GroupName=gname)['AttachedPolicies']
    print(f"  {gname} attached: {[p['PolicyName'] for p in gp]}")

# Simulate the InvokeAgent call to see what's denied
print("\n=== Simulate bedrock:InvokeAgent ===")
try:
    # Try a simple bedrock list call to confirm bedrock access
    br = boto3.client('bedrock', region_name='us-east-1')
    models = br.list_foundation_models(byOutputModality='TEXT')
    ids = [m['modelId'] for m in models['modelSummaries'][:3]]
    print("bedrock:ListFoundationModels OK, sample:", ids)
except Exception as e:
    print("bedrock access error:", e)

# Try bedrock-agent list
try:
    bc = boto3.client('bedrock-agent', region_name='us-east-1')
    agents = bc.list_agents()['agentSummaries']
    print(f"bedrock-agent:ListAgents OK, {len(agents)} agents")
except Exception as e:
    print("bedrock-agent error:", e)

# Try bedrock-agent-runtime directly
try:
    import uuid
    rt = boto3.client('bedrock-agent-runtime', region_name='us-east-1')
    resp = rt.invoke_agent(
        agentId='XLAYW801JO',
        agentAliasId='LV9RD6DF2S',
        sessionId=str(uuid.uuid4()),
        inputText='Hello',
    )
    text = ''
    for event in resp.get('completion', []):
        chunk = event.get('chunk', {})
        if 'bytes' in chunk:
            text += chunk['bytes'].decode('utf-8')
    print("InvokeAgent OK:", text[:100])
except Exception as e:
    print("InvokeAgent error:", type(e).__name__, str(e))
