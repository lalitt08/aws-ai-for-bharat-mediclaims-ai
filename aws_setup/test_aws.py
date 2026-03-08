"""Quick AWS connectivity test — S3 + Bedrock Agents"""
import boto3, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

REGION = "us-east-1"
BUCKET = "alpha-claims-demo-390783052961"

print("=== S3 Test ===")
s3 = boto3.client("s3", region_name=REGION)
objs = s3.list_objects_v2(Bucket=BUCKET)
for o in objs.get("Contents", []):
    print(f"  {o['Key']}  ({o['Size']} bytes)")

print("\n=== Bedrock Agents ===")
ba = boto3.client("bedrock-agent", region_name=REGION)
for a in ba.list_agents()["agentSummaries"]:
    print(f"  {a['agentName']:30s} | {a['agentId']} | {a['agentStatus']}")

print("\n=== S3 Storage module ===")
from tools.s3_storage import load_claim_status
data = load_claim_status()
print(f"  claim_status.json has {len(data)} entries")

print("\nAll checks passed.")
