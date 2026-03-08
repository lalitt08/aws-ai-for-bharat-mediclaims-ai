"""Find which Claude model is actually accessible — check full error messages."""
import boto3, json

rt = boto3.client('bedrock-runtime', region_name='us-east-1')

MODELS_TO_TRY = [
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
]

accessible = []
for model_id in MODELS_TO_TRY:
    try:
        resp = rt.invoke_model(
            modelId=model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hi"}]
            }),
            contentType='application/json',
            accept='application/json'
        )
        body = json.loads(resp['body'].read())
        print(f"  ACCESSIBLE: {model_id}")
        accessible.append(model_id)
    except Exception as e:
        print(f"  BLOCKED: {model_id}")
        print(f"    Error: {e}")

print(f"\nAccessible: {accessible}")

# Also check what the account has enabled via model access
print("\nChecking model access list...")
bc = boto3.client('bedrock', region_name='us-east-1')
try:
    access = bc.list_foundation_model_agreement_offers()
    print("Agreement offers:", access)
except Exception as e:
    print(f"list_foundation_model_agreement_offers: {e}")

# Try get_foundation_model_availability
try:
    for mid in ["anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-5-haiku-20241022-v1:0"]:
        avail = bc.get_foundation_model(modelIdentifier=mid)
        print(f"{mid}: {avail['modelDetails'].get('modelLifecycle', {})}")
except Exception as e:
    print(f"get_foundation_model: {e}")
