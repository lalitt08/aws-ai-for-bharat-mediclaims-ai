"""Check available inference profiles and test them."""
import boto3, json

bc = boto3.client('bedrock', region_name='us-east-1')
rt = boto3.client('bedrock-runtime', region_name='us-east-1')

# List inference profiles
try:
    profiles = bc.list_inference_profiles()
    print("Inference profiles:")
    for p in profiles.get('inferenceProfileSummaries', []):
        print(f"  {p['inferenceProfileId']} - {p['inferenceProfileName']} - {p.get('status', 'unknown')}")
except Exception as e:
    print(f"list_inference_profiles: {e}")

# Try cross-region inference profile IDs (standard format)
PROFILE_IDS = [
    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "us.anthropic.claude-3-haiku-20240307-v1:0",
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
]

print("\nTesting inference profiles:")
accessible = []
for profile_id in PROFILE_IDS:
    try:
        resp = rt.invoke_model(
            modelId=profile_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 20,
                "messages": [{"role": "user", "content": "Say OK"}]
            }),
            contentType='application/json',
            accept='application/json'
        )
        body = json.loads(resp['body'].read())
        text = body.get('content', [{}])[0].get('text', '')
        print(f"  ACCESSIBLE: {profile_id} -> {text[:30]}")
        accessible.append(profile_id)
    except Exception as e:
        print(f"  BLOCKED: {profile_id}")
        print(f"    {str(e)[:120]}")

print(f"\nAccessible profiles: {accessible}")
