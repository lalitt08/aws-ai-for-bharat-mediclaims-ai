import boto3

# Check available models in us-east-1
bc = boto3.client('bedrock', region_name='us-east-1')
models = bc.list_foundation_models(byOutputModality='TEXT')['modelSummaries']

print("Claude models in us-east-1:")
for m in models:
    if 'claude' in m['modelId'].lower():
        status = m.get('modelLifecycle', {}).get('status', 'unknown')
        print(f"  {m['modelId']} - {status}")

print("\nAll models with 'haiku' or 'sonnet':")
for m in models:
    mid = m['modelId'].lower()
    if 'haiku' in mid or 'sonnet' in mid:
        print(f"  {m['modelId']}")

# Check model access
print("\nChecking model access for Claude 3 Haiku...")
try:
    rt = boto3.client('bedrock-runtime', region_name='us-east-1')
    import json
    resp = rt.invoke_model(
        modelId='anthropic.claude-3-haiku-20240307-v1:0',
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Hi"}]
        }),
        contentType='application/json',
        accept='application/json'
    )
    print("  Claude 3 Haiku: ACCESSIBLE")
except Exception as e:
    print(f"  Claude 3 Haiku: NOT ACCESSIBLE - {e}")
