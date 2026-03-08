"""
Fix BedrockAgentsClaimsRole to allow invoking Claude 3.5 Haiku
via cross-region inference profile.

The agent role needs:
  - bedrock:InvokeModel on the inference profile ARN
  - bedrock:InvokeModelWithResponseStream on the inference profile ARN
  - bedrock:GetInferenceProfile (to resolve the profile)
  - bedrock:InvokeModel on the underlying model ARN (us-east-1 + us-west-2)
"""
import boto3, json

iam = boto3.client('iam', region_name='us-east-1')
ROLE = 'BedrockAgentsClaimsRole'
ACCOUNT = '390783052961'

policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            # Allow invoking any Bedrock model or inference profile
            "Sid": "BedrockInvokeAll",
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
                "bedrock:InvokeAgent",
            ],
            "Resource": "*"
        },
        {
            # Allow resolving and using inference profiles
            "Sid": "BedrockInferenceProfiles",
            "Effect": "Allow",
            "Action": [
                "bedrock:GetInferenceProfile",
                "bedrock:ListInferenceProfiles",
                "bedrock:GetFoundationModel",
                "bedrock:ListFoundationModels",
            ],
            "Resource": "*"
        },
        {
            # S3 access for claims data
            "Sid": "S3ClaimsData",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket",
                "s3:DeleteObject"
            ],
            "Resource": [
                f"arn:aws:s3:::alpha-claims-demo-{ACCOUNT}",
                f"arn:aws:s3:::alpha-claims-demo-{ACCOUNT}/*"
            ]
        },
        {
            # Lambda action groups
            "Sid": "LambdaActionGroups",
            "Effect": "Allow",
            "Action": ["lambda:InvokeFunction"],
            "Resource": f"arn:aws:lambda:us-east-1:{ACCOUNT}:function:mediclaims-*"
        },
        {
            # CloudWatch Logs
            "Sid": "CloudWatchLogs",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:*:*:*"
        }
    ]
}

print(f"Updating inline policy 'BedrockS3Access' on role {ROLE}...")
iam.put_role_policy(
    RoleName=ROLE,
    PolicyName='BedrockS3Access',
    PolicyDocument=json.dumps(policy)
)
print("✓ Policy updated")

# Verify
doc = iam.get_role_policy(RoleName=ROLE, PolicyName='BedrockS3Access')['PolicyDocument']
print("\nNew policy:")
print(json.dumps(doc, indent=2))
