#!/bin/bash
set -e
exec > /var/log/mediclaims-setup.log 2>&1

echo "=== MediClaims AI Setup Starting ==="
date

# Update and install dependencies
yum update -y
yum install -y python3 python3-pip git nginx unzip

# Install Python packages
pip3 install --upgrade pip
pip3 install flask flask-cors fastapi uvicorn[standard] langgraph langchain \
    langchain-openai openai python-dotenv httpx pydantic pandas numpy \
    requests aiohttp fpdf boto3 botocore waitress jinja2

# Create app directory
mkdir -p /opt/mediclaims
cd /opt/mediclaims

# Download app code from S3
aws s3 cp s3://alpha-claims-demo-390783052961/app/mediclaims.zip /opt/mediclaims/mediclaims.zip --region us-east-1
unzip -o mediclaims.zip -d /opt/mediclaims/
cd /opt/mediclaims/alpha

# Write .env file with all credentials
cat > /opt/mediclaims/alpha/.env << 'ENVEOF'
AWS_ACCESS_KEY_ID=AKIAVV7D4FSQ2JNOUB5J
AWS_SECRET_ACCESS_KEY=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
AWS_DEFAULT_REGION=us-east-1
AWS_ACCOUNT_ID=390783052961
AWS_BEDROCK_MODEL_ID=us.amazon.nova-micro-v1:0
S3_BUCKET_NAME=alpha-claims-demo-390783052961
BEDROCK_AGENT_RISK_PREDICTOR=XLAYW801JO
BEDROCK_AGENT_APPEAL_GENERATOR=S4YKZVC69F
BEDROCK_AGENT_AUTO_CORRECTOR=53KNQP1PMD
BEDROCK_AGENT_CLAIM_SUBMITTER=BSI3CF17OU
BEDROCK_AGENT_RESUBMITTER=VCKGXKAZN0
BEDROCK_AGENT_FEEDBACK_LEARNER=WUKCB3RG8Z
BEDROCK_AGENT_ALIAS=TSTALIASID
BEDROCK_AGENTS_ROLE_ARN=arn:aws:iam::390783052961:role/BedrockAgentsClaimsRole
BEDROCK_AGENT_ALIAS_RISK_PREDICTOR=GMML5M64KW
BEDROCK_AGENT_ALIAS_AUTO_CORRECTOR=FYFALB8NSI
BEDROCK_AGENT_ALIAS_CLAIM_SUBMITTER=BZBIFWBIBO
BEDROCK_AGENT_ALIAS_APPEAL_GENERATOR=F7V98MQXIZ
BEDROCK_AGENT_ALIAS_RESUBMITTER=EWC6FLUFBK
BEDROCK_AGENT_ALIAS_FEEDBACK_LEARNER=U9I9LCMJIL
OPERATIONAL_MODE=mcp
DATA_SOURCE=openemr
MCP_API_ACCESS_TOKEN=dev-access-token-123
RISK_THRESHOLD=0.4
LOG_LEVEL=INFO
TIMEOUT=15
ENABLE_LOG_REDACTION=true
ENVEOF

# Configure nginx as reverse proxy
cat > /etc/nginx/conf.d/mediclaims.conf << 'NGINXEOF'
server {
    listen 80;
    server_name _;

    # Pre-submission dashboard (Flask :5000) — main app
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
    }

    # Post-submission appeals dashboard (FastAPI :8003)
    location /appeals/ {
        proxy_pass http://127.0.0.1:8003/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 300;
    }

    # Direct API access for post-submission
    location /api/appeals/ {
        proxy_pass http://127.0.0.1:8003/api/appeals/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
NGINXEOF

# Remove default nginx config
rm -f /etc/nginx/conf.d/default.conf

# Create systemd service for pre-submission (Flask)
cat > /etc/systemd/system/mediclaims-pre.service << 'SVCEOF'
[Unit]
Description=MediClaims Pre-Submission Dashboard
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/mediclaims/alpha
Environment=PYTHONPATH=/opt/mediclaims/alpha
ExecStart=/usr/bin/python3 web_dashboard/api_server.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SVCEOF

# Create systemd service for post-submission (FastAPI)
cat > /etc/systemd/system/mediclaims-post.service << 'SVCEOF'
[Unit]
Description=MediClaims Post-Submission Appeals Dashboard
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/mediclaims/alpha/post_submission_demo
Environment=PYTHONPATH=/opt/mediclaims/alpha
ExecStart=/usr/bin/python3 -m uvicorn app:app --host 0.0.0.0 --port 8003
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SVCEOF

# Create systemd service for MCP server
cat > /etc/systemd/system/mediclaims-mcp.service << 'SVCEOF'
[Unit]
Description=MediClaims MCP Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/mediclaims/alpha
Environment=PYTHONPATH=/opt/mediclaims/alpha
ExecStart=/usr/bin/python3 mcp_server/main.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SVCEOF

# Create systemd service for insurer APIs
cat > /etc/systemd/system/mediclaims-insurer-primary.service << 'SVCEOF'
[Unit]
Description=MediClaims Insurer API Primary
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/mediclaims/alpha
Environment=PYTHONPATH=/opt/mediclaims/alpha
ExecStart=/usr/bin/python3 tools/insurer_api_primary.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SVCEOF

cat > /etc/systemd/system/mediclaims-insurer-secondary.service << 'SVCEOF'
[Unit]
Description=MediClaims Insurer API Secondary
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/mediclaims/alpha
Environment=PYTHONPATH=/opt/mediclaims/alpha
ExecStart=/usr/bin/python3 tools/insurer_api_secondary.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SVCEOF

# Enable and start all services
systemctl daemon-reload
systemctl enable nginx mediclaims-pre mediclaims-post mediclaims-mcp mediclaims-insurer-primary mediclaims-insurer-secondary
systemctl start mediclaims-mcp
sleep 3
systemctl start mediclaims-insurer-primary mediclaims-insurer-secondary
sleep 3
systemctl start mediclaims-pre mediclaims-post
sleep 5
systemctl start nginx

echo "=== MediClaims AI Setup Complete ==="
date
systemctl status mediclaims-pre --no-pager
systemctl status mediclaims-post --no-pager
