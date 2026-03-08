#!/bin/bash
exec > /var/log/mediclaims-setup.log 2>&1

echo "=== MediClaims AI Setup Starting ==="
date

# Install nginx if not already running
amazon-linux-extras install nginx1 -y 2>/dev/null || true
yum install -y unzip 2>/dev/null || true

# Python 3.8 is already installed at /usr/bin/python3.8
PYTHON=/usr/bin/python3.8
PIP="$PYTHON -m pip"
echo "Python: $($PYTHON --version)"

# Upgrade pip
$PIP install --upgrade pip --quiet

# Install all required packages
$PIP install --quiet \
    "flask==2.2.5" \
    "flask-cors==4.0.0" \
    "fastapi==0.103.2" \
    "uvicorn[standard]==0.23.2" \
    "langgraph==0.0.40" \
    "langchain==0.1.20" \
    "langchain-core==0.1.52" \
    "langchain-community==0.0.38" \
    "python-dotenv==1.0.0" \
    "httpx==0.24.1" \
    "pydantic==1.10.13" \
    "pandas==2.0.3" \
    "numpy==1.24.4" \
    "requests==2.31.0" \
    "aiohttp==3.9.5" \
    "fpdf2==2.7.9" \
    "boto3==1.34.69" \
    "botocore==1.34.69" \
    "waitress==3.0.0" \
    "jinja2==3.1.4" \
    "python-multipart==0.0.9" \
    "aiofiles==23.2.1"

echo "=== Packages installed ==="

# Create and populate app directory
mkdir -p /opt/mediclaims
cd /opt/mediclaims

aws s3 cp s3://alpha-claims-demo-390783052961/app/mediclaims.zip /opt/mediclaims/mediclaims.zip --region us-east-1
unzip -o mediclaims.zip -d /opt/mediclaims/

APP_DIR=/opt/mediclaims
echo "APP_DIR=$APP_DIR"

# Write .env file
cat > "$APP_DIR/.env" << 'ENVEOF'
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

# Configure nginx
cat > /etc/nginx/nginx.conf << 'NGINXEOF'
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log;
pid /run/nginx.pid;

events {
    worker_connections 1024;
}

http {
    log_format main '$remote_addr - $remote_user [$time_local] "$request" $status $body_bytes_sent';
    access_log /var/log/nginx/access.log main;
    sendfile on;
    keepalive_timeout 65;
    client_max_body_size 50M;

    server {
        listen 80;
        server_name _;

        location /appeals/ {
            proxy_pass http://127.0.0.1:8003/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_read_timeout 300;
            proxy_connect_timeout 300;
        }

        location /post/ {
            proxy_pass http://127.0.0.1:8003/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_read_timeout 300;
        }

        location / {
            proxy_pass http://127.0.0.1:5000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_read_timeout 300;
            proxy_connect_timeout 300;
        }
    }
}
NGINXEOF

# Systemd: pre-submission Flask
cat > /etc/systemd/system/mediclaims-pre.service << SVCEOF
[Unit]
Description=MediClaims Pre-Submission Dashboard
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/mediclaims
Environment=PYTHONPATH=/opt/mediclaims
ExecStart=/usr/bin/python3.8 /opt/mediclaims/web_dashboard/api_server.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SVCEOF

# Systemd: post-submission FastAPI
cat > /etc/systemd/system/mediclaims-post.service << SVCEOF
[Unit]
Description=MediClaims Post-Submission Appeals Dashboard
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/mediclaims/post_submission_demo
Environment=PYTHONPATH=/opt/mediclaims
ExecStart=/usr/bin/python3.8 -m uvicorn app:app --host 0.0.0.0 --port 8003
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SVCEOF

# Systemd: MCP server
cat > /etc/systemd/system/mediclaims-mcp.service << SVCEOF
[Unit]
Description=MediClaims MCP Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/mediclaims
Environment=PYTHONPATH=/opt/mediclaims
ExecStart=/usr/bin/python3.8 /opt/mediclaims/mcp_server/main.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SVCEOF

# Systemd: Insurer APIs
cat > /etc/systemd/system/mediclaims-insurer-primary.service << SVCEOF
[Unit]
Description=MediClaims Insurer API Primary
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/mediclaims
Environment=PYTHONPATH=/opt/mediclaims
ExecStart=/usr/bin/python3.8 /opt/mediclaims/tools/insurer_api_primary.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SVCEOF

cat > /etc/systemd/system/mediclaims-insurer-secondary.service << SVCEOF
[Unit]
Description=MediClaims Insurer API Secondary
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/mediclaims
Environment=PYTHONPATH=/opt/mediclaims
ExecStart=/usr/bin/python3.8 /opt/mediclaims/tools/insurer_api_secondary.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SVCEOF

# Enable and start services
systemctl daemon-reload
systemctl enable nginx mediclaims-pre mediclaims-post mediclaims-mcp mediclaims-insurer-primary mediclaims-insurer-secondary

systemctl start mediclaims-mcp || true
sleep 3
systemctl start mediclaims-insurer-primary mediclaims-insurer-secondary || true
sleep 3
systemctl start mediclaims-pre mediclaims-post
sleep 5
systemctl start nginx

echo "=== Setup Complete ==="
date
systemctl status mediclaims-pre --no-pager || true
systemctl status mediclaims-post --no-pager || true
systemctl status nginx --no-pager || true
