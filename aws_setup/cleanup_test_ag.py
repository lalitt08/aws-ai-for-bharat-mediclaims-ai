import boto3
bc = boto3.client('bedrock-agent', region_name='us-east-1')

# List all AGs on risk predictor agent to find the test one
ags = bc.list_agent_action_groups(agentId='XLAYW801JO', agentVersion='DRAFT').get('actionGroupSummaries', [])
print('Current AGs:', [(a['actionGroupName'], a['actionGroupId']) for a in ags])

for ag in ags:
    if ag['actionGroupName'] == 'TestAG':
        try:
            bc.update_agent_action_group(
                agentId='XLAYW801JO', agentVersion='DRAFT',
                actionGroupId=ag['actionGroupId'],
                actionGroupName='TestAG',
                actionGroupState='DISABLED',
                actionGroupExecutor={'lambda': 'arn:aws:lambda:us-east-1:390783052961:function:mediclaims-risk-predictor'},
                apiSchema={'payload': open('lambda/risk_predictor/openapi_schema.json').read()}
            )
            bc.delete_agent_action_group(
                agentId='XLAYW801JO', agentVersion='DRAFT',
                actionGroupId=ag['actionGroupId']
            )
            print('Deleted TestAG')
        except Exception as e:
            print(f'Could not delete: {e}')
