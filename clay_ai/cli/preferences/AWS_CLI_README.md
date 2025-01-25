# AWS CLI Command Management System

## Overview
This system provides a structured approach to managing AWS CLI commands with:
- Organized command hierarchy
- Frequency tracking
- FastAPI integration
- Lambda deployment automation

## Quick Start
1. Ensure AWS CLI is installed and configured:
```bash
aws --version
aws configure
```

2. Verify connection:
```bash
aws sts get-caller-identity
```

## Command Categories

### High Priority Commands
#### S3 Operations
- List buckets: `aws s3 ls`
- Sync upload: `aws s3 sync . s3://bucket-name`
- Sync download: `aws s3 sync s3://bucket-name .`

#### Lambda Operations
- List functions: `aws lambda list-functions`
- Invoke function: `aws lambda invoke --function-name FUNCTION_NAME output.json`
- Update function: `aws lambda update-function-code --function-name FUNCTION_NAME --zip-file fileb://function.zip`

### Medium Priority Commands
#### IAM Operations
- List users: `aws iam list-users`
- List roles: `aws iam list-roles`

#### EC2 Operations
- List instances: `aws ec2 describe-instances`
- Start instance: `aws ec2 start-instances --instance-ids INSTANCE_ID`

## FastAPI Integration
The system includes a FastAPI application that wraps commonly used AWS CLI commands:

### API Endpoints
- `/aws/s3`: S3 operations
- `/aws/lambda`: Lambda function management
- `/aws/iam`: IAM operations
- `/aws/ec2`: EC2 instance management

### Command Frequency Tracking
- All commands are tracked for usage frequency
- Analytics available through monitoring endpoints
- Automatic logging of command history

## Automation Scripts
1. **Bucket Backup**
   - Daily automated S3 bucket backups
   - Configurable backup locations

2. **Lambda Deployment**
   - Streamlined function deployment
   - Version control integration

## Best Practices
1. Always use named profiles for different AWS accounts
2. Implement least privilege access
3. Use MFA for sensitive operations
4. Regular rotation of access keys

## Monitoring
- Command history tracking
- Usage frequency analytics
- Error logging
- Performance metrics

## Troubleshooting
1. Check AWS credentials: `aws configure list`
2. Verify region setting: `aws configure get region`
3. Test connectivity: `aws sts get-caller-identity` 