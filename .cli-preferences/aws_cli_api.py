from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import yaml
import subprocess
from datetime import datetime
from collections import Counter

app = FastAPI(title="AWS CLI Command API")

# Pydantic Models
class CommandExecute(BaseModel):
    command: str
    parameters: Optional[Dict] = None

class CommandResponse(BaseModel):
    command: str
    output: str
    execution_time: float
    timestamp: datetime

# Command Tracking
command_history: List[CommandResponse] = []
command_frequency = Counter()

def load_preferences():
    with open('aws.yaml', 'r') as file:
        return yaml.safe_load(file)

def update_command_frequency(command: str):
    command_frequency[command] += 1
    # Update the YAML file
    prefs = load_preferences()
    for category in prefs['command_categories'].values():
        for cmd in category['frequently_used']:
            if cmd['command'] == command:
                cmd['frequency'] += 1
    
    with open('aws.yaml', 'w') as file:
        yaml.dump(prefs, file)

async def execute_aws_command(command: str, parameters: Optional[Dict] = None) -> CommandResponse:
    try:
        full_command = command
        if parameters:
            for key, value in parameters.items():
                full_command = full_command.replace(f"{{{key}}}", str(value))
        
        start_time = datetime.now()
        result = subprocess.run(
            full_command.split(),
            capture_output=True,
            text=True,
            check=True
        )
        execution_time = (datetime.now() - start_time).total_seconds()
        
        update_command_frequency(command)
        
        response = CommandResponse(
            command=full_command,
            output=result.stdout,
            execution_time=execution_time,
            timestamp=start_time
        )
        command_history.append(response)
        return response
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=f"Command execution failed: {e.stderr}")

# API Routes
@app.post("/aws/s3/list-buckets")
async def list_buckets():
    return await execute_aws_command("aws s3 ls")

@app.post("/aws/s3/sync")
async def sync_s3(source: str, destination: str):
    return await execute_aws_command(
        "aws s3 sync {source} {destination}",
        {"source": source, "destination": destination}
    )

@app.post("/aws/lambda/list")
async def list_lambda_functions():
    return await execute_aws_command("aws lambda list-functions")

@app.post("/aws/lambda/invoke/{function_name}")
async def invoke_lambda(function_name: str):
    return await execute_aws_command(
        "aws lambda invoke --function-name {function_name} output.json",
        {"function_name": function_name}
    )

@app.get("/aws/iam/users")
async def list_iam_users():
    return await execute_aws_command("aws iam list-users")

@app.get("/aws/ec2/instances")
async def list_ec2_instances():
    return await execute_aws_command("aws ec2 describe-instances")

# Analytics Routes
@app.get("/analytics/frequency")
async def get_command_frequency():
    return {"command_frequency": dict(command_frequency)}

@app.get("/analytics/history")
async def get_command_history():
    return {"command_history": command_history}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 