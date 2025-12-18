"""
This serves the "sample_agent" agent. This is an example of self-hosting an agent
through our FastAPI integration. However, you can also host in LangGraph platform.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to allow imports when running directly
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from dotenv import load_dotenv
load_dotenv() # pylint: disable=wrong-import-position

from fastapi import FastAPI
import uvicorn
from copilotkit import LangGraphAGUIAgent
from sample_agent.agent import graph
from ag_ui_langgraph import add_langgraph_fastapi_endpoint

app = FastAPI()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "langgraph-agent"}

add_langgraph_fastapi_endpoint(
    app=app,
    agent=LangGraphAGUIAgent(
        name="sample_agent",
        description="An example agent to use as a starting point for your own agent.",
        graph=graph
    ),
    path="/"
)

def main():
    """Run the uvicorn server."""
    port = int(os.getenv("PORT", "8123"))
    uvicorn.run(
        "sample_agent.demo:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )
