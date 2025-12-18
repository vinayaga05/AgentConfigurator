from typing import Any, List, Dict, Optional
from typing_extensions import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import json
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Look for .env in the agent directory (parent of sample_agent)
agent_dir = Path(__file__).parent.parent
env_path = agent_dir / ".env"
load_dotenv(dotenv_path=env_path, override=False)

# Set up logging
logger = logging.getLogger(__name__)

# Lazy import AgentConfigurationAssistant to avoid dependency issues at module load time
# The import will happen when tools are actually called

# Model configuration - can be set via environment variable
# Default: "gemini" (switch to "ollama" in .env to use Ollama)
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "gemini").lower()
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.7"))

class AgentState(MessagesState):
    """
    Here we define the state of the agent

    In this instance, we're inheriting from CopilotKitState, which will bring in
    the CopilotKitState fields. We're also adding a custom field, `language`,
    which will be used to set the language of the agent.
    """
    proverbs: List[str] = []
    tools: List[Any]
    agent_analysis: Dict[str, Any] = {}
    agent_config_assistant: Optional[Any] = None  # Type is Any to avoid import at module load time
    model_provider: str = MODEL_PROVIDER  # Model provider: "gemini" or "ollama"
    model_name: str = MODEL_NAME  # Model name for the selected provider
    # your_custom_agent_state: str = ""

def _get_assistant_class():
    """Lazy import AgentConfigurationAssistant to avoid dependency issues."""
    from sample_agent.agent_configuration import AgentConfigurationAssistant
    return AgentConfigurationAssistant


def get_model_config_from_state(state: AgentState, temperature: Optional[float] = None) -> Dict[str, Any]:
    """
    Get common model configuration from state with defaults.
    
    Args:
        state: The agent state containing model configuration
        temperature: Optional temperature override. If None, uses DEFAULT_TEMPERATURE
        
    Returns:
        Dictionary with provider, model_name, and temperature
    """
    provider = state.get("model_provider", MODEL_PROVIDER)
    
    # Get model name from state, or use provider-specific default
    model_name = state.get("model_name")
    if model_name is None:
        # Use provider-specific default
        if provider == "ollama":
            model_name = OLLAMA_MODEL
        else:
            model_name = MODEL_NAME
    
    # Use provided temperature or default
    if temperature is None:
        temperature = DEFAULT_TEMPERATURE
    
    return {
        "provider": provider,
        "model_name": model_name,
        "temperature": temperature
    }


def get_llm_model(provider: Optional[str] = None, model_name: Optional[str] = None, temperature: float = 0.7):
    """Get LLM model instance based on provider (gemini or ollama)."""
    provider = (provider or MODEL_PROVIDER).lower()
    
    if provider == "gemini":
        try:
            return ChatGoogleGenerativeAI(
                model=model_name or MODEL_NAME,
                temperature=temperature
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise
    
    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=model_name or OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=temperature
            )
        except ImportError:
            raise ImportError("langchain-ollama not installed. Install with: pip install langchain-ollama")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            raise
    
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'gemini' or 'ollama'")

def get_assistant_from_state(state: AgentState, config: Optional[RunnableConfig] = None):
    """Get or create AgentConfigurationAssistant instance from state."""
    AgentConfigurationAssistant = _get_assistant_class()
    
    # Check for existing assistant with correct model
    existing = state.get("agent_config_assistant")
    model_config = get_model_config_from_state(state)
    model_name = model_config["model_name"]
    
    if existing and getattr(existing, 'model_name', None) == model_name:
        return existing
    
    # Create new assistant
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")
    
    assistant = AgentConfigurationAssistant(model=model_name, thread_id=thread_id)
    state["agent_config_assistant"] = assistant
    return assistant


@tool
async def analyze_agent_requirements(agent_description: str) -> str:
    """
    Analyze user input to extract agent role, responsibilities, and requirements.
    
    This tool analyzes an agent description to identify:
    - The purpose of the agent
    - Core responsibilities
    - Required user information
    - Expected workflow steps
    - Whether clarification is needed
    
    Args:
        agent_description: User's description of the agent they want to create
        
    Returns:
        JSON string with analysis including purpose, role, responsibility, 
        core_responsibilities, expected_workflow, needs_clarification, and requirement_clarity_score
    """
    try:
        # Lazy import to avoid dependency issues at module load time
        AgentConfigurationAssistant = _get_assistant_class()
        # Note: Tools don't have direct access to state, so we read model from environment
        # Get model name from environment (for Gemini) or use default
        model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")
        assistant = AgentConfigurationAssistant(model=model_name)
        
        # Call the analysis method
        result = await assistant.analyze_agent_role_and_responsibility(agent_description)
        
        # Return as JSON string
        return json.dumps(result, indent=2)
    except Exception as e:
        error_result = {
            "error": str(e),
            "purpose": "",
            "role": "",
            "responsibility": "",
            "core_responsibilities": [],
            "required_user_information": [],
            "expected_workflow": [],
            "analysis_summary": f"Error during analysis: {str(e)}",
            "needs_clarification": True,
            "requirement_clarity_score": 0.0
        }
        return json.dumps(error_result)


@tool
async def search_tools_by_categories(categories: str, user_input: str = "", limit: int = 30) -> str:
    """
    Search for tools from vector database based on categories and user input.
    
    Args:
        categories: JSON string with tool categories (primary_categories, required_tools, keywords, etc.)
        user_input: User's original request for context (optional)
        limit: Maximum number of tools to return (default: 30)
        
    Returns:
        JSON string with list of tools matching the categories
    """
    try:
        # Lazy import to avoid dependency issues at module load time
        AgentConfigurationAssistant = _get_assistant_class()
        # Get model name from environment (for Gemini) or use default
        model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")
        assistant = AgentConfigurationAssistant(model=model_name)
        
        # Parse categories JSON
        categories_dict = json.loads(categories) if isinstance(categories, str) else categories
        
        # Call the search method
        result = assistant.get_tools_from_categories(categories_dict, limit=limit, user_input=user_input)
        
        # Extract tools list from result
        tools_list = result.get("tools_list", [])
        
        return json.dumps(tools_list, indent=2)
    except Exception as e:
        error_result = {
            "error": str(e),
            "tools": []
        }
        return json.dumps(error_result)


@tool
async def recommend_tools_for_agent(user_input: str, current_instruction: str, available_tools: str) -> str:
    """
    Use semantic search to recommend the most relevant tools for an agent configuration.
    
    This tool analyzes the user's request and available tools to recommend the best matches.
    
    Args:
        user_input: User's original request describing what they want the agent to do
        current_instruction: JSON string of current agent instruction (role, responsibility, process_steps)
        available_tools: JSON string of available tools list to choose from
        
    Returns:
        JSON string with recommended_tools, can_fulfill, missing_capabilities, confidence_score
    """
    try:
        # Lazy import to avoid dependency issues at module load time
        AgentConfigurationAssistant = _get_assistant_class()
        # Get model name from environment (for Gemini) or use default
        model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")
        assistant = AgentConfigurationAssistant(model=model_name)
        
        # Parse JSON inputs
        instruction_dict = json.loads(current_instruction) if isinstance(current_instruction, str) else current_instruction
        tools_list = json.loads(available_tools) if isinstance(available_tools, str) else available_tools
        
        # Call the semantic search method
        result = await assistant.semantic_search_from_filtered_tools(
            user_input=user_input,
            current_instruction=instruction_dict,
            tools=tools_list
        )
        
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        error_result = {
            "error": str(e),
            "can_fulfill": False,
            "recommended_tools": [],
            "missing_capabilities": [f"Error during tool recommendation: {str(e)}"],
            "overall_confidence": 0.0,
            "alternative_approach": "Unable to recommend tools due to error"
        }
        return json.dumps(error_result)


@tool
async def generate_agent_instruction(current_instruction: str, recommended_tools: str, message_type: str = "agent_creation") -> str:
    """
    Generate comprehensive agent instruction combining role with suggested tools.
    
    This tool creates a complete agent instruction that includes:
    - Role and responsibility
    - Process steps
    - Tool-specific steps
    - Integration of recommended tools
    
    Args:
        current_instruction: JSON string of current instruction (role, responsibility, process_steps)
        recommended_tools: JSON string of recommended tools list
        message_type: Type of message (default: "agent_creation", can be "modification")
        
    Returns:
        JSON string with enhanced instruction including role, responsibility, process_steps, tool_steps
    """
    try:
        # Lazy import to avoid dependency issues at module load time
        AgentConfigurationAssistant = _get_assistant_class()
        # Get model name from environment (for Gemini) or use default
        model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")
        assistant = AgentConfigurationAssistant(model=model_name)
        
        # Parse JSON inputs
        instruction_dict = json.loads(current_instruction) if isinstance(current_instruction, str) else current_instruction
        tools_list = json.loads(recommended_tools) if isinstance(recommended_tools, str) else recommended_tools
        
        # Call the instruction generation method
        result = await assistant.get_instruction(
            current_instruction=instruction_dict,
            recommended_tools=tools_list,
            message_type=message_type
        )
        
        # Convert result to dict if it's a Pydantic model
        if hasattr(result, 'model_dump'):
            result_dict = result.model_dump()
        elif hasattr(result, 'dict'):
            result_dict = result.dict()
        else:
            result_dict = result
        
        return json.dumps(result_dict, indent=2, default=str)
    except Exception as e:
        error_result = {
            "error": str(e),
            "role": "",
            "responsibility": "",
            "process_title": "Main Process",
            "process_steps": [],
            "tool_usage_title": None,
            "tool_steps": []
        }
        return json.dumps(error_result)


@tool
async def generate_clarification_questions(analysis_result: str) -> str:
    """
    Generate questions to clarify missing agent configuration details.
    
    This tool takes an analysis result and generates specific questions to gather
    missing information needed to fully configure the agent.
    
    Args:
        analysis_result: JSON string from analyze_agent_requirements containing
                        purpose, role, responsibilities, expected_workflow, etc.
        
    Returns:
        JSON string with questions list and success status
    """
    try:
        # Lazy import to avoid dependency issues at module load time
        AgentConfigurationAssistant = _get_assistant_class()
        # Get model name from environment (for Gemini) or use default
        model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")
        assistant = AgentConfigurationAssistant(model=model_name)
        
        # Parse analysis result JSON
        analysis_dict = json.loads(analysis_result) if isinstance(analysis_result, str) else analysis_result
        
        # Call the question generation method
        result = await assistant.generate_questions(analysis_dict)
        
        return json.dumps(result, indent=2)
    except Exception as e:
        error_result = {
            "error": str(e),
            "questions": [],
            "success": False
        }
        return json.dumps(error_result)


# @tool
# def your_tool_here(your_arg: str):
#     """Your tool description here."""
#     print(f"Your tool logic here")
#     return "Your tool response here."

tools = [
    analyze_agent_requirements,
    search_tools_by_categories,
    recommend_tools_for_agent,
    generate_agent_instruction,
    generate_clarification_questions
    # your_tool_here
]


async def chat_node(state: AgentState, config: RunnableConfig) -> Command[Literal["tool_node", "__end__"]]:
    """
    Standard chat node based on the ReAct design pattern. It handles:
    - The model to use (and binds in CopilotKit actions and the tools defined above)
    - The system prompt
    - Getting a response from the model
    - Handling tool calls

    For more about the ReAct design pattern, see:
    https://www.perplexity.ai/search/react-agents-NcXLQhreS0WDzpVaS4m9Cg
    """
    try:
        logger.info("Chat node: Starting processing")

            # 1. Define the model using common configuration
        model_config = get_model_config_from_state(state, temperature=DEFAULT_TEMPERATURE)
        provider = model_config["provider"]
        model_name = model_config["model_name"]
        model = get_llm_model(
            provider=provider,
            model_name=model_name,
            temperature=model_config["temperature"]
        )

            # 2. Collect tools, avoiding duplicates
        # Frontend actions (like addProverb, setThemeColor) are handled by CopilotKit
        # and shouldn't be bound as backend tools to avoid duplicate function declarations with Gemini
        ag_ui_tools = state.get("tools", [])
        backend_tools = [
            analyze_agent_requirements,
            search_tools_by_categories,
            recommend_tools_for_agent,
            generate_agent_instruction,
            generate_clarification_questions
        ]
        
        # Frontend-only actions that should not be bound as model tools
        # These are handled by CopilotKit runtime on the frontend
        frontend_only_actions = {"addProverb", "setThemeColor"}
        
        # Create a set of tool names to avoid duplicates
        tool_names = set()
        unique_tools = []
        
        # Helper function to get tool name
        def get_tool_name(tool):
            """Extract the name of a tool."""
            if hasattr(tool, 'name'):
                return tool.name
            elif hasattr(tool, '__name__'):
                return tool.__name__
            else:
                return str(tool)
        
        # Add backend tools first
        for tool in backend_tools:
            tool_name = get_tool_name(tool)
            if tool_name not in tool_names:
                tool_names.add(tool_name)
                unique_tools.append(tool)
        
        # Add ag-ui tools, but skip frontend-only actions that might conflict
        # Frontend actions are handled by CopilotKit runtime, not as model tools
        for tool in ag_ui_tools:
            tool_name = get_tool_name(tool)
            # Skip frontend actions that are handled by CopilotKit
            if tool_name not in frontend_only_actions and tool_name not in tool_names:
                tool_names.add(tool_name)
                unique_tools.append(tool)

        # 3. Bind the tools to the model
        # Ollama's ChatOllama doesn't support parallel_tool_calls parameter
        # Use common provider from model_config to determine binding method
        if provider == "ollama":
            # Ollama handles tools differently - bind without parallel_tool_calls
            model_with_tools = model.bind_tools(unique_tools)
        else:
            # Gemini and other models support parallel_tool_calls
            model_with_tools = model.bind_tools(
                unique_tools,
                # 3.1 Disable parallel tool calls to avoid race conditions,
                #     enable this for faster performance if you want to manage
                #     the complexity of running tool calls in parallel.
                parallel_tool_calls=False,
            )

        # 4. Define the system message by which the chat model will be run
        # Ensure proverbs is always a list
        proverbs = state.get('proverbs', [])
        if not isinstance(proverbs, list):
            proverbs = []
        
        # Enhanced system message with agent configuration capabilities
        system_content = f"""You are a helpful assistant specialized in configuring AI agents. The current proverbs are {proverbs}.

You have access to powerful agent configuration tools that allow you to:
1. Analyze agent requirements - Use analyze_agent_requirements to understand what the user wants their agent to do
2. Search for tools - Use search_tools_by_categories to find relevant tools from the database
3. Recommend tools - Use recommend_tools_for_agent to intelligently select the best tools for an agent
4. Generate instructions - Use generate_agent_instruction to create comprehensive agent configurations
5. Ask clarifying questions - Use generate_clarification_questions when you need more information

When a user wants to create or configure an agent:
- First analyze their requirements using analyze_agent_requirements
- If clarification is needed, use generate_clarification_questions
- Search for relevant tools using search_tools_by_categories or recommend_tools_for_agent
- Generate the final agent instruction using generate_agent_instruction

Always provide clear, helpful responses and guide users through the agent configuration process."""
        
        system_message = SystemMessage(content=system_content)

        # 5. Run the model to generate a response
        try:
            logger.info(f"Chat node: Invoking model ({provider}, {model_name}) with {len(state.get('messages', []))} messages")
            response = await model_with_tools.ainvoke([system_message, *state["messages"]], config)
            logger.info(f"Chat node: Model response received, has tool_calls: {hasattr(response, 'tool_calls') and bool(response.tool_calls)}")
        except Exception as e:
            from langchain_core.messages import AIMessage
            import traceback
            error_msg = str(e)
            error_trace = traceback.format_exc()
            logger.error(f"Chat node error ({provider}): {error_msg}")
            logger.error(f"Traceback: {error_trace}")
            
            is_connection_error = "connection" in error_msg.lower() or "connect" in error_msg.lower()
            if is_connection_error and provider == "ollama":
                response = AIMessage(content=f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. Start it with: ollama serve")
            elif "api key" in error_msg.lower() or "credentials" in error_msg.lower():
                response = AIMessage(content=f"API key error. Please check your GOOGLE_API_KEY in .env file.")
            else:
                response = AIMessage(content=f"Error: {error_msg}")

        # 6. Check if the response contains tool calls
        # The response from ainvoke is typically a single AIMessage
        # When tools are bound, tool calls are in the tool_calls attribute
        has_tool_calls = False
        if hasattr(response, 'tool_calls') and response.tool_calls and len(response.tool_calls) > 0:
            has_tool_calls = True
        elif isinstance(response, list):
            # Handle case where response is a list of messages
            for msg in response:
                if hasattr(msg, 'tool_calls') and msg.tool_calls and len(msg.tool_calls) > 0:
                    has_tool_calls = True
                    break

        # 7. Route based on whether there are tool calls
        # Ensure response is in the correct format for state update
        if isinstance(response, list):
            messages_update = response
        else:
            messages_update = [response]

        if has_tool_calls:
            # Route to tool_node to execute the tool calls
            logger.info("Chat node: Routing to tool_node")
            return Command(
                goto="tool_node",
                update={
                    "messages": messages_update
                }
            )
        else:
            # No tool calls, end the graph
            logger.info("Chat node: No tool calls, ending graph")
            return Command(
                goto=END,
                update={
                    "messages": messages_update
                }
            )
    except Exception as e:
        import traceback
        logger.error(f"Unexpected error in chat_node: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        from langchain_core.messages import AIMessage
        return Command(
            goto=END,
            update={
                "messages": [AIMessage(content=f"An error occurred: {str(e)}")]
            }
        )

# Define the workflow graph
workflow = StateGraph(AgentState)
workflow.add_node("chat_node", chat_node)
workflow.add_node("tool_node", ToolNode(tools=tools))
workflow.add_edge("tool_node", "chat_node")
workflow.set_entry_point("chat_node")

# Use checkpointer for local FastAPI development
# LangGraph API will ignore this checkpointer and use its own persistence when deployed
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
