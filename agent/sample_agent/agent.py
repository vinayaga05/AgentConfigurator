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
# Default: "gemini" (switch to "ollama" or "huggingface_hub" in .env to use different providers)
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "gemini").lower()
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "meta-llama/Llama-2-7b-chat-hf")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
DEFAULT_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.7"))

class AgentState(MessagesState):
    """
    Here we define the state of the agent

    In this instance, we're inheriting from CopilotKitState, which will bring in
    the CopilotKitState fields. We're also adding a custom field, `language`,
    which will be used to set the language of the agent.
    """
    proverbs: List[str] = []
    instruction: Optional[Dict[str, Any]] = None  # Current agent instruction (role, responsibility, process_steps, etc.)
    recommended_tools: List[Dict[str, Any]] = []  # Recommended tools for the agent
    tools: List[Any]
    agent_analysis: Dict[str, Any] = {}
    agent_config_assistant: Optional[Any] = None  # Type is Any to avoid import at module load time
    model_provider: str = MODEL_PROVIDER  # Model provider: "gemini", "ollama", or "huggingface_hub"
    model_name: str = MODEL_NAME  # Model name for the selected provider
    agent_config_state: Dict[str, Any] = {}  # Stores configuration workflow state
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
        elif provider == "huggingface_hub":
            model_name = HUGGINGFACE_MODEL
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
    """Get LLM model instance based on provider (gemini, ollama, or huggingface_hub)."""
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
    
    elif provider == "huggingface_hub":
        try:
            from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
            
            # Step 1: Create the HuggingFaceEndpoint with text-generation task
            # This uses the standard Hugging Face Inference API (not chat completion endpoint)
            endpoint = HuggingFaceEndpoint(
                repo_id=model_name or HUGGINGFACE_MODEL,
                task="text-generation",  # IMPORTANT: Use text-generation task
                temperature=temperature,
                huggingfacehub_api_token=HUGGINGFACE_API_KEY if HUGGINGFACE_API_KEY else None
            )
            
            # Step 2: Wrap it with ChatHuggingFace
            return ChatHuggingFace(llm=endpoint)
        except ImportError:
            raise ImportError("langchain-huggingface not installed. Install with: pip install langchain-huggingface")
        except Exception as e:
            logger.error(f"Failed to initialize Hugging Face Hub: {e}")
            raise
    
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'gemini', 'ollama', or 'huggingface_hub'")

def get_assistant_from_state(state: AgentState, config: Optional[RunnableConfig] = None):
    """Get or create AgentConfigurationAssistant instance from state."""
    AgentConfigurationAssistant = _get_assistant_class()
    
    # Check for existing assistant with correct model and provider
    existing = state.get("agent_config_assistant")
    model_config = get_model_config_from_state(state)
    model_name = model_config["model_name"]
    provider = model_config["provider"]
    
    if existing and getattr(existing, 'model_name', None) == model_name and getattr(existing, 'provider', None) == provider:
        return existing
    
    # Create new assistant
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")
    
    assistant = AgentConfigurationAssistant(model=model_name, provider=provider, thread_id=thread_id)
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
        assistant = AgentConfigurationAssistant(model=MODEL_NAME, provider=MODEL_PROVIDER)
        
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
        # Use centralized model name
        assistant = AgentConfigurationAssistant(model=MODEL_NAME)
        
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
        # Use centralized model name
        assistant = AgentConfigurationAssistant(model=MODEL_NAME)
        
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
        # Use centralized model name
        assistant = AgentConfigurationAssistant(model=MODEL_NAME)
        
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
        # Use centralized model name
        assistant = AgentConfigurationAssistant(model=MODEL_NAME)
        
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


@tool
async def configure_agent_complete(
    user_input: str,
    current_state_json: Optional[str] = None,
    conversation_history_json: Optional[str] = None,
    agent_id: Optional[str] = None
) -> str:
    """
    Complete agent configuration workflow that handles the entire process from analysis to instruction generation.
    
    This tool intelligently manages the agent configuration workflow:
    1. Checks current state (initial, analysis, clarification, tool_selection, instruction_generation, complete)
    2. Analyzes user input with unrelated detection (handles greetings/off-topic)
    3. Processes clarification answers if in clarification stage
    4. Extracts tool categories from analysis
    5. Searches and recommends tools from vector database
    6. Generates comprehensive agent instruction
    7. Supports instruction and tool modifications
    
    Args:
        user_input: User's message describing what they want or answering questions
        current_state_json: Optional JSON string with current configuration state
        conversation_history_json: Optional JSON string with conversation history for context
        agent_id: Optional agent UUID for database sync operations
        
    Returns:
        JSON string with complete configuration result including stage, response_message, 
        needs_clarification, questions, current_instruction, recommended_tools, etc.
    """
    try:
        # Lazy import to avoid dependency issues
        AgentConfigurationAssistant = _get_assistant_class()
        # Use centralized model name
        assistant = AgentConfigurationAssistant(model=MODEL_NAME)
        
        # Parse current state
        current_state = {}
        if current_state_json:
            try:
                current_state = json.loads(current_state_json) if isinstance(current_state_json, str) else current_state_json
            except:
                current_state = {}
        
        # Parse conversation history
        conversation_history = []
        if conversation_history_json:
            try:
                conversation_history = json.loads(conversation_history_json) if isinstance(conversation_history_json, str) else conversation_history_json
            except:
                conversation_history = []
        
        # Build full context for methods that need it
        full_context = {
            "conversation_history": conversation_history,
            "accumulated_context": current_state.get("accumulated_context", ""),
            "clarification_history": current_state.get("clarification_history", [])
        }
        
        # Determine current stage
        stage = current_state.get("stage", "initial")
        current_instruction = current_state.get("current_instruction", {})
        pending_questions = current_state.get("pending_questions", [])
        previous_analysis = current_state.get("analysis_result", {})
        agent_summary = current_state.get("agent_summary", {})
        
        # Initialize result structure
        result = {
            "stage": stage,
            "response_message": "",
            "needs_clarification": False,
            "questions": [],
            "is_unrelated": False,
            "current_instruction": current_instruction,
            "recommended_tools": current_state.get("recommended_tools", []),
            "analysis_result": previous_analysis,
            "agent_summary": agent_summary,
            "success": True
        }
        
        # STAGE 1: Initial or Requirement Analysis
        if stage in ["initial", "requirement_analysis"]:
            # Check if instruction already exists
            has_instruction = current_instruction and current_instruction.get("role")
            
            # Call _analyze_user_input with appropriate context
            agent_summary_context = None
            if has_instruction:
                # Use existing instruction as context
                agent_summary_context = {
                    "role": current_instruction.get("role"),
                    "responsibility": current_instruction.get("responsibility"),
                    "process_steps": current_instruction.get("process_steps", []),
                    "core_responsibilities": current_instruction.get("process_steps", []),
                    "expected_workflow": current_instruction.get("process_steps", [])
                }
            
            # Analyze user input to identify intent, extract requirements, and get tool categories
            analysis_result = await assistant._analyze_user_input(
                user_input,
                agent_summary_context=agent_summary_context,
                full_context=full_context
            )
            
            # Extract intent and routing information
            message_analysis = analysis_result.get("message_analysis", {})
            message_type = message_analysis.get("type", "general_chat")
            modification_type = message_analysis.get("modification_type")
            tool_categories = analysis_result.get("tool_categories", {})
            extracted_requirements = analysis_result.get("extracted_requirements", {})
            needs_clarification = analysis_result.get("needs_clarification", False)
            clarification_questions = analysis_result.get("clarification_questions", [])
            
            # Build agent summary from extracted requirements
            agent_summary = {
                "purpose": extracted_requirements.get("role", ""),
                "role": extracted_requirements.get("role", ""),
                "responsibility": extracted_requirements.get("responsibility", ""),
                "core_responsibilities": extracted_requirements.get("process_steps", []),
                "expected_workflow": extracted_requirements.get("process_steps", []),
                "analysis_summary": tool_categories.get("analysis_summary", "")
            }
            
            # Route based on intent
            if message_type == "agent_creation":
                if needs_clarification and clarification_questions:
                    # Need clarification - ask questions
                    summary_text = (
                        f"**Agent Summary:**\n\n"
                        f"**Role:** {agent_summary.get('role', 'N/A')}\n\n"
                        f"**Responsibility:** {agent_summary.get('responsibility', 'N/A')}\n\n"
                        f"{agent_summary.get('analysis_summary', '')}\n"
                    )
                    questions_text = "\n\n".join([f"**{i+1}.** {q.strip()}" for i, q in enumerate(clarification_questions)])
                    response_message = (
                        f"{summary_text}\n\n"
                        f"To configure this agent completely, I need some additional information:\n\n"
                        f"{questions_text}\n\n"
                        f"Please provide answers to these questions so I can proceed with the configuration.\n"
                    )
                    
                    result.update({
                        "stage": "clarification",
                        "response_message": response_message,
                        "needs_clarification": True,
                        "questions": clarification_questions,
                        "analysis_result": analysis_result,
                        "agent_summary": agent_summary,
                        "tool_categories": tool_categories,
                        "extracted_requirements": extracted_requirements,
                        "message_type": message_type,
                        "question_round": 1  # Initialize question round tracking
                    })
                else:
                    # Clear requirements - proceed to tool selection
                    summary_text = (
                        f"**Agent Summary:**\n\n"
                        f"**Role:** {agent_summary.get('role', 'N/A')}\n\n"
                        f"**Responsibility:** {agent_summary.get('responsibility', 'N/A')}\n\n"
                        f"{agent_summary.get('analysis_summary', '')}\n\n"
                        f"Great! I have all the information needed. Let me find the appropriate tools..."
                    )
                    
                    result.update({
                        "stage": "tool_selection",
                        "response_message": summary_text,
                        "needs_clarification": False,
                        "analysis_result": analysis_result,
                        "agent_summary": agent_summary,
                        "tool_categories": tool_categories,
                        "extracted_requirements": extracted_requirements,
                        "message_type": message_type
                    })
            
            elif message_type == "modification":
                # Route to modification handler
                # Store modification info for handling in modification stage
                result.update({
                    "stage": "modification",
                    "response_message": f"Processing modification request: {modification_type or 'general modification'}",
                    "analysis_result": analysis_result,
                    "agent_summary": agent_summary,
                    "tool_categories": tool_categories,
                    "extracted_requirements": extracted_requirements,
                    "message_type": message_type,
                    "modification_type": modification_type
                })
            
            elif message_type == "clarification":
                # Process clarification - if instruction exists, use it as context
                # This will be handled in STAGE 2
                result.update({
                    "stage": "clarification",
                    "response_message": "Processing your clarification...",
                    "analysis_result": analysis_result,
                    "agent_summary": agent_summary,
                    "tool_categories": tool_categories,
                    "extracted_requirements": extracted_requirements,
                    "message_type": message_type
                })
            
            elif message_type == "general_chat":
                # Return general chat response
                response_message = analysis_result.get("response_message", "I'm here to help you configure AI agents. How can I assist you?")
                result.update({
                    "stage": "initial",
                    "response_message": response_message,
                    "is_unrelated": True,
                    "analysis_result": analysis_result
                })
            
            else:
                # Fallback to general chat
                result.update({
                    "stage": "initial",
                    "response_message": "I'm here to help you configure AI agents. How can I assist you?",
                    "is_unrelated": True,
                    "analysis_result": analysis_result
                })
        
        # STAGE 2: Clarification - Process user answers
        elif stage == "clarification":
            # Check if instruction already exists
            has_instruction = current_instruction and current_instruction.get("role")
            
            if has_instruction:
                # Use _analyze_user_input with existing instruction as context
                agent_summary_context = {
                    "role": current_instruction.get("role"),
                    "responsibility": current_instruction.get("responsibility"),
                    "process_steps": current_instruction.get("process_steps", []),
                    "core_responsibilities": current_instruction.get("process_steps", []),
                    "expected_workflow": current_instruction.get("process_steps", [])
                }
                
                # Analyze user input with instruction context
                analysis_result = await assistant._analyze_user_input(
                    user_input,
                    agent_summary_context=agent_summary_context,
                    full_context=full_context
                )
                
                # Extract updated requirements
                extracted_requirements = analysis_result.get("extracted_requirements", {})
                message_type = analysis_result.get("message_analysis", {}).get("type", "clarification")
                tool_categories = analysis_result.get("tool_categories", {})
                
                # Update agent summary from extracted requirements
                agent_summary = {
                    "role": extracted_requirements.get("role", current_instruction.get("role", "")),
                    "responsibility": extracted_requirements.get("responsibility", current_instruction.get("responsibility", "")),
                    "core_responsibilities": extracted_requirements.get("process_steps", current_instruction.get("process_steps", [])),
                    "expected_workflow": extracted_requirements.get("process_steps", current_instruction.get("process_steps", []))
                }
                
                # If message_type is agent_creation, proceed to tool selection
                if message_type == "agent_creation":
                    result.update({
                        "stage": "tool_selection",
                        "response_message": "Thank you for the clarification. Let me find the appropriate tools...",
                        "needs_clarification": False,
                        "agent_summary": agent_summary,
                        "analysis_result": analysis_result,
                        "tool_categories": tool_categories,
                        "extracted_requirements": extracted_requirements,
                        "message_type": message_type
                    })
                else:
                    # Still in clarification or other type
                    result.update({
                        "stage": "clarification",
                        "response_message": "I've updated the agent configuration based on your clarification.",
                        "needs_clarification": False,
                        "agent_summary": agent_summary,
                        "analysis_result": analysis_result,
                        "tool_categories": tool_categories,
                        "extracted_requirements": extracted_requirements,
                        "message_type": message_type
                    })
            else:
                # No instruction exists, use existing clarification flow
                if not agent_summary:
                    agent_summary = previous_analysis
                
                if not pending_questions:
                    pending_questions = previous_analysis.get("questions", [])
                
                # Track question rounds to prevent infinite loops
                question_round = current_state.get("question_round", 1)
                previous_clarity = agent_summary.get("requirement_clarity_score", 0.0)
                
                # Process clarification answers using analyze_agent_role_and_responsibility
                analysis_result = await assistant.analyze_agent_role_and_responsibility(
                    user_input,
                    streaming_events=None,
                    full_context=full_context,
                    previous_analysis=agent_summary,
                    pending_questions=pending_questions,
                    is_clarification_response=True
                )
            
            # Extract updated information from analysis result
            response_message = analysis_result.get("response_message", "")
            still_needs_clarification = analysis_result.get("needs_clarification", True)
            remaining_questions = analysis_result.get("questions", [])
            new_clarity_score = analysis_result.get("requirement_clarity_score", previous_clarity)
            clarity_improvement = new_clarity_score - previous_clarity
            
            # Update agent summary with new analysis
            agent_summary = {
                **agent_summary,
                "purpose": analysis_result.get("purpose", agent_summary.get("purpose", "")),
                "role": analysis_result.get("role", agent_summary.get("role", "")),
                "responsibility": analysis_result.get("responsibility", agent_summary.get("responsibility", "")),
                "core_responsibilities": analysis_result.get("core_responsibilities", agent_summary.get("core_responsibilities", [])),
                "expected_workflow": analysis_result.get("expected_workflow", agent_summary.get("expected_workflow", [])),
                "requirement_clarity_score": new_clarity_score
            }
            
            # Enhanced decision logic with answer completeness check first, then clarity score thresholds
            should_proceed = False
            
            # Check if user provided substantial response (not empty or too short)
            user_response_length = len(user_input.strip()) if user_input else 0
            has_substantial_response = user_response_length > 10
            
            # PRIORITY RULE 1: If user answered all questions (no remaining questions) AND provided substantial response
            # AND clarity score is acceptable, proceed immediately
            if (not remaining_questions or len(remaining_questions) == 0) and has_substantial_response:
                if new_clarity_score >= 0.7:
                    should_proceed = True
                    logger.info(f"Proceeding: User answered all {len(pending_questions)} questions, proceeding with score {new_clarity_score:.2f}")
                elif new_clarity_score >= 0.65 and question_round >= 2:
                    # After 2 rounds, be lenient if all questions answered
                    should_proceed = True
                    logger.info(f"Proceeding: User answered all {len(pending_questions)} questions after {question_round} rounds, score {new_clarity_score:.2f}")
            
            # Rule 2: High clarity score - always proceed
            elif new_clarity_score >= 0.8:
                should_proceed = True
                logger.info(f"Proceeding: High clarity score ({new_clarity_score:.2f})")
            
            # Rule 3: Good score with significant improvement - proceed
            elif new_clarity_score >= 0.75 and clarity_improvement >= 0.15:
                should_proceed = True
                logger.info(f"Proceeding: Good score ({new_clarity_score:.2f}) with significant improvement ({clarity_improvement:.2f})")
            
            # Rule 4: Good score and no remaining questions
            elif new_clarity_score >= 0.75 and (not remaining_questions or len(remaining_questions) == 0):
                should_proceed = True
                logger.info(f"Proceeding: Good score ({new_clarity_score:.2f}) with no remaining questions")
            
            # Rule 5: After 2 rounds, be more lenient
            elif question_round >= 2 and new_clarity_score >= 0.7:
                should_proceed = True
                logger.info(f"Proceeding: After {question_round} rounds, clarity score is acceptable ({new_clarity_score:.2f})")
            
            # Rule 6: After 3 rounds, proceed unless score is very low
            elif question_round >= 3:
                if new_clarity_score >= 0.65:
                    should_proceed = True
                    logger.info(f"Proceeding: After {question_round} rounds, proceeding with score {new_clarity_score:.2f}")
                else:
                    # Even after 3 rounds, if score is too low, proceed anyway (we'll refine during instruction generation)
                    should_proceed = True
                    logger.warning(f"Proceeding after {question_round} rounds despite low score ({new_clarity_score:.2f}) - will refine during instruction generation")
            
            # Rule 7: If still needs clarification but no new questions, proceed
            elif still_needs_clarification and (not remaining_questions or len(remaining_questions) == 0):
                should_proceed = True
                logger.info("Proceeding: No remaining questions despite needs_clarification flag")
            
            if should_proceed or not (still_needs_clarification and remaining_questions):
                # Clarification complete - proceed to tool selection
                if not response_message:
                    if clarity_improvement > 0:
                        response_message = (
                            f"Perfect! I have all the information I need. "
                            f"Clarity improved from {previous_clarity:.2f} to {new_clarity_score:.2f}. "
                            f"Let me find the appropriate tools for your agent...\n"
                        )
                    else:
                        response_message = (
                            f"Perfect! I have all the information I need. "
                            f"Let me find the appropriate tools for your agent...\n"
                        )
                
                result.update({
                    "stage": "tool_selection",
                    "response_message": response_message,
                    "needs_clarification": False,
                    "agent_summary": agent_summary,
                    "analysis_result": analysis_result,
                    "question_round": 0  # Reset for next agent
                })
            else:
                # Still need more clarification
                question_round += 1
                if not response_message:
                    questions_text = "\n\n".join([f"**{i+1}.** {q.strip()}" for i, q in enumerate(remaining_questions)])
                    if clarity_improvement > 0:
                        response_message = (
                            f"Thank you for the information. "
                            f"Clarity improved from {previous_clarity:.2f} to {new_clarity_score:.2f}. "
                            f"I still need a few more details:\n\n"
                            f"{questions_text}\n\n"
                            f"Please provide answers to these remaining questions.\n"
                        )
                    else:
                        response_message = (
                            f"Thank you for the information. I still need a few more details:\n\n"
                            f"{questions_text}\n\n"
                            f"Please provide answers to these remaining questions.\n"
                        )
                
                result.update({
                    "stage": "clarification",
                    "response_message": response_message,
                    "needs_clarification": True,
                    "questions": remaining_questions,
                    "agent_summary": agent_summary,
                    "analysis_result": analysis_result,
                    "question_round": question_round
                })
        
        # STAGE 2.5: Modification - Handle agent modifications
        elif stage == "modification":
            # Get modification info from state
            modification_type = current_state.get("modification_type")
            tool_categories = current_state.get("tool_categories", {})
            extracted_requirements = current_state.get("extracted_requirements", {})
            
            # Only search tools for modifications that might need new tools
            needs_tool_search = modification_type in ["add_tool", "add_step", "add_requirement"]
            recommended_tools = current_state.get("recommended_tools", [])
            
            if needs_tool_search and tool_categories:
                # Search tools from vector DB
                tools_result = assistant.filter_tools_from_vector_db(
                    categories=tool_categories,
                    limit=50,
                    user_input=user_input
                )
                
                tools_list = tools_result.get("tools_list", [])
                
                if tools_list:
                    # Build current instruction dict for semantic search
                    current_instruction_dict = current_instruction if current_instruction else {
                        "role": "",
                        "responsibility": "",
                        "process_steps": []
                    }
                    
                    # Semantic search and rank tools
                    semantic_result = await assistant.semantic_search_from_filtered_tools(
                        user_input=user_input,
                        current_instruction=current_instruction_dict,
                        tools=tools_list,
                        streaming_events=None,
                        full_context=full_context
                    )
                    
                    recommended_tools = semantic_result.get("recommended_tools", [])
            
            # Call get_instruction with modification_type
            instruction = await assistant.get_instruction(
                current_instruction=current_instruction,
                recommended_tools=recommended_tools,
                streaming_events=None,
                message_type="modification",
                modification_type=modification_type,
                extracted_requirements=extracted_requirements,
                user_input=user_input,
                full_context=full_context
            )
            
            # Convert to dict if needed
            if hasattr(instruction, 'model_dump'):
                instruction_dict = instruction.model_dump()
            elif hasattr(instruction, 'dict'):
                instruction_dict = instruction.dict()
            else:
                instruction_dict = instruction
            
            # Format response message
            role = instruction_dict.get("role", "")
            responsibility = instruction_dict.get("responsibility", "")
            process_steps = instruction_dict.get("process_steps", [])
            
            response_message = (
                f"✅ **Agent Updated!**\n\n"
                f"**Modification Type:** {modification_type or 'general modification'}\n\n"
                f"**Role:** {role}\n\n"
                f"**Responsibility:** {responsibility}\n\n"
                f"**Process Steps:**\n"
            )
            for i, step in enumerate(process_steps, 1):
                response_message += f"{i}. {step}\n"
            
            if recommended_tools:
                response_message += f"\n**Recommended Tools:** {len(recommended_tools)} tools selected\n"
            
            result.update({
                "stage": "complete",
                "response_message": response_message,
                "current_instruction": instruction_dict,
                "recommended_tools": recommended_tools,
                "needs_clarification": False
            })
        
        # STAGE 3: Tool Selection - Use tool_categories from _analyze_user_input
        elif stage == "tool_selection":
            # Get tool_categories from state (set in STAGE 1)
            tool_categories = current_state.get("tool_categories", {})
            
            # If not in state, extract from agent_summary (fallback)
            if not tool_categories or not tool_categories.get("search_queries"):
                if not agent_summary:
                    agent_summary = previous_analysis
                
                # Fallback: Build tool categories from agent summary
                workflow = agent_summary.get("expected_workflow", [])
                responsibilities = agent_summary.get("core_responsibilities", [])
                
                if not tool_categories:
                    tool_categories = {
                        "primary_categories": [],
                        "required_tools": [],
                        "keywords": [],
                        "search_queries": []
                    }
                
                # Create search queries from workflow if not already present
                if workflow and not tool_categories.get("search_queries"):
                    tool_categories["search_queries"] = [
                        {"query": step, "weight": 0.8} for step in workflow[:5]
                    ]
            
            # Filter tools from vector DB using tool_categories
            tools_result = assistant.filter_tools_from_vector_db(
                categories=tool_categories,
                limit=50,
                user_input=user_input
            )
            
            tools_list = tools_result.get("tools_list", [])
            
            if not tools_list:
                result.update({
                    "stage": "tool_selection",
                    "response_message": "I couldn't find specific tools. Let me proceed with instruction generation based on the analysis.",
                    "recommended_tools": []
                })
            else:
                # Perform semantic search to rank tools
                current_instruction_dict = {
                    "role": agent_summary.get("role", ""),
                    "responsibility": agent_summary.get("responsibility", ""),
                    "process_steps": workflow
                }
                
                semantic_result = await assistant.semantic_search_from_filtered_tools(
                    user_input=user_input,
                    current_instruction=current_instruction_dict,
                    tools=tools_list,
                    streaming_events=None,
                    full_context=full_context
                )
                
                recommended_tools = semantic_result.get("recommended_tools", [])
                
                result.update({
                    "stage": "instruction_generation",
                    "recommended_tools": recommended_tools,
                    "response_message": f"Found {len(recommended_tools)} relevant tools. Generating agent instruction..."
                })
                
                # Continue to instruction generation
                stage = "instruction_generation"
        
        # STAGE 4: Instruction Generation
        if stage == "instruction_generation" or (stage == "tool_selection" and result.get("recommended_tools")):
            # Get extracted_requirements from state (set in STAGE 1)
            extracted_requirements = current_state.get("extracted_requirements", {})
            
            # Generate instruction from extracted requirements and tools
            if not agent_summary:
                agent_summary = previous_analysis
            
            recommended_tools = result.get("recommended_tools", [])
            
            # Build current instruction from extracted_requirements if available, otherwise from agent_summary
            if extracted_requirements and extracted_requirements.get("role"):
                current_instruction_dict = {
                    "role": extracted_requirements.get("role", ""),
                    "responsibility": extracted_requirements.get("responsibility", ""),
                    "process_title": extracted_requirements.get("process_title", "Main Process"),
                    "process_steps": extracted_requirements.get("process_steps", [])
                }
            else:
                # Fallback to agent_summary
                current_instruction_dict = {
                    "role": agent_summary.get("role", ""),
                    "responsibility": agent_summary.get("responsibility", ""),
                    "process_title": "Main Process",
                    "process_steps": agent_summary.get("expected_workflow", [])
                }
            
            # Generate comprehensive instruction
            instruction = await assistant.get_instruction(
                current_instruction=current_instruction_dict,
                recommended_tools=recommended_tools,
                streaming_events=None,
                message_type="agent_creation",
                extracted_requirements=extracted_requirements if extracted_requirements else current_instruction_dict,
                full_context=full_context
            )
            
            # Convert to dict if needed
            if hasattr(instruction, 'model_dump'):
                instruction_dict = instruction.model_dump()
            elif hasattr(instruction, 'dict'):
                instruction_dict = instruction.dict()
            else:
                instruction_dict = instruction
            
            # Format response message
            role = instruction_dict.get("role", "")
            responsibility = instruction_dict.get("responsibility", "")
            process_steps = instruction_dict.get("process_steps", [])
            
            response_message = (
                f"✅ **Agent Configuration Complete!**\n\n"
                f"**Role:** {role}\n\n"
                f"**Responsibility:** {responsibility}\n\n"
                f"**Process Steps:**\n"
            )
            for i, step in enumerate(process_steps, 1):
                response_message += f"{i}. {step}\n"
            
            if recommended_tools:
                response_message += f"\n**Recommended Tools:** {len(recommended_tools)} tools selected\n"
            
            result.update({
                "stage": "complete",
                "response_message": response_message,
                "current_instruction": instruction_dict,
                "recommended_tools": recommended_tools,
                "needs_clarification": False
            })
        
        return json.dumps(result, indent=2, default=str)
        
    except Exception as e:
        import traceback
        error_msg = f"Error in configure_agent_complete: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        
        error_result = {
            "stage": "error",
            "response_message": f"I encountered an error: {error_msg}",
            "needs_clarification": False,
            "questions": [],
            "is_unrelated": False,
            "current_instruction": {},
            "recommended_tools": [],
            "success": False,
            "error": str(e)
        }
        return json.dumps(error_result, indent=2, default=str)


# @tool
# def your_tool_here(your_arg: str):
#     """Your tool description here."""
#     print(f"Your tool logic here")
#     return "Your tool response here."

tools = [
    configure_agent_complete,  # Primary comprehensive tool
    analyze_agent_requirements,  # Keep for backward compatibility
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
            configure_agent_complete,  # Primary tool
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
        elif provider == "huggingface_hub":
            # Hugging Face models may have limited tool support - bind without parallel_tool_calls
            try:
                model_with_tools = model.bind_tools(unique_tools)
            except Exception as e:
                logger.warning(f"Hugging Face model may not support tools: {e}. Using model without tools.")
                model_with_tools = model
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

You have access to a powerful agent configuration tool:

**PRIMARY TOOL: configure_agent_complete**
This is your main tool for agent configuration. It handles the complete workflow:
1. Analyzes user requirements (with unrelated input detection)
2. Asks clarification questions if needed
3. Processes clarification answers
4. Searches and recommends tools from vector database
5. Generates comprehensive agent instructions

**WORKFLOW:**
- When user wants to create/configure an agent, use configure_agent_complete
- Pass the user's input and optionally current_state_json if you have state information
- The tool will return the current stage and what to do next
- If needs_clarification=true, ask the questions to the user
- If stage="complete", present the final instruction and tools to the user

**STATE MANAGEMENT:**
- Store the returned state in your memory/context
- On subsequent calls, pass the previous state as current_state_json
- This allows the tool to continue from where it left off

**OTHER TOOLS (for specific needs):**
- analyze_agent_requirements: Quick analysis only
- search_tools_by_categories: Direct tool search
- recommend_tools_for_agent: Tool ranking
- generate_agent_instruction: Instruction generation
- generate_clarification_questions: Question generation

Always use configure_agent_complete as your primary tool for agent configuration workflows.
Provide clear, helpful responses and guide users through the process step by step."""
        
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
                if provider == "gemini":
                    response = AIMessage(content=f"API key error. Please check your GOOGLE_API_KEY in .env file.")
                elif provider == "huggingface_hub":
                    response = AIMessage(content=f"Hugging Face API key error. Please check your HUGGINGFACE_API_KEY in .env file. Some models require authentication.")
                else:
                    response = AIMessage(content=f"API key error. Please check your API key configuration.")
            elif "403" in error_msg or "forbidden" in error_msg.lower():
                if provider == "huggingface_hub":
                    response = AIMessage(content=f"Hugging Face API returned 403 Forbidden. This usually means: 1) Missing or invalid HUGGINGFACE_API_KEY, 2) The model requires access request (check on huggingface.co), or 3) Rate limit exceeded. Please check your API key and model access.")
                else:
                    response = AIMessage(content=f"Access forbidden (403). Please check your API credentials.")
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
# When using LangGraphHttpAgent via add_langgraph_fastapi_endpoint, it should automatically
# provide thread_id in the config. Set USE_CHECKPOINTER=false to disable if having issues.
use_checkpointer = os.getenv("USE_CHECKPOINTER", "true").lower() == "true"

if use_checkpointer:
    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)
else:
    # Compile without checkpointer - useful if endpoint doesn't provide thread_id
    logger.info("Compiling graph without checkpointer (USE_CHECKPOINTER=false)")
    graph = workflow.compile()
