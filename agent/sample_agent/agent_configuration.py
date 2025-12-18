from typing import TypedDict, Annotated, List, Dict, Optional, Any, Tuple
import socket
import os
import sys
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import operator
import time
from functools import wraps, lru_cache
from dotenv import load_dotenv
from datetime import datetime
import google.generativeai as genai  # type: ignore
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json, logging, sys
import requests
import httpx
import re

# Add backend directory to sys.path to enable proper imports
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Add project root to sys.path for src imports
project_root = os.path.dirname(os.path.dirname(backend_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from qdrant_client.models import Distance, VectorParams, PointStruct
from src.backend.qdrant.dependencies import get_qdrant_client
from src.backend.api.v1.knowledge_base.service.qdrant import QdrantService
from src.backend.qdrant.embedding import EmbeddingService
from src.backend.qdrant.helper import EmbeddingModelManager
from src.backend.helpers import get_ai_provider_key
from src.backend.api.v1.knowledge_base.helpers import utc_now
from src.backend.environment import Config
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.memory import InMemorySaver
import uuid

# Import LangGraph state and graph components
from src.backend.agent_configuration.state import AgentConfigState
from src.backend.agent_configuration.graph import compile_agent_config_graph
from src.backend.agent_configuration.nodes import _calculate_eta

# Import token estimation utilities
from src.backend.graph.v1.history import estimate_tokens, count_messages_tokens

# Set socket timeout to prevent hanging imports
socket.setdefaulttimeout(10)

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = "GOOGLE_API_KEY"

# Configure logging to go to stderr
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)
# Caching utilities
def create_cache_key(*args, **kwargs) -> str:
    """Create a hash key for caching based on arguments"""
    content = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(content.encode()).hexdigest()

class LLMCache:
    """Simple LRU cache for LLM responses"""
    def __init__(self, maxsize: int = 100):
        self.maxsize = maxsize
        self.cache = {}
        self.access_order = []
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        if key in self.cache:
            # Update existing
            self.access_order.remove(key)
        elif len(self.cache) >= self.maxsize:
            # Remove least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
        
        self.cache[key] = value
        self.access_order.append(key)
    
    def clear(self) -> None:
        self.cache.clear()
        self.access_order.clear()

# Global LLM cache
llm_cache = LLMCache(maxsize=50)

# Simple thread ID generator
def generate_thread_id() -> str:
    """Generate a unique thread ID"""
    return str(uuid.uuid4())

# Timing utilities for LLM calls and methods
def time_method(method_name: str = None, log_args: bool = False):
    """Decorator to time method execution and log the duration with detailed information"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            method_display_name = method_name or func.__name__
            
            # Log method start with optional arguments
            # if log_args and args:
            #     # Show first argument if it's a string (usually user input)
            #     first_arg = str(args[0]) + "..." if len(str(args[0])) > 100 else str(args[0])
            #     logger.info(f"⏱️  Starting method: {method_display_name} | Input: '{first_arg}'")
            # else:
            #     logger.info(f"⏱️  Starting method: {method_display_name}")
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                duration = end_time - start_time
                
                # Enhanced completion logging
                if hasattr(result, '__len__') and not isinstance(result, str):
                    result_info = f" | Results: {len(result)} items"
                else:
                    result_info = ""
                
                logger.info(f"✅ Method completed: {method_display_name} - Duration: {duration:.3f}s{result_info}")
                return result
                
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                logger.error(f"❌ Method failed: {method_display_name} - Duration: {duration:.3f}s - Error: {str(e)}")
                raise
                
        return wrapper
    return decorator

class TimingContext:
    """Context manager for timing specific code blocks"""
    def __init__(self, operation_name: str, assistant_instance=None):
        self.operation_name = operation_name
        self.start_time = None
        self.assistant = assistant_instance
        
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"⏱️  Starting operation: {self.operation_name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            if exc_type is None:
                logger.info(f"✅ Operation completed: {self.operation_name} - Duration: {duration:.3f}s")
            else:
                logger.error(f"❌ Operation failed: {self.operation_name} - Duration: {duration:.3f}s - Error: {exc_val}")
            
            # Track vector search time if this is a vector search operation
            if self.assistant and "Vector Search" in self.operation_name:
                if not hasattr(self.assistant, 'vector_search_time'):
                    self.assistant.vector_search_time = 0
                self.assistant.vector_search_time += duration


# Define the structured output schema
class AgentInstruction(BaseModel):
    role: str = Field(description="Brief description of the assistant's role")
    responsibility: str = Field(description="Main responsibility of the assistant")
    key_tool: Optional[str] = Field(description="Primary tool or platform mentioned by user", default=None)
    process_title: str = Field(description="Title for the main process")
    process_steps: List[str] = Field(description="Main steps in the process")
    tool_usage_title: Optional[str] = Field(description="Title for tool-specific steps", default=None)
    tool_steps: List[str] = Field(description="Steps for using the specific tool", default_factory=list)

# Define the assistant reply schema
class AssistantReply(BaseModel):
    acknowledge_intent: str = Field(description="Acknowledgment of the user's goal in a concise, empathetic way")
    explain_changes: str = Field(description="Clear explanation of specific capabilities, tools, and instructions that will be added, modified, or removed")
    follow_up_questions: List[str] = Field(description="1-2 specific, relevant questions to refine the implementation", default_factory=list)
    suggest_next_steps: str = Field(description="Clear, logical action for the user to take")

# Conversation state
class ConversationState(TypedDict):
    messages: List[Dict[str, str]]
    current_instruction: Dict
    user_input: str
    assistant_response: str
    is_complete: bool
    needs_clarification: bool
    clarification_questions: List[str]

# LLM Timing Context Manager (must be defined before AgentConfigurationAssistant)
class LLMTimingContext:
    """Context manager for timing LLM calls with session tracking"""
    def __init__(self, method_name: str, assistant_instance=None):
        self.method_name = method_name
        self.start_time = None
        self.assistant = assistant_instance
        
    def __enter__(self):
        self.start_time = time.time()
        
        # Track session timing
        if self.assistant:
            if self.assistant.llm_session_start_time is None:
                self.assistant.llm_session_start_time = self.start_time
                # logger.info(f"🚀 LLM Session Started: {time.strftime('%H:%M:%S', time.localtime(self.start_time))}")
            
            self.assistant.llm_call_count += 1
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = end_time - self.start_time
        
        # Update session end time and track total LLM time
        if self.assistant:
            self.assistant.llm_session_end_time = end_time
            # Track total LLM time
            if not hasattr(self.assistant, 'llm_total_time'):
                self.assistant.llm_total_time = 0
            self.assistant.llm_total_time += duration
        
        return False  # Don't suppress exceptions

class AgentConfigurationAssistant:
    def __init__(self, model="gemini-2.5-flash", thread_id: Optional[str] = None):
        """Initialize with Gemini model and optional thread_id for memory"""
        try:
            # Configure Google Gemini (moved from module level to avoid import-time hangs)
            if GOOGLE_API_KEY:
                try:
                    genai.configure(api_key=GOOGLE_API_KEY)
                except Exception as genai_error:
                    logger.warning(f"Warning: Could not configure genai directly: {genai_error}. Continuing with ChatGoogleGenerativeAI.")
            else:
                raise ValueError("GOOGLE_API_KEY is required")
            
            # Log the model being used for debugging
            logger.info(f"Initializing AgentConfigurationAssistant with model: {model}")
            self.llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=0.3,
                google_api_key=GOOGLE_API_KEY
            )
            # Store model name for later comparison
            self.model_name = model
        except Exception as e:
            logger.error(f"Error initializing AgentConfigurationAssistant: {e}")
            logger.info("Make sure you have set GOOGLE_API_KEY and installed langchain-google-genai")
            raise
            
        self.parser = PydanticOutputParser(pydantic_object=AgentInstruction)
        self.reply_parser = PydanticOutputParser(pydantic_object=AssistantReply)
        
        # Memory system integration - use same system as load_agent_graph
        self.thread_id = thread_id or generate_thread_id()
        self.checkpointer = None  # Will be initialized when needed
        self._checkpointer_initialized = False
        self._checkpointer_cm = None  # Store the context manager
        
        # Initialize conversation state
        self.conversation_history = []
        
        # Conversation summarization settings
        self.max_conversation_messages = 10  # Trigger: message count
        self.max_conversation_tokens = 3000  # Trigger: token count
        self.preserve_recent_messages = 4    # Keep last 4 messages full
        self.conversation_summary = ""       # Stores compressed history
        self.current_stage = "agent_role"
        
        # Store analysis results for later access
        self.analysis_result = None
        self.questions_result = None
        self.combined_result = None

        self.current_instruction = AgentInstruction(
            role="",
            responsibility="",
            process_title="Main Process",
            process_steps=[]
        )
        self.previous_instruction = AgentInstruction(
            role="",
            responsibility="",
            process_title="Main Process",
            process_steps=[]
        )
        
        # Session timing tracking
        self.llm_session_start_time = None
        self.llm_session_end_time = None
        self.llm_call_count = 0
        
        # Session caching for performance
        self.session_cache = {}
        self.tools_cache = {}
        self.categories_cache = {}
        
        # LangGraph state graph initialization
        self.graph = None  # Will be compiled with checkpointer when needed
    
    def _log_llm_prompt(self, messages: List, prompt_name: str = "LLM Call"):
        """Helper method to log LLM prompt"""
        try:
            # Extract prompt content from messages
            prompt_content = ""
            for msg in messages:
                if hasattr(msg, 'content'):
                    prompt_content += f"{msg.__class__.__name__}: {msg.content}\n"
                elif isinstance(msg, dict):
                    prompt_content += f"{msg.get('type', 'Message')}: {msg.get('content', '')}\n"
                else:
                    prompt_content += f"{str(msg)}\n"
            
            logger.info(f"🤖 [{prompt_name}] LLM PROMPT:\n{'='*80}\n{prompt_content}\n{'='*80}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to log LLM prompt: {e}")
    
    def _log_llm_response(self, response: str, prompt_name: str = "LLM Call"):
        """Helper method to log LLM response"""
        try:
            # Truncate very long responses for readability
            max_length = 5000
            if len(response) > max_length:
                truncated = response[:max_length] + f"\n... (truncated, total length: {len(response)} chars)"
                logger.info(f"🤖 [{prompt_name}] LLM RESPONSE:\n{'='*80}\n{truncated}\n{'='*80}")
            else:
                logger.info(f"🤖 [{prompt_name}] LLM RESPONSE:\n{'='*80}\n{response}\n{'='*80}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to log LLM response: {e}")
    
    async def _stream_llm_response(self, messages: List, prompt_name: str = "LLM Response", streaming_events: Optional[List] = None):
        """
        Stream LLM response as message_delta events for CopilotKit.
        
        Args:
            messages: List of LangChain messages to send to LLM
            prompt_name: Name of the prompt/operation for logging
            streaming_events: Optional list to append events to (mutates the list)
            
        Yields:
            dict: message_delta events with delta and accumulated content
        """
        accumulated = ""
        try:
            with LLMTimingContext(prompt_name, self):
                response = await self.llm.ainvoke(messages)
                if hasattr(response, 'content') and response.content:
                    accumulated = response.content
                    
                    event = {
                        "type": "message_delta",
                        "timestamp": time.time(),
                        "data": {
                            "delta": accumulated,
                            "accumulated": accumulated,
                            "role": "assistant",
                            "content": accumulated
                        }
                    }
                    
                    # Append to streaming_events if provided
                    if streaming_events is not None:
                        streaming_events.append(event)
                    
                    # Yield message_delta event for CopilotKit streaming
                    yield event
        except Exception as e:
            logger.error(f"Error streaming LLM response in {prompt_name}: {e}")
            error_event = {
                "type": "message",
                "timestamp": time.time(),
                "data": {
                    "role": "assistant",
                    "content": f"Error: {str(e)}"
                }
            }
            # Append error to streaming_events if provided
            if streaming_events is not None:
                streaming_events.append(error_event)
            # Yield error as message
            yield error_event
            raise
    
    def print_llm_session_summary(self):
        """Print summary of LLM session timing"""
        if self.llm_session_start_time and self.llm_session_end_time:
            total_duration = self.llm_session_end_time - self.llm_session_start_time
            start_time_str = time.strftime('%H:%M:%S', time.localtime(self.llm_session_start_time))
            end_time_str = time.strftime('%H:%M:%S', time.localtime(self.llm_session_end_time))
            
            logger.info(f"\n📊 LLM Session Summary:")
            logger.info(f"   🚀 Session Started: {start_time_str}")
            logger.info(f"   🏁 Session Ended: {end_time_str}")
            logger.info(f"   ⏱️  Total Session Duration: {total_duration:.2f}s")
            logger.info(f"   🔢 Total LLM Calls: {self.llm_call_count}")
            if self.llm_call_count > 0:
                avg_duration = total_duration / self.llm_call_count
                logger.info(f"   📈 Average Call Duration: {avg_duration:.2f}s")
            logger.info("=" * 50)
        
    def start_conversation(self):
        """Initialize the conversation"""
        welcome_message = """Hi! I can help you configure this agent with specific capabilities and access to various tools. What type of agent are you trying to build?"""
        # print(welcome_message)
        return welcome_message
    
    async def _initialize_checkpointer(self):
        """Initialize persistent checkpointer instance"""
        if not self._checkpointer_initialized:
            # Initialize in-memory checkpointer (no network/DB, no setup needed)
            self.checkpointer = InMemorySaver()
            self._checkpointer_initialized = True
            return self.checkpointer
    
    async def close_checkpointer(self):
        """Close checkpointer connection"""
        if self.checkpointer:
            # In-memory saver: nothing to close, just drop the reference
            self.checkpointer = None
            self._checkpointer_cm = None
            self._checkpointer_initialized = False
    
    @time_method("Process Message Stream", log_args=True)
    async def process_message_stream(self, user_input: str, agent_id: Optional[str] = None, cookies: Optional[Dict[str, str]] = None):
        """Process message using LangGraph with streaming via .astream() method
        
        Args:
            user_input: User's message input
            agent_id: Optional agent UUID
            cookies: Optional dict of cookies for authenticated API calls
        """
        # Initialize checkpointer if needed
        await self._initialize_checkpointer()
        
        try:
            # Compile graph with checkpointer
            graph = compile_agent_config_graph(self, self.checkpointer)
            
            # Configuration for this thread
            config = {"configurable": {"thread_id": self.thread_id}}

            # Try to get existing state from checkpointer
            existing_state = None
            try:
                checkpoint = await self.checkpointer.aget_tuple(config)
                
                if checkpoint and checkpoint.checkpoint:
                    existing_state = checkpoint.checkpoint.get("channel_values")
            except Exception as e:
                logger.error(f"Could not load existing state: {e}")
                existing_state = None

            # Prepare current instructions data
            current_instruction_data = (
                self.current_instruction.model_dump() 
                if hasattr(self.current_instruction, 'model_dump') 
                else self.current_instruction
            )
            previous_instruction_data = (
                self.previous_instruction.model_dump() 
                if hasattr(self.previous_instruction, 'model_dump') 
                else self.previous_instruction
            )

            # Create or update state
            if existing_state:
                # Merge new user message with existing state
                agent_config_fields = {field.name for field in AgentConfigState.__dataclass_fields__.values()}
                filtered_state = {k: v for k, v in existing_state.items() if k in agent_config_fields}
                
                initial_state = AgentConfigState(**filtered_state)
                initial_state.messages.append(HumanMessage(content=user_input))
                initial_state.user_input = user_input
                # Update agent_id if provided and not already set
                if agent_id and not initial_state.agent_id:
                    initial_state.agent_id = agent_id
                # Update cookies if provided (for authenticated API calls)
                if cookies:
                    initial_state.request_cookies = cookies
            else:
                # First message - create fresh state
                initial_state = AgentConfigState(
                    messages=[HumanMessage(content=user_input)],
                    current_instruction=current_instruction_data,
                    previous_instruction=previous_instruction_data,
                    user_input=user_input,
                    agent_id=agent_id or "",
                    current_agent_state="requirement_analysis",
                    request_cookies=cookies or {}
                )
            
            last_event_count = 0
            final_state = None
            
            logger.info(f"🔍 [STREAM] Starting LangGraph stream with initial state")
            
            # Use LangGraph with manual event streaming
            async for chunk in graph.astream(initial_state, config, stream_mode="updates"):
                # Extract state from chunk
                for node_name, state_update in chunk.items():
                    logger.info(f"🔍 [STREAM] Received chunk from node: {node_name}")
                    final_state = state_update
                    
                    # Get the updated state and extract new events
                    streaming_events = None
                    if hasattr(state_update, 'streaming_events'):
                        streaming_events = state_update.streaming_events
                    elif isinstance(state_update, dict) and 'streaming_events' in state_update:
                        streaming_events = state_update['streaming_events']
                    
                    if streaming_events:
                        new_events = streaming_events[last_event_count:]
                        logger.info(f"🔍 [STREAM] Node {node_name}: Found {len(new_events)} new events (total: {len(streaming_events)}, last_count: {last_event_count})")
                        for event in new_events:
                            event_type = event.get("type", "unknown") if isinstance(event, dict) else "unknown"
                            logger.info(f"🔍 [STREAM] Yielding event: type={event_type}, node={node_name}")
                            if event_type == "node_progress":
                                event_data = event.get("data", {})
                                logger.info(f"🔍 [STREAM] Node progress event details: {event_data}")
                            yield event
                        last_event_count = len(streaming_events)
                    else:
                        logger.debug(f"🔍 [STREAM] Node {node_name}: No streaming_events found")
            
            # Update conversation history from final state
            if final_state:
                # Handle both dict and object types
                current_instruction = None
                previous_instruction = None
                assistant_response = None
                recommended_tools = []
                needs_clarification = False
                questions = []
            
            if hasattr(final_state, 'current_instruction'):
                current_instruction = final_state.current_instruction
            elif isinstance(final_state, dict):
                current_instruction = final_state.get('current_instruction')
            
            if hasattr(final_state, 'previous_instruction'):
                previous_instruction = final_state.previous_instruction
            elif isinstance(final_state, dict):
                previous_instruction = final_state.get('previous_instruction')
            
            if hasattr(final_state, 'assistant_response'):
                assistant_response = final_state.assistant_response
            elif isinstance(final_state, dict):
                assistant_response = final_state.get('assistant_response')
            
            if hasattr(final_state, 'recommended_tools'):
                recommended_tools = final_state.recommended_tools
            elif isinstance(final_state, dict):
                recommended_tools = final_state.get('recommended_tools', [])
            
            if hasattr(final_state, 'needs_clarification'):
                needs_clarification = final_state.needs_clarification
            elif isinstance(final_state, dict):
                needs_clarification = final_state.get('needs_clarification', False)
            
            if hasattr(final_state, 'questions'):
                questions = final_state.questions
            elif isinstance(final_state, dict):
                questions = final_state.get('questions', [])
            
            # Update internal state
            if current_instruction:
                self.current_instruction = current_instruction
            if previous_instruction:
                self.previous_instruction = previous_instruction
            
            # Restore assistant instance stored results from state
            if final_state:
                # Restore analysis_result
                if hasattr(final_state, 'message_analysis'):
                    analysis_result = final_state.message_analysis.get("analysis_result")
                    if analysis_result:
                        self.analysis_result = analysis_result
                
                # Restore questions_result
                if hasattr(final_state, 'message_analysis'):
                    questions_result = final_state.message_analysis.get("questions_result")
                    if questions_result:
                        self.questions_result = questions_result
                
                # Restore combined_result if available
                if hasattr(final_state, 'message_analysis'):
                    analysis_result = final_state.message_analysis.get("analysis_result", {})
                    questions_result = final_state.message_analysis.get("questions_result", {})
                    if analysis_result and questions_result:
                        self.combined_result = {
                            **analysis_result,
                            "questions": questions_result.get("questions", []),
                            "questions_success": questions_result.get("success", False)
                        }
                
                # Restore original_agent_request to determine stage
                if hasattr(final_state, 'original_agent_request'):
                    if final_state.original_agent_request:
                        self.current_stage = "requirement_clarification"
            
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            if assistant_response:
                self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            # Check if summarization is needed
            self._check_and_summarize_if_needed()
            
            # Print LLM session summary
            self.print_llm_session_summary()
            
            # Emit final response if no assistant_response event was found
            if assistant_response:
                print(f"🔍 Emitting final assistant_response event")
                yield {
                    "type": "assistant_response",
                    "message": assistant_response,
                    "instruction": current_instruction,
                    "recommended_tools": recommended_tools,
                    "needs_clarification": needs_clarification,
                    "questions": questions
                }
        
        except Exception as e:
            logger.error(f"❌ Error in process_message_stream: {e}")
            yield {
                "type": "error",
                "error": "ProcessingError",
                "message": str(e)
            }


    async def _handle_general_chat(self, user_input: str, full_context: Optional[Dict] = None) -> Dict:
        """Handle general conversation"""
        # Build conversation history context using reusable helper
        conversation_context = self._build_conversation_context_string(full_context, max_messages=5, include_clarifications=False)
        if conversation_context:
            conversation_context = f"\n        {conversation_context.replace(chr(10), chr(10) + '        ')}\n        "
        
        chat_prompt = f"""
        {conversation_context}
        User message: "{user_input}"
        
        Current agent being configured: {self._get_current_state_summary()}
        
        Respond helpfully to the user while keeping the conversation focused on agent configuration.
        If they seem to want to modify the agent, guide them on how to do so.
        """
        
        try:
            with LLMTimingContext("Handle General Chat", self):
                response = await self.llm.ainvoke([HumanMessage(content=chat_prompt)])
                content = response.content.strip() if hasattr(response, 'content') else ""
            
                # For general chat, return the raw text response
                return {
                    "response": content,
                    "needs_clarification": False,
                    "questions": []
                }

        except Exception as e:
                logger.info(f"general chat error: {e}")
                return {
                    "response": "I'm here to help you configure AI agents for your business needs. What would you like your agent to accomplish?",
                    "needs_clarification": False,
                    "questions": []
                }

    async def _generate_clarification_response(self, user_input: str, extracted_requirements: Dict, clarification_questions: Optional[List[str]] = None, streaming_events: Optional[List] = None) -> str:
        """Generate a natural, conversational LLM response that processes the user's clarification input.
        
        Args:
            user_input: The user's clarification input/response
            extracted_requirements: Dict containing role, responsibility, process_steps, etc.
            clarification_questions: Optional list of questions that were asked (for context)
            streaming_events: Optional list to append streaming events to
            
        Returns:
            The generated response text (string)
        """
        # Build context from extracted requirements
        role = extracted_requirements.get("role", "")
        responsibility = extracted_requirements.get("responsibility", "")
        process_steps = extracted_requirements.get("process_steps", [])
        
        requirements_context = ""
        if role or responsibility or process_steps:
            requirements_context = f"""
            Based on what I understand so far about the agent:
            - Role: {role if role else "Not specified"}
            - Responsibility: {responsibility if responsibility else "Not specified"}
            - Process Steps: {json.dumps(process_steps, indent=2) if process_steps else "Not specified"}
            """
        
        # Add context about previous questions if available
        questions_context = ""
        if clarification_questions:
            questions_text = "\n".join([f"- {q}" for q in clarification_questions])
            questions_context = f"""
            Previous clarification questions that were asked:
            {questions_text}
            """
        
        clarification_prompt = f"""
        You are an expert AI assistant helping users configure AI agents. The user has provided clarification information in response to previous questions or user has asked a question .
        
        {requirements_context}
        
        {questions_context}
        
        User's clarification input: "{user_input}"
        
        Generate a natural, conversational, and helpful response that:

        - If the user is clarifying previous questions, follow steps 1-5 below.
        - If the user is asking a new question related to agent instruction or configuration, answer their question thoughtfully using the extracted_requirements context and offer further assistance as needed.
        
        1. Acknowledge and confirm what the user has provided in their clarification or question.
        2. Use the extracted_requirements context to explain how their clarification or question fits into the agent configuration.
        3. Provide helpful feedback, guidance, or next steps based on their input.
        4. If more information is still needed, ask natural follow-up questions.
        5. If enough information is provided, clearly acknowledge that you can proceed with creating or updating the agent instructions.
        
        Be warm, professional, and concise. Show that you understand their clarification and how it relates to building their agent.
        """
        
        try:
            content = ""
            accumulated = ""
            with LLMTimingContext("Generate Clarification Response", self):
                response = await self.llm.ainvoke([HumanMessage(content=clarification_prompt)])
                if hasattr(response, 'content') and response.content:
                    content = response.content
                    accumulated = content
                    
                    # Stream as message_delta if streaming_events list is provided
                    if streaming_events is not None:
                        streaming_events.append({
                            "type": "message_delta",
                            "timestamp": time.time(),
                            "data": {
                                "delta": accumulated,
                                "accumulated": accumulated,
                                "role": "assistant",
                                "content": accumulated
                            }
                        })
                content = accumulated.strip()
            
            # Log the response
            self._log_llm_response(content, "Generate Clarification Response")
            
            return content
            
        except Exception as e:
            logger.error(f"Error generating clarification response: {e}")
            # Fallback response
            return f"Thank you for the clarification. I understand: {user_input}. Let me process this information to help configure your agent."

    async def _detect_if_unrelated_to_agent_role(self, user_input: str, streaming_events: Optional[List] = None) -> Dict:
        """
        Detect if user input is a greeting or unrelated to agent role/responsibility.
        
        Args:
            user_input: The user's input to check
            streaming_events: Optional list to append streaming events to
            
        Returns:
            Dict with:
            - is_unrelated: bool - True if input is greeting or unrelated
            - reason: str - Brief reason for the classification
        """
        detection_prompt = f"""
        Analyze this user input and determine if it is:
        1. A greeting (hi, hello, hey, good morning, etc.)
        2. Unrelated to agent role/responsibility (e.g., asking about weather, general questions, casual conversation)
        3. Related to agent role/responsibility (describing what an agent should do, agent configuration, etc.)
        
        User input: "{user_input}"
        
        Return a JSON object with:
        {{
            "is_unrelated": true/false,
            "reason": "Brief explanation (e.g., 'greeting', 'unrelated question', 'agent-related')"
        }}
        
        IMPORTANT: Only set "is_unrelated" to true if the input is clearly a greeting OR completely unrelated to agent configuration.
        If the input mentions anything about agents, tasks, automation, workflows, or business processes, set it to false.
        """
        
        try:
            content = ""
            accumulated = ""
            with LLMTimingContext("Detect Unrelated Input", self):
                response = await self.llm.ainvoke([HumanMessage(content=detection_prompt)])
                if hasattr(response, 'content') and response.content:
                    content = response.content
                    accumulated = content
                    
                    # Stream as message_delta if streaming_events list is provided
                    if streaming_events is not None:
                        streaming_events.append({
                            "type": "message_delta",
                            "timestamp": time.time(),
                            "data": {
                                "delta": accumulated,
                                "accumulated": accumulated,
                                "role": "assistant",
                                "content": accumulated
                            }
                        })
                content = accumulated.strip()
            
            # Clean up JSON if wrapped in code blocks
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
            
            # Extract JSON from content
            content = self._extract_json_from_content(content)
            result = json.loads(content)
            
            return {
                "is_unrelated": result.get("is_unrelated", False),
                "reason": result.get("reason", "unknown")
            }
            
        except Exception as e:
            logger.warning(f"Error in unrelated input detection: {e}")
            # Default to false (assume related) if detection fails
            return {
                "is_unrelated": False,
                "reason": "detection_error"
            }

    async def _generate_contextual_response_for_unrelated(self, user_input: str, streaming_events: Optional[List] = None) -> str:
        """
        Generate a contextual, friendly response when user input is unrelated to agent role/responsibility.
        
        Args:
            user_input: The user's input that was detected as unrelated
            streaming_events: Optional list to append streaming events to
            
        Returns:
            str: A friendly, contextual response guiding the user to focus on agent role and responsibilities
        """
        response_prompt = f"""
        The user said: "{user_input}"
        
        This input appears unrelated to configuring an agent's role and responsibilities.
        Generate a friendly, empathetic, and helpful response that:
        1. Naturally acknowledges what the user said
        2. Gently encourages them to describe the kind of agent they want to create, focusing on the agent's role and responsibilities
        3. Gives a practical example or two, such as "Create a meeting scheduling agent" or "Build an agent that organizes emails," to illustrate the type of information that's helpful
        4. Keeps a warm, approachable tone

        The response must be concise (2-3 sentences), sound natural (not robotic or overly formal), and provide at least one example relevant to agent roles like "Create a meeting scheduling agent" or similar.

        Return ONLY the response text, no JSON, no quotes, and no extra formatting.
        """
        print(f"Resp_generate_contextual_response_for_unrelated Prompt: {response_prompt}")
        try:
            content = ""
            accumulated = ""
            with LLMTimingContext("Generate Contextual Response", self):
                response = await self.llm.ainvoke([HumanMessage(content=response_prompt)])
                if hasattr(response, 'content') and response.content:
                    content = response.content
                    accumulated = content
                    
                    # Stream as message_delta if streaming_events list is provided
                    if streaming_events is not None:
                        streaming_events.append({
                            "type": "message_delta",
                            "timestamp": time.time(),
                            "data": {
                                "delta": accumulated,
                                "accumulated": accumulated,
                                "role": "assistant",
                                "content": accumulated
                            }
                        })
                content = accumulated.strip()
            
            return content.strip()
            
        except Exception as e:
            logger.warning(f"Error generating contextual response: {e}")
            # Fallback to a friendly default message
            return "I'm here to help you configure an AI agent! Please describe what role and responsibilities you'd like your agent to have. For example, you could say 'Create an agent that handles customer support inquiries' or 'I need an agent to process invoices'."

    async def analyze_agent_role_and_responsibility(self, agent_description_or_name: str, streaming_events: Optional[List] = None, full_context: Optional[Dict] = None) -> Dict:
        """
        Agent Role & Responsibility Analyzer.
        
        Analyzes an agent description or name to identify:
        - The purpose of the agent
        - Its core responsibilities
        - The type of information it needs to collect from the user
        - The expected workflow or actions it should perform
        
        Args:
            agent_description_or_name: The agent's description or name to analyze
            
        Returns:
            Dict containing structured analysis with:
            - purpose: The primary purpose/goal of the agent
            - core_responsibilities: List of core responsibilities
            - required_user_information: List of information types needed from users
            - expected_workflow: List of expected actions/workflow steps
            - role: agent's role
            - responsibilities: agent's responsibilities
            - analysis_summary: Brief summary of the analysis
        """
        
        # Early detection: Check if input is greeting or unrelated to agent role/responsibility
        detection_result = await self._detect_if_unrelated_to_agent_role(agent_description_or_name, streaming_events)
        
        if detection_result.get("is_unrelated", False):
            # Generate contextual response for unrelated input
            response_message = await self._generate_contextual_response_for_unrelated(agent_description_or_name, streaming_events)
            
            # Return early with same structure but empty/default values and response_message
            # Do not store unrelated response in analysis_summary - keep it empty
            return {
                "purpose": "",
                "role": "",
                "responsibility": "",
                "core_responsibilities": [],
                "required_user_information": [],
                "expected_workflow": [],
                "analysis_summary": "",
                "success": True,
                "questions": [],
                "needs_clarification": False,
                "response_message": response_message
            }
        
        # Build conversation history context using reusable helper
        conversation_context = self._build_conversation_context_string(full_context, max_messages=10)
        if conversation_context:
            conversation_context = f"\n        {conversation_context.replace(chr(10), chr(10) + '        ')}\n        "
        
        analysis_prompt = f"""
        You are an Agent Configuration Analyzer. Your job is to understand the role and responsibilities of an agent from a given description or name.

        {conversation_context}
        Agent Description/Name: "{agent_description_or_name}"

        Based on the above, analyze and provide a comprehensive breakdown of:
        1. The purpose of the agent - What is its primary goal or objective?
        2. Core responsibilities - What are the main tasks and duties this agent should handle?
        3. Required user information - What type of information does this agent need to collect from users to function effectively?
        4. Expected workflow - What actions or workflow steps should this agent perform? Your role is to create practical, real-world workflows that describe how the given agent will actually operate step by step in execution — not just conceptual responsibilities
        5. Role - What is the role of this agent? (Short description of the agent's role)
        6. Responsibility - What is the primary responsibility of this agent? (Short description of the agent's main responsibility)
        7. Clarification Need Assessment - Determine if clarification is needed:
           - Set "needs_clarification" to TRUE if ANY of the following apply:
             * The agent description is vague, generic, or lacks specific details about its primary function
             * Critical information is missing (e.g., target users, data sources, integration requirements, business rules)
             * The description contains ambiguities that prevent you from understanding HOW the agent should operate
             * Required context is unclear (e.g., what external systems it interacts with, what data formats it uses)
             * The workflow cannot be determined with confidence from the description alone
             * The agent's scope or boundaries are undefined or unclear
             * Specific use cases, constraints, or edge cases are not mentioned but would be essential for configuration
           - Set "needs_clarification" to FALSE only if:
             * The agent description is specific, detailed, and provides clear context
             * You can confidently determine the agent's purpose, workflow, and requirements
             * All essential information needed to configure the agent is present or can be reasonably inferred
             * The description clearly explains what the agent does, how it operates, and what it needs
           - Be strict: When in doubt, set needs_clarification to TRUE. It's better to ask for clarification than to make incorrect assumptions.
        8. On a scale of 0.0 to 1.0, how clear and complete is the current requirement or description? Provide this number (float between 0.0 and 1.0) as "requirement_clarity_score" (1.0 = perfectly clear and nothing missing, 0.0 = nothing is clear).
           - This score should correlate with needs_clarification: scores below 0.7 typically indicate clarification is needed.

        Return a single JSON object with these keys (keep names exactly as below):
        {{
            "purpose": "Clear statement of the agent's primary purpose and goal",
            "core_responsibilities": [
                "Responsibility 1",
                "Responsibility 2",
                "Responsibility 3"
            ],
            "required_user_information": [
                "Type of information 1 (e.g., user preferences, contact details, task requirements)",
                "Type of information 2",
                "Type of information 3"
            ],
            "expected_workflow": [
                "Description of first action/workflow step",
                "Description of second action/workflow step",
                "Description of third action/workflow step"
            ],
            "analysis_summary": "Brief comprehensive summary of the agent's role, purpose, and how it should operate",
            "role": "Short description of the agent's role (e.g., 'Customer Support Agent', 'Data Analyst Assistant')",
            "responsibility": "Short description of the agent's primary responsibility (e.g., 'Handle customer inquiries and resolve issues')",
            "needs_clarification": true,
            "requirement_clarity_score": 0.9
        }}

        IMPORTANT GUIDELINES FOR "needs_clarification":
        - "needs_clarification" (boolean): CRITICAL FIELD - Determines if the user needs to provide more information before agent configuration can proceed.
        
        **MANDATORY RULE: You MUST set "needs_clarification" to TRUE if ANY of the following conditions apply:**
        1. requirement_clarity_score is less than 0.7 (if score < 0.7, needs_clarification MUST be TRUE)
        2. You cannot confidently answer ALL of these questions from the description:
          * What is the agent's primary function and what problem does it solve?
          * Who are the target users or stakeholders?
          * What are the specific steps/workflows the agent follows?
          * What inputs does the agent need and in what format?
          * What outputs does the agent produce and in what format?
          * What external systems, APIs, or services does it interact with?
          * What are the business rules, constraints, or edge cases?
          * What success criteria or metrics define the agent's effectiveness?
        3. The agent description is vague, generic, or lacks specific details (e.g., "customer service agent", "data assistant", "helpful bot")
        4. You had to make significant assumptions or inferences to understand the requirements
        5. Critical information is missing (target users, data sources, integration requirements, business rules, etc.)
        6. The description contains ambiguities that prevent you from understanding HOW the agent should operate
        
        **You may ONLY set "needs_clarification" to FALSE if ALL of the following are true:**
        - requirement_clarity_score is 0.7 or higher
        - The description is specific, detailed, and provides clear context
        - You can confidently determine the agent's purpose, workflow, and requirements
        - All essential information needed to configure the agent is present or can be reasonably inferred
        - The description clearly explains what the agent does, how it operates, and what it needs
        - You did not need to make significant assumptions
        
        **CRITICAL: The relationship between requirement_clarity_score and needs_clarification:**
        - If requirement_clarity_score < 0.7, then needs_clarification MUST be TRUE (no exceptions)
        - If requirement_clarity_score >= 0.9, you may consider needs_clarification = FALSE (but only if ALL other conditions are met)
        - If requirement_clarity_score is between 0.7 and 0.9, you should carefully evaluate based on the completeness checklist above
        
        - "requirement_clarity_score" (float): How clear and complete are the instructions or requirements? 
          * 1.0 = perfectly clear, nothing missing
          * 0.8-0.9 = mostly clear, minor details may be missing
          * 0.7 = some gaps present, clarification beneficial
          * < 0.7 = significant gaps, clarification required (needs_clarification MUST be TRUE)

        Be strict and conservative. When in doubt, set needs_clarification to TRUE. It's better to ask for clarification than to make incorrect assumptions.
        """
        
        content = ""
        accumulated = ""
        try:
            # Log the prompt
            self._log_llm_prompt([HumanMessage(content=analysis_prompt)], "Agent Role & Responsibility Analysis")
            
            with LLMTimingContext("Agent Role & Responsibility Analysis", self):
                response = await self.llm.ainvoke([HumanMessage(content=analysis_prompt)])
                if hasattr(response, 'content') and response.content:
                    content = response.content
                    accumulated = content
                    
                    # Stream as message_delta if streaming_events list is provided
                    if streaming_events is not None:
                        streaming_events.append({
                            "type": "message_delta",
                            "timestamp": time.time(),
                            "data": {
                                "delta": accumulated,
                                "accumulated": accumulated,
                                "role": "assistant",
                                "content": accumulated
                            }
                        })
                content = accumulated.strip()
            
            # Log the response
            self._log_llm_response(content, "Agent Role & Responsibility Analysis")
            
            # Clean up JSON if wrapped in code blocks
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
            
            # Parse JSON response
            analysis_result = json.loads(content)
            
            # Store analysis result for later use
            self.analysis_result = analysis_result
            
            logger.info(f"Agent Role & Responsibility Analysis completed for: {agent_description_or_name}")
            # Step 2: Generate configuration questions from the analysis
            self.questions_result = await self.generate_questions(analysis_result)
            questions_list = self.questions_result.get("questions", [])
            # print(f"Questions Result: {self.questions_result}")

            # Include needs_clarification in output with programmatic enforcement
            needs_clarification = analysis_result.get("needs_clarification", False)
            requirement_clarity_score = analysis_result.get("requirement_clarity_score", 1.0)
            
            # Programmatic enforcement: If clarity score is low, force needs_clarification to True
            # This ensures consistency even if LLM doesn't follow the prompt correctly
            if requirement_clarity_score < 0.7:
                logger.info(f"Enforcing needs_clarification=True due to low clarity score: {requirement_clarity_score}")
                needs_clarification = True
            
            # Also enforce based on presence of questions - if questions were generated, likely needs clarification
            if questions_list and len(questions_list) > 0 and not needs_clarification:
                logger.info(f"Enforcing needs_clarification=True due to presence of {len(questions_list)} clarification questions")
                needs_clarification = True

            return {
                "purpose": analysis_result.get("purpose", ""),
                "role": analysis_result.get("role", ""),
                "responsibility": analysis_result.get("responsibility", ""),
                "core_responsibilities": analysis_result.get("core_responsibilities", []),
                "required_user_information": analysis_result.get("required_user_information", []),
                "expected_workflow": analysis_result.get("expected_workflow", []),
                "analysis_summary": analysis_result.get("analysis_summary", ""),
                "success": True,
                "questions": questions_list,
                "needs_clarification": needs_clarification
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in agent role analysis: {e}")
            if content:
                logger.error(f"Raw LLM response: {content}")
            return {
                "purpose": "",
                "role": "",
                "responsibility": "",
                "core_responsibilities": [],
                "required_user_information": [],
                "expected_workflow": [],
                "analysis_summary": f"Error parsing analysis response: {str(e)}",
                "success": False,
                "error": "json_parse_error",
                "needs_clarification": True
            }
        except Exception as e:
            logger.error(f"Error in agent role and responsibility analysis: {e}")
            return {
                "purpose": "",
                "role": "",
                "responsibility": "",
                "core_responsibilities": [],
                "required_user_information": [],
                "expected_workflow": [],
                "analysis_summary": f"Error during analysis: {str(e)}",
                "success": False,
                "error": str(e),
                "needs_clarification": True
            }

    async def generate_questions(self, analysis_result: Dict, streaming_events: Optional[List] = None) -> Dict:
        """
        Agent Configuration Question Generator.
        
        Takes a structured analysis result from analyze_agent_role_and_responsibility and generates
        a list of questions that must be asked to the user to gather all missing details required
        to configure or build the agent completely.
        
        Args:
            analysis_result: Dictionary containing structured analysis with:
                - purpose: The primary purpose/goal of the agent
                - core_responsibilities: List of core responsibilities
                - required_user_information: List of information types needed from users
                - expected_workflow: List of expected actions/workflow steps
                - analysis_summary: Brief summary of the analysis
                - success: Boolean indicating if analysis was successful
                
        Returns:
            Dict containing:
                - questions: List of user-facing questions to gather missing information
                - success: Boolean indicating if question generation was successful
        """
        
        # Extract components from analysis result
        purpose = analysis_result.get("purpose", "")
        core_responsibilities = analysis_result.get("core_responsibilities", [])
        required_user_information = analysis_result.get("required_user_information", [])
        expected_workflow = analysis_result.get("expected_workflow", [])
        analysis_summary = analysis_result.get("analysis_summary", "")
        agent_role = analysis_result.get("role", "")
        responsibility = analysis_result.get("responsibility", "")
        
        question_generation_prompt = f"""
        You are an Agent Configuration Question Designer.
        Your responsibility is to create role-specific configuration questions that help gather all essential information from the user to fully configure and operationalize a given agent.

        All questions must be exclusively related to the agent's defined role, purpose, and workflow — they should help fill the missing configuration details required for real-world functionality.
        Focus on questions that identify missing elements needed for the agent to function properly.

        Agent Role:
        You are creating configuration questions for a role: {agent_role} and responsibilities: {responsibility}.

        Carefully review the following JSON input describing the agent's purpose, core responsibilities, required user information, and workflow:
        {{
          "purpose": "{purpose}",
          "core_responsibilities": {json.dumps(core_responsibilities)},
          "required_user_information": {json.dumps(required_user_information)},
          "expected_workflow": {json.dumps(expected_workflow)},
          "analysis_summary": "{analysis_summary}"
        }}

        Your Task:

        Review the input JSON carefully.
        Focus only on the agent role mentioned — all generated questions must directly support this role's functions and responsibilities.
        
        CRITICAL EXCLUSIONS - DO NOT ASK ABOUT:
        - Integration methods or authentication (e.g., "How should the agent access calendars?", "What authentication method?", "Which API keys?") - These are handled by the app infrastructure
        - Technical setup details that the app manages automatically
        - Less critical configuration like reminder settings, notification preferences (unless they are core to the agent's primary function)
        
        Priority Order (generate questions in this priority):
        1. Tool-related questions - Which tools/systems the agent needs to use, what data sources to access, what actions to perform (e.g., "Which calendar should the agent monitor?", "What information should be extracted?", "What actions should be taken?")
        2. Main task/role questions - Essential questions about the agent's primary function, triggers, and behavior
        3. Main missing elements - Critical information gaps that prevent the agent from functioning (coordination needs, complexity requirements)
        4. Missing details - Important details needed for proper configuration (communication preferences, specific requirements)
        5. Edge cases - Important edge cases or special scenarios the agent should handle
        
        IMPORTANT CONSTRAINTS:
        - Generate 5 questions maximum. Ask questions that are directly related to the agent's workflow.
        - Prioritize tool-related questions first - what tools/systems/data the agent needs to interact with
        - Then prioritize questions based on the agent's expected workflow
        - Focus on questions that directly impact the agent's core functionality and identify missing elements
        - Address the most critical gaps in the workflow
        - Avoid redundant or overlapping questions
        - Do NOT ask about integration/auth methods or technical setup handled by the app
        - Do NOT ask about less critical preferences unless they are essential to the agent's primary function
        
        Transform each item from the agent's expected workflow into clear, user-facing questions that collect actionable configuration information about the agent's behavior and requirements.
        Ensure all questions are specific, realistic, and focused on what the agent needs to know to operate effectively — phrased as if an onboarding chat or configuration UI were collecting the data.

        Output format:
        {{
          "questions": [
            "Question 1 (highest priority - tool/system/data source needed)",
            "Question 2 (main task/role/trigger or another tool question)",
            "Question 3 (main missing element - coordination/complexity)"
          ]
        }}
        
        Remember: Questions are prioritized by importance to the agent's core functionality. Prioritize tool-related questions first, then focus on identifying missing elements needed for the agent to function, not technical setup.
        """
# MAXIMUM 8 Maximum 8
        content = ""
        accumulated = ""
        try:
            # Log the prompt
            self._log_llm_prompt([HumanMessage(content=question_generation_prompt)], "Generate Configuration Questions")
            
            with LLMTimingContext("Generate Configuration Questions", self):
                response = await self.llm.ainvoke([HumanMessage(content=question_generation_prompt)])
                if hasattr(response, 'content') and response.content:
                    content = response.content
                    accumulated = content
                    
                    # Stream as message_delta if streaming_events list is provided
                    if streaming_events is not None:
                        streaming_events.append({
                            "type": "message_delta",
                            "timestamp": time.time(),
                            "data": {
                                "delta": accumulated,
                                "accumulated": accumulated,
                                "role": "assistant",
                                "content": accumulated
                            }
                        })
                content = accumulated.strip()
            
            # Log the response
            self._log_llm_response(content, "Generate Configuration Questions")
            
            # Clean up JSON if wrapped in code blocks
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
            
            # Extract JSON from content (handles cases where JSON is embedded in text)
            content = self._extract_json_from_content(content)
            
            # Parse JSON response with improved error handling
            result = None
            try:
                result = json.loads(content)
            except json.JSONDecodeError as parse_error:
                logger.warning(f"Initial JSON parse failed, trying to extract JSON from content: {parse_error}")
                # Try to extract JSON more aggressively
                # Look for content between first { and last }
                first_brace = content.find('{')
                last_brace = content.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    potential_json = content[first_brace:last_brace+1]
                    try:
                        result = json.loads(potential_json)
                        logger.info(f"Successfully extracted JSON from content after initial parse failure")
                    except json.JSONDecodeError as second_error:
                        logger.warning(f"Second JSON parse attempt also failed: {second_error}")
                        # Try to fix common JSON issues
                        # Remove trailing commas before closing braces/brackets
                        fixed_json = re.sub(r',\s*}', '}', potential_json)
                        fixed_json = re.sub(r',\s*]', ']', fixed_json)
                        try:
                            result = json.loads(fixed_json)
                            logger.info(f"Successfully parsed JSON after fixing trailing commas")
                        except json.JSONDecodeError:
                            # All JSON parsing attempts failed
                            logger.error(f"All JSON parsing attempts failed, raising error")
                            raise second_error
                else:
                    # No JSON structure found
                    logger.error(f"No JSON structure found, raising parse error")
                    raise parse_error
            
            questions = result.get("questions", [])
            
            # Limit to maximum 8 questions and filter out empty/invalid questions
            questions = [q.strip() for q in questions if q and q.strip()][:8]
            
            logger.info(f"Generated {len(questions)} configuration questions from analysis")
            
            return {
                "questions": questions,
                "success": True
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in configuration question generation: {e}")
            error_pos = getattr(e, 'pos', None)
            error_msg = getattr(e, 'msg', str(e))
            if error_pos is not None:
                logger.error(f"Error at position {error_pos}: {error_msg}")
                # Show context around the error
                if error_pos < len(content):
                    start = max(0, error_pos - 100)
                    end = min(len(content), error_pos + 100)
                    context = content[start:end]
                    logger.error(f"Content around error position:\n{context}")
            if content:
                logger.error(f"Raw LLM response (first 500 chars): {content[:500]}")
                logger.error(f"Raw LLM response (last 500 chars): {content[-500:] if len(content) > 500 else content}")
            return {
                "questions": [],
                "success": False,
                "error": "json_parse_error"
            }
        except Exception as e:
            logger.error(f"Error in configuration question generation: {e}")
            return {
                "questions": [],
                "success": False,
                "error": str(e)
            }

    async def get_answers_for_agent_role_questions(self, user_input: str, agent_summary: Dict, questions_list: List[str], streaming_events: Optional[List] = None) -> Dict:
        """
        Analyze user answers to clarification questions.
        
        This method:
        1. Analyzes the user input to identify answers
        2. Matches answers to the most relevant questions from the questions list
        3. Reconstructs/updates the agent summary based on the answers
        4. Determines remaining questions, clarity score, and if clarification is still needed
        
        Args:
            user_input: The user's response containing answers to clarification questions
            agent_summary: Dictionary containing the current agent analysis/summary with:
                - purpose: Agent's purpose
                - role: Agent's role
                - responsibility: Agent's responsibility
                - core_responsibilities: List of core responsibilities
                - required_user_information: List of required information types
                - expected_workflow: List of workflow steps
                - analysis_summary: Summary of the analysis
            questions_list: List of clarification questions that were asked
        
        Returns:
            Dict containing:
                - updated_agent_summary: Updated agent summary incorporating answers
                - answered_questions: List of questions that were answered (with their answers)
                - remaining_questions: List of questions that still need answers (max 8)
                - clarity_score: Float between 0-1 indicating how clear the configuration is
                - needs_clarification: Boolean indicating if more clarification is needed
                - success: Boolean indicating if processing was successful
        """
        
        answer_analysis_prompt = f"""
        You are an Agent Configuration Answer Analyzer. Your job is to:
        1. Analyze user input to extract answers to clarification questions
        2. Match each answer to the most relevant question from the provided list
        3. Update the agent summary with the information provided in the answers
        4. Identify which questions remain unanswered
        5. Calculate a clarity score and determine if more clarification is needed
        
        Current Agent Summary:
        {{
            "purpose": "{agent_summary.get('purpose', '')}",
            "role": "{agent_summary.get('role', '')}",
            "responsibility": "{agent_summary.get('responsibility', '')}",
            "core_responsibilities": {json.dumps(agent_summary.get('core_responsibilities', []))},
            "required_user_information": {json.dumps(agent_summary.get('required_user_information', []))},
            "expected_workflow": {json.dumps(agent_summary.get('expected_workflow', []))},
            "analysis_summary": "{agent_summary.get('analysis_summary', '')}"
        }}
        
        Questions Asked:
        {json.dumps(questions_list, indent=2)}
        
        User Input (Answers):
        "{user_input}"
        
        Your Task:
        1. Analyze the user input and identify which questions from the list are being answered
        2. For each answer, determine which question it relates to (match to the most relevant question)
        3. Extract the key information from each answer
        4. Update the agent summary by incorporating the information from the answers
        5. Identify questions that remain unanswered or unclear
        6. Calculate a clarity score (0.0 to 1.0) based on:
           - How many questions were answered
           - How complete and clear the answers are
           - How much information is available to configure the agent
        7. Determine if more clarification is needed (needs_clarification: true if clarity_score < 0.8 or if critical questions remain unanswered)
        
        Return a JSON object with these keys (keep names exactly as below):
        {{
            "updated_agent_summary": {{
                "purpose": "Updated purpose based on answers",
                "role": "Updated role based on answers",
                "responsibility": "Updated responsibility based on answers",
                "core_responsibilities": ["Updated responsibilities"],
                "required_user_information": ["Updated required information"],
                "expected_workflow": ["Updated workflow steps"],
                "analysis_summary": "Updated comprehensive summary incorporating answers"
            }},
            "answered_questions": [
                {{
                    "question": "Question text",
                    "answer": "Extracted answer from user input",
                    "relevance_score": 0.9
                }}
            ],
            "remaining_questions": [
                "Question 1 that still needs an answer",
                "Question 2 that still needs an answer"
            ],
            "clarity_score": 0.85,
            "needs_clarification": false,
            "analysis": "Brief analysis of what was clarified and what remains unclear"
        }}
        
        Important Guidelines:
        - Match answers to questions intelligently - a single user input may answer multiple questions
        - If user input doesn't clearly answer any question, mark them as remaining questions
        - Update the agent summary by incorporating new information from answers
        - Limit remaining_questions to maximum 8 (prioritize the most critical ones)
        - Calculate clarity_score based on completeness: 0.0 = no information, 1.0 = fully clear
        - Set needs_clarification to true if clarity_score < 0.8 or critical information is missing
        - Be specific and actionable in the updated summary
        """
        
        content = ""
        accumulated = ""
        try:
            # Log the prompt
            self._log_llm_prompt([HumanMessage(content=answer_analysis_prompt)], "Analyze Clarification Answers")
            
            with LLMTimingContext("Analyze Clarification Answers", self):
                response = await self.llm.ainvoke([HumanMessage(content=answer_analysis_prompt)])
                if hasattr(response, 'content') and response.content:
                    content = response.content
                    accumulated = content
                    
                    # Stream as message_delta if streaming_events list is provided
                    if streaming_events is not None:
                        streaming_events.append({
                            "type": "message_delta",
                            "timestamp": time.time(),
                            "data": {
                                "delta": accumulated,
                                "accumulated": accumulated,
                                "role": "assistant",
                                "content": accumulated
                            }
                        })
                content = accumulated.strip()
            
            # Log the response
            self._log_llm_response(content, "Analyze Clarification Answers")
            
            # Clean up JSON if wrapped in code blocks
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
            
            # Parse JSON response
            result = json.loads(content)
            
            # Extract and limit remaining questions to max 8
            remaining_questions = result.get("remaining_questions", [])
            remaining_questions = [q.strip() for q in remaining_questions if q and q.strip()][:8]
            
            logger.info(f"Processed clarification answers: {len(result.get('answered_questions', []))} answered, {len(remaining_questions)} remaining")
            
            return {
                "updated_agent_summary": result.get("updated_agent_summary", agent_summary),
                "answered_questions": result.get("answered_questions", []),
                "remaining_questions": remaining_questions,
                "clarity_score": result.get("clarity_score", 0.0),
                "needs_clarification": result.get("needs_clarification", True),
                "analysis": result.get("analysis", ""),
                "success": True
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in clarification answer analysis: {e}")
            if content:
                logger.error(f"Raw LLM response: {content}")
            return {
                "updated_agent_summary": agent_summary,
                "answered_questions": [],
                "remaining_questions": questions_list[:8],
                "clarity_score": 0.0,
                "needs_clarification": True,
                "analysis": f"Error parsing analysis response: {str(e)}",
                "success": False,
                "error": "json_parse_error"
            }
        except Exception as e:
            logger.error(f"Error in clarification answer analysis: {e}")
            return {
                "updated_agent_summary": agent_summary,
                "answered_questions": [],
                "remaining_questions": questions_list[:8],
                "clarity_score": 0.0,
                "needs_clarification": True,
                "analysis": f"Error during analysis: {str(e)}",
                "success": False,
                "error": str(e)
            }



    @time_method("Get Tools From Categories", log_args=True)
    def get_tools_from_categories(self, tool_categories: Dict, limit: int = 30, user_input: str = ""):
        """Get tools by categories and search queries - OPTIMIZED WITH CACHING"""
        
        logger.info(f"\n🔍 Using advanced tool filtering with categories: {tool_categories}")

        primary_categories = tool_categories.get("primary_categories", [])

        try:
            # Extract categories and required tools from the input
            required_tools = []
            if isinstance(tool_categories, dict):
                required_tools = tool_categories.get("required_tools", [])
            
            logger.info(f"\n🔍 Using direct method call with:")
            logger.info(f"   📂 Primary categories: {primary_categories}")
            logger.info(f"   ⭐ Required tools: {len(required_tools)}")
            if required_tools:
                tool_names = [rt.get('tool_name', 'Unknown') for rt in required_tools[:3]]
                logger.info(f"   🛠️  Tool names: {tool_names}")
            
            # Call the direct method with categories, required tools, and user input
            tools_list = self.filter_tools_from_vector_db(
                categories=tool_categories,
                limit=limit,
                user_input=user_input
            )

            
            logger.info(f"✅ Successfully retrieved {len(tools_list)} tools from direct method")
                
        except Exception as e:
            logger.warning(f"⚠️  Error in direct method call: {e}")
            logger.info(f"   Returning empty tools list")
            tools_list = []

        logger.info(f"   📊 Tools: {len(tools_list)} tools")

        # Create result object
        result_obj = {
            'categories': tool_categories,
            'limit': limit,
            'result_count': len(tools_list),
            'results': tools_list,
            'tools_list': tools_list
        }
               
        return result_obj

    def filter_tools_from_vector_db(self, categories: Dict, limit: int = 30, user_input: str = ""):
        """
        Direct method to get tools by criteria without API call.
        Filters using primary_categories only for category-based filtering.
        """
        try:
            # Validate and fix limit parameter
            if limit is None or limit <= 0:
                logger.warning(f"⚠️  Invalid limit {limit}, setting to default 10")
                limit = 10
            
            # Ensure limit is reasonable
            limit = min(limit, 50)  # Hard cap at 50 tools
            
            collection_name = "chanId-5_name-ComposioTools"
            
            # Get Qdrant client
            qdrant_client_ = get_qdrant_client().__next__()
            
            # Check if collection exists
            collection_exists = qdrant_client_.collection_exists(collection_name)
            
            if not collection_exists:
                logger.warning(f"⚠️  Collection '{collection_name}' not found. Please store the data first.")
                return []
            
            # Get embedding service for semantic search
            embedding_service = None
            query_embedding = None
            
            # Extract filtering criteria
            primary_categories = categories.get("primary_categories", [])
            required_tools = categories.get("required_tools", [])
            keywords = categories.get("keywords", [])
            app_names = categories.get("app_names", [])
            tool_types = categories.get("tool_types", [])
            integration_needs = categories.get("integration_needs", [])

            search_queries = categories.get("search_queries", [])
            query_list = [q.get("query", "") for q in search_queries if q.get("query")]

            # Create user-focused search query prioritizing user input
            search_components = []

            # PRIORITY 1: User input as the main context (most important)
            # if user_input and user_input.strip():
            #     user_context = f"User wants to create an agent that: {user_input.strip()}"
            #     search_components.append(user_context)
            #     print(f"🎯 User-focused search: {user_input.strip()}")

            # PRIORITY 2: Required tools (high priority)
            if required_tools:
                tool_names = [rt.get('tool_name', '') for rt in required_tools if rt.get('tool_name')]
                if tool_names:
                    tools_text = "Specific tools needed: " + ", ".join(tool_names)
                    search_components.append(tools_text)

            # PRIORITY 3: Primary categories (medium priority)
            if primary_categories:
                category_text = "Categories: " + ", ".join(primary_categories)
                search_components.append(category_text)

            # PRIORITY 4: Keywords as capabilities (medium priority)
            if keywords:
                keywords_text = "Capabilities: " + ", ".join(keywords)
                search_components.append(keywords_text)

            # PRIORITY 5: App names for specific app targeting (low priority)
            if app_names:
                apps_text = "Applications: " + ", ".join(app_names)
                search_components.append(apps_text)

            # PRIORITY 6: Tool types (low priority)
            if tool_types:
                types_text = "Tool types: " + ", ".join(tool_types)
                search_components.append(types_text)

            # PRIORITY 7: Integration needs (low priority)
            if integration_needs:
                integration_text = "Integration needs: " + ", ".join(integration_needs)
                search_components.append(integration_text)

            # Create user-focused search query optimized for embeddings
            if search_components:
                search_query = ". ".join(search_components)
                print(f"🔍 User-focused search query: {search_query}")
                
                embedding_model = "text-embedding-3-large"
                provider = EmbeddingModelManager.get_provider(embedding_model)
                provider_auth = get_ai_provider_key(provider, 5)
                embedding_service = EmbeddingService(embedding_model=embedding_model, api_key=provider_auth["api_key"])
                query_embedding = embedding_service.embed_query(search_query)
            
            # Perform search with optimized limits
            if query_embedding:
                # Increased limit to find more relevant tools, especially required tools
                # For 50 tools target, fetch ~50-100 apps maximum
                search_limit = min(100, max(50, limit * 2))  # Increased limit to find more relevant tools
                search_results = qdrant_client_.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    limit=search_limit,
                    with_payload=True
                )
            else:
                # Get limited results if no text criteria - avoid getting all data
                scroll_limit = min(25, max(20, limit // 2))  # Match search_limit
                scroll_results = qdrant_client_.scroll(
                    collection_name=collection_name,
                    limit=scroll_limit,
                    with_payload=True
                )
                search_results = scroll_results[0] if scroll_results else []
            
            # Filter and format results with high priority for required tools
            filtered_tools = []
            required_tools_found = []
            other_tools = []
            seen_tools = set()  # To avoid duplicates
            
            # Extract required tool names for matching (with None safety)
            required_tool_names = []
            if required_tools:
                required_tool_names = [
                    rt.get('tool_name', '') 
                    for rt in required_tools 
                    if rt and rt.get('tool_name') and isinstance(rt.get('tool_name'), str)
                ]
                if required_tool_names:
                    print(f"required_tools: {required_tool_names[:3]}")  # Print first 3 for debugging
            
            for result in search_results:
                payload = result.payload if hasattr(result, 'payload') else result
                
                # Extract app and tool information
                app_name = payload.get('app_name') or ''
                app_description = payload.get('app_description') or ''
                app_categories = payload.get('app_categories', [])
                # tools = payload.get('tools', [])
                
                # # Apply category filters
                # if primary_categories:
                #     # Check if any requested category matches any app category
                #     requested_categories_lower = [cat.lower() for cat in primary_categories if cat]
                #     app_categories_lower = [cat.lower() for cat in app_categories]
                #     if not any(req_cat in app_categories_lower for req_cat in requested_categories_lower):
                #         continue
                
                # Process tools within this app
                # for tool in tools:
                tool_name = payload.get('tool_name') or ''
                tool_description = payload.get('tool_description') or ''
                tool_id = payload.get('tool_id') or ''

                # Create unique identifier for this tool
                # tool_id = f"{app_name}:{tool_name}"
                if tool_id in seen_tools:
                    continue
                seen_tools.add(tool_id)
                
                # Calculate relevance score based on matches
                relevance_score = result.score if hasattr(result, 'score') else 1.0
                
                # Check if this tool matches any required tool name (with None safety)
                is_required_tool = False
                if required_tool_names and tool_name and isinstance(tool_name, str):
                    for req_tool_name in required_tool_names:
                        if req_tool_name and isinstance(req_tool_name, str):
                            req_tool_stripped = req_tool_name.strip()
                            tool_name_stripped = tool_name.strip()
                            
                            # Exact match gets highest priority
                            if req_tool_stripped == tool_name_stripped:
                                relevance_score += 1.0  # Highest priority boost for exact matches
                                is_required_tool = True
                                break
                            # Partial match gets medium priority
                            elif req_tool_stripped in tool_name_stripped or tool_name_stripped in req_tool_stripped:
                                relevance_score += 0.5  # Medium priority boost for partial matches
                                is_required_tool = True
                                break
                
                # Apply keyword filtering to tool description
                if keywords and not is_required_tool and tool_description:
                    print(f"keywords: {keywords}")
                    print(f"tool_description: {tool_description}")
                    tool_desc_lower = tool_description.lower()
                    print(f"tool_desc_lower: {tool_desc_lower}")
                    keyword_match = any(
                        keyword and keyword.lower().strip() in tool_desc_lower
                        for keyword in keywords
                    )
                    if keyword_match:
                        relevance_score += 0.3  # Small boost for keyword matches
                
                # Format tool result
                tool_result = {
                    "app_name": app_name,
                    "app_description": app_description,
                    "app_categories": app_categories,
                    "tool_id": tool_id,
                    "tool_name": tool_name,
                    "tool_description": tool_description,
                    "logo_url": payload.get('logo_url', ''),
                    "toolkit_name": payload.get('toolkit_name', ''),
                    "similarity_score": result.score if hasattr(result, 'score') else 1.0,
                    "relevance_score": relevance_score,
                    "is_required_tool": is_required_tool
                }
                
                # Add additional tool properties if available
                # if 'parameters' in tool:
                #     tool_result["parameters"] = tool['parameters']
                # if 'input_schema' in tool:
                #     tool_result["input_schema"] = tool['input_schema']
                # if 'output_schema' in tool:
                #     tool_result["output_schema"] = tool['output_schema']
                
                # Separate required tools from other tools for priority handling
                if is_required_tool:
                    required_tools_found.append(tool_result)
                else:
                    other_tools.append(tool_result)
            
            # Combine results with required tools first (high priority)
            filtered_tools = required_tools_found + other_tools
            
            # Limit to requested number
            filtered_tools = filtered_tools[:limit]
            
            # Sort by relevance score (combination of similarity and tool matches)
            filtered_tools.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            # Limit to requested number
            final_results = filtered_tools[:limit]
            
            logger.info(f"✅ Direct method retrieved {len(final_results)} tools")
            if final_results:
                print(f"   📊 Top 3 tools: {[t.get('tool_name', 'Unknown')[:20] for t in final_results[:3]]}")
                print(f"   🎯 Relevance scores: {[f'{t.get('relevance_score', 0):.2f}' for t in final_results[:3]]}")
                
                # Show filtering statistics
                required_count = sum(1 for t in final_results if t.get('is_required_tool', False))
                if required_count > 0:
                    print(f"   ⭐ Required tools found: {required_count}/{len(required_tool_names) if required_tool_names else 0}")
                
                # Show category distribution
                if primary_categories:
                    category_dist = {}
                    for tool in final_results:
                        for cat in tool.get('app_categories', []):
                            category_dist[cat] = category_dist.get(cat, 0) + 1
                    print(f"   📈 Category distribution: {dict(list(category_dist.items())[:3])}")
                
                # Show app distribution
                app_dist = {}
                for tool in final_results:
                    app_name = tool.get('app_name', 'Unknown')
                    app_dist[app_name] = app_dist.get(app_name, 0) + 1
                print(f"   🏢 Top apps: {dict(list(app_dist.items())[:3])}")
            return final_results
            
        except Exception as e:
            logger.warning(f"⚠️  Error in filter_tools_from_vector_db method: {str(e)}")
            return []


    @time_method("Check Core Tools Capability", log_args=True)
    async def semantic_search_from_filtered_tools(self, user_input: str, current_instruction: Dict, tools: List[Dict], streaming_events: Optional[List] = None, full_context: Optional[Dict] = None) -> Dict:
        """
        Hybrid tool selection combining vector search with LLM reasoning.

        PHASE 1: Vector search (already done - tools parameter)
        PHASE 2: LLM-based re-ranking and filtering
        
        Args:
            full_context: Full context dictionary containing conversation history and state information
        """

        print(f"All Tools: {len(tools)}", file=sys.stderr)
        # print(f"Tools list: {tools}", file=sys.stderr)

        # Build conversation history context using reusable helper
        conversation_context = self._build_conversation_context_string(full_context, max_messages=5, include_clarifications=False)
        if conversation_context:
            conversation_context = f"\n        {conversation_context.replace(chr(10), chr(10) + '        ')}\n        "

        prompt = f"""
        {conversation_context}
        You are an expert tool selection assistant. Your task is to find the MOST RELEVANT tools for the user's specific request.
        
        USER'S SPECIFIC REQUEST: "{user_input}"
        
        CURRENT AGENT INSTRUCTION: {self._get_current_state_summary()}
        
        AVAILABLE TOOLS ({len(tools)} total):
        {json.dumps(tools, indent=2)}
        
        INSTRUCTIONS:
        1. Focus ONLY on tools that directly help fulfill the user's specific request: "{user_input}"
        2. Ignore tools that are not relevant to the user's actual needs
        3. Prioritize tools that match the specific functionality described in the user input
        4. Consider the role and process steps, but the user input is the PRIMARY context
        5. Select 3-8 tools maximum that are most directly relevant
        
        EVALUATION CRITERIA:
        - Does this tool directly help with what the user asked for?
        - Is this tool essential for the specific task described?
        - Would this tool be used in the process steps mentioned?
        - Is this tool mentioned or implied in the user's request?
        
        Return JSON with:
        1. can_fulfill: boolean - can the selected tools handle the user's specific request?
        2. recommended_tools: array of 3-8 most relevant tools (only tools that directly help with the user's request)
           - Each tool MUST be a full object with all properties from the AVAILABLE TOOLS list above
           - Include tool_id, tool_name, app_name, and all other properties from the original tool object
           - DO NOT return just tool_id strings - return the complete tool objects
        3. missing_capabilities: array of specific capabilities not covered by selected tools
        4. alternative_approach: string suggesting how to handle the user's request with selected tools
        5. confidence_score: number between 0-1 indicating confidence in the recommendations
        
        IMPORTANT: Only recommend tools that are directly relevant to the user's specific request. Quality over quantity.
        CRITICAL: recommended_tools must contain full tool objects (not just tool_id strings) - copy the complete object from AVAILABLE TOOLS.
        
        Return ONLY valid JSON with these exact keys: can_fulfill, recommended_tools, missing_capabilities, alternative_approach, confidence_score
        """

        try:
            # Log the prompt
            self._log_llm_prompt([HumanMessage(content=prompt)], "Semantic Search From Filtered Tools")
            
            content = ""
            accumulated = ""
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            if hasattr(response, 'content') and response.content:
                content = response.content
                accumulated = content
                
                # Stream as message_delta if streaming_events list is provided
                if streaming_events is not None:
                    streaming_events.append({
                        "type": "message_delta",
                        "timestamp": time.time(),
                        "data": {
                            "delta": accumulated,
                            "accumulated": accumulated,
                            "role": "assistant",
                            "content": accumulated
                        }
                    })
            content = accumulated.strip()
            
            # Log the response
            self._log_llm_response(content, "Semantic Search From Filtered Tools")
            
            # Extract JSON from ```json code blocks
            content = self._extract_json_from_content(content)
            
            result = json.loads(content)

            # Merge LLM rankings with original tool data
            ranked_tools = []
            for llm_tool in result.get("recommended_tools", []):
                # Handle case where LLM returns strings (tool_ids) instead of full objects
                if isinstance(llm_tool, str):
                    # If it's a string, treat it as a tool_id and find the matching tool
                    tool_id = llm_tool
                    original_tool = next(
                        (t for t in tools if t.get("tool_id") == tool_id),
                        None
                    )
                    if original_tool:
                        # Create a copy to avoid modifying original
                        enhanced_tool = original_tool.copy()
                        enhanced_tool["confidence_score"] = 0.5  # Default confidence for string matches
                        enhanced_tool["reasoning"] = ""
                        enhanced_tool["status"] = "draft"  # Set initial status
                        ranked_tools.append(enhanced_tool)
                    else:
                        logger.warning(f"Could not find original tool for LLM recommendation (string tool_id): {tool_id}")
                    continue
                
                # Handle case where LLM returns full objects
                # Skip tools with empty or invalid data
                if not isinstance(llm_tool, dict):
                    logger.warning(f"Skipping invalid tool format (expected dict or string): {llm_tool}")
                    continue
                    
                if not llm_tool.get("tool_name") and not llm_tool.get("app_name") and not llm_tool.get("tool_id"):
                    logger.warning(f"Skipping tool with empty tool_name, app_name, and tool_id: {llm_tool}")
                    continue
                
                # Find original tool with full metadata using multiple matching strategies
                original_tool = None
                
                # Strategy 1: Match by tool_id if available
                if llm_tool.get("tool_id"):
                    original_tool = next(
                        (t for t in tools if t.get("tool_id") == llm_tool.get("tool_id")),
                        None
                    )
                
                # Strategy 2: Match by tool_name and app_name if tool_id fails
                if not original_tool and llm_tool.get("tool_name") and llm_tool.get("app_name"):
                    original_tool = next(
                        (t for t in tools 
                         if t.get("tool_name") == llm_tool.get("tool_name") 
                         and t.get("app_name") == llm_tool.get("app_name")),
                        None
                    )
                
                # Strategy 3: Match by app_name only if other strategies fail
                if not original_tool and llm_tool.get("app_name"):
                    original_tool = next(
                        (t for t in tools if t.get("app_name") == llm_tool.get("app_name")),
                        None
                    )
                
                if original_tool:
                    # Create a copy to avoid modifying original
                    enhanced_tool = original_tool.copy()
                    enhanced_tool["confidence_score"] = llm_tool.get("confidence_score", 0.5)
                    enhanced_tool["reasoning"] = llm_tool.get("reasoning", "")
                    enhanced_tool["status"] = "draft"  # Set initial status
                    ranked_tools.append(enhanced_tool)
                else:
                    logger.warning(f"Could not find original tool for LLM recommendation: {llm_tool.get('app_name', 'Unknown')} - {llm_tool.get('tool_name', 'Unknown')}")

            return {
                "recommended_tools": ranked_tools,
                "can_fulfill": result.get("can_fulfill", True),
                "missing_capabilities": result.get("missing_capabilities", []),
                "overall_confidence": result.get("overall_confidence", 0.5),
                "alternative_approach": f"Use the {len(ranked_tools)} recommended tools to fulfill requirements"
            }

        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ JSON parsing error in hybrid tool ranking: {e}")
            # Specially handle the specific error for empty JSON content
            if "Expecting value: line 1 column 1 (char 0)" in str(e):
                msg_cap = "No structured response from LLM - received empty content. Please retry, or refine agent instructions."
            else:
                msg_cap = "JSON parsing error occurred"
            return {
                "can_fulfill": len(tools) > 0,
                "recommended_tools": tools[:10],  # Take first 10 as fallback
                "missing_capabilities": [msg_cap],
                "overall_confidence": 0.3,
                "alternative_approach": "Using vector search results as fallback"
            }
        except Exception as e:
            logger.warning(f"⚠️ Hybrid tool ranking error: {e}")
            # Fallback: return basic response
            return {
                "can_fulfill": len(tools) > 0,
                "recommended_tools": tools[:10],  # Take first 10 as fallback
                "missing_capabilities": ["Error occurred during analysis"],
                "overall_confidence": 0.3,
                "alternative_approach": "Using vector search results as fallback"
            }
    
    def print_timing_summary(self):
        """Print a summary of timing information for the current session"""
        if hasattr(self, 'llm_session_start_time') and hasattr(self, 'llm_session_end_time'):
            total_duration = self.llm_session_end_time - self.llm_session_start_time
            logger.info(f"\n📊 TIMING SUMMARY:")
            logger.info(f"   Total Session Duration: {total_duration:.3f}s")
            
            if hasattr(self, 'llm_total_time'):
                llm_percentage = (self.llm_total_time / total_duration) * 100 if total_duration > 0 else 0
                logger.info(f"   LLM Calls Time: {self.llm_total_time:.3f}s ({llm_percentage:.1f}%)")
                
            if hasattr(self, 'vector_search_time'):
                vector_percentage = (self.vector_search_time / total_duration) * 100 if total_duration > 0 else 0
                logger.info(f"   Vector Search Time: {self.vector_search_time:.3f}s ({vector_percentage:.1f}%)")
                
            logger.info(f"   Other Operations: {total_duration - getattr(self, 'llm_total_time', 0) - getattr(self, 'vector_search_time', 0):.3f}s")
            logger.info("=" * 50)


    @time_method("Get Assistant Reply", log_args=True)
    async def get_assistant_response_stream(self, current_instruction: Dict[str, Any], previous_instruction: Dict[str, Any], user_input: str, recommended_tools: List[Dict], full_context: Optional[Dict] = None):
        """
        Stream assistant reply as message_delta events for CopilotKit.
        Yields message_delta events as LLM generates the response.
        
        Args:
            full_context: Full context dictionary containing conversation history and state information
        """
        # Safely convert instructions to dict format
        try:
            if hasattr(current_instruction, 'model_dump'):
                current_instruction = current_instruction.model_dump()
            elif hasattr(current_instruction, 'dict'):
                current_instruction = current_instruction.dict()
            elif isinstance(current_instruction, dict):
                current_instruction = dict(current_instruction)
            else:
                # Fallback for unexpected types
                current_instruction = {}
        except Exception as e:
            logger.warning(f"Error converting current_instruction to dict: {e}")
            current_instruction = {}
            
        try:
            if hasattr(previous_instruction, 'model_dump'):
                previous_instruction = previous_instruction.model_dump()
            elif hasattr(previous_instruction, 'dict'):
                previous_instruction = previous_instruction.dict()
            elif isinstance(previous_instruction, dict):
                previous_instruction = dict(previous_instruction)
            else:
                # Fallback for unexpected types
                previous_instruction = {}
        except Exception as e:
            logger.warning(f"Error converting previous_instruction to dict: {e}")
            previous_instruction = {}

        # Only identify changes if previous_instruction has process_steps data
        changes = {}
        if previous_instruction and previous_instruction.get('process_steps'):
            changes = self._identify_changes(previous_instruction, current_instruction)
            
        # Build conversation history context using reusable helper
        if full_context:
            conversation_context_raw = self._build_conversation_context_string(full_context, max_messages=10)
            if conversation_context_raw:
                # Format for markdown-style prompt with proper headers
                lines = conversation_context_raw.split(chr(10))
                formatted_lines = []
                for line in lines:
                    if line.startswith("Previous Conversation History:"):
                        formatted_lines.append(f"**Previous Conversation History:**")
                    elif line.startswith("Accumulated Context from Previous Interactions:"):
                        formatted_lines.append(f"**Accumulated Context from Previous Interactions:**")
                    elif line.startswith("Previous Clarifications:"):
                        formatted_lines.append(f"**Previous Clarifications:**")
                    else:
                        formatted_lines.append(line)
                conversation_context = "\n".join(formatted_lines)
            else:
                conversation_context = ""
        else:
            # Fallback to instance conversation history if full_context not provided
            conversation_context = f"""
**Conversation context:**
{self.conversation_summary if self.conversation_summary else "No previous summary"}

**Recent conversation history:**
{json.dumps(self.conversation_history[-5:], indent=2) if self.conversation_history else "No conversation history"}
"""
        
        # Create comprehensive prompt with limited conversation history
        SYSTEM_PROMPT = f"""You are a helpful and precise Configuration Assistant for creating agent instructions.

**Current Agent Context (Previous State):**
{json.dumps(previous_instruction, indent=2)}

**User's latest request:** {user_input}

**Updated Agent Context (New State):**
{json.dumps(current_instruction, indent=2)}

**Changes:**
{chr(10).join([f"• {change}" for change in changes])}

**Recommended Tools:**
{json.dumps(recommended_tools, indent=2)}

{conversation_context}

{self.reply_parser.get_format_instructions()}

You must return a JSON response with these exact fields:
- acknowledge_intent: Acknowledge what the user requested in a concise, empathetic way
- explain_changes: Explain what you've added to the instructions based on the previous and current configuration
- follow_up_questions: List of 1-2 clarifying questions if needed (can be empty array)
- suggest_next_steps: Suggest next steps or improvements

Return ONLY the JSON response, no additional formatting, markdown, or explanatory text.
"""

        try:
            # print(f"Starting streaming response for user input: {user_input}...")
            # print(f"Prompt length: {len(SYSTEM_PROMPT)} characters")
            
            # Stream LLM response as message_delta events
            accumulated_content = ""
            async for event in self._stream_llm_response([HumanMessage(content=SYSTEM_PROMPT)], "Assistant Response"):
                accumulated_content = event.get("data", {}).get("accumulated", "")
                yield event
            
            # After streaming is complete, parse and format the response
            if accumulated_content:
                try:
                    # Extract JSON from ```json code blocks
                    content = self._extract_json_from_content(accumulated_content)
                    result = json.loads(content)
                    formatted_assistant_response = self._format_assistant_response(result)
                    
                    # Yield final formatted response as a complete message (for backward compatibility)
                    yield {
                        "type": "assistant_response",
                        "timestamp": time.time(),
                        "data": {
                            "message": formatted_assistant_response,
                            "instruction": current_instruction,
                            "recommended_tools": recommended_tools
                        }
                    }
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(f"⚠️ Error parsing assistant response: {e}")
                    # If parsing fails, the streamed content is already sent, so just log the error
                    yield {
                        "type": "error",
                        "timestamp": time.time(),
                        "data": {
                            "error": "ParseError",
                            "message": f"Failed to parse response: {str(e)}"
                        }
                    }

        except Exception as e:
            logger.warning(f"⚠️ Error in get_assistant_response_stream: {e}")
            # Yield error message
            yield {
                "type": "error",
                "timestamp": time.time(),
                "data": {
                    "error": "StreamError",
                    "message": f"Failed to generate response: {str(e)}"
                }
            }

    def _extract_structured_reply(self, content: str, user_input: str, current_instruction: Dict, previous_instruction: Dict) -> Dict:
        """Extract structured information from natural language response"""
        try:
            # Try to use LLM to extract structured information from the response
            extraction_prompt = f"""
            Extract structured information from this assistant response and format it as JSON:
            
            Response: {content}
            
            Extract and format as JSON with these exact fields:
            - acknowledge_intent: What the assistant acknowledged about the user's request
            - explain_changes: What changes were explained
            - follow_up_questions: Any questions asked (as array, can be empty)
            - suggest_next_steps: Any next steps suggested
            
            Return ONLY valid JSON, no additional text.
            """
            
            with LLMTimingContext("Extract Requirements", self):
                extraction_response = self.llm.invoke([HumanMessage(content=extraction_prompt)])
            extraction_content = extraction_response.content.strip()
            
            # Clean the response
            extraction_content = self._extract_json_from_content(extraction_content)
            
            # Parse the extracted JSON
            extracted_data = json.loads(extraction_content)
            return extracted_data
            
        except Exception as e:
            logger.warning(f"⚠️ Structured extraction failed: {e}")
    
    def _is_question_or_doubt(self, user_input: str, agent_summary_context: Optional[Dict] = None) -> Tuple[bool, str]:
        """Rule-based detection for questions and doubts.
        
        Args:
            user_input: The user's input message
            agent_summary_context: Optional context about existing agent configuration
            
        Returns:
            Tuple of (is_question, suggested_type) where:
            - is_question: True if input appears to be a question or doubt
            - suggested_type: "general_chat" or "clarification" based on context
        """
        if not user_input or not isinstance(user_input, str):
            return (False, "general_chat")
        
        text = user_input.strip().lower()
        
        # Check for question mark
        has_question_mark = "?" in user_input
        
        # Question words and patterns
        question_words = [
            "what", "how", "when", "why", "where", "who", "which", "whose",
            "can you", "could you", "would you", "should i", "should we",
            "may i", "might i", "will you", "do you", "does it", "did you",
            "is it", "are you", "was it", "were you", "have you", "has it",
            "will it", "would it", "could it", "should it", "can it",
            "what if", "how about", "what about", "tell me", "explain",
            "help me", "i need help", "i don't understand", "i'm confused",
            "not sure", "unclear", "clarify", "clarification", "doubt",
            "question", "questions", "wondering", "wonder if", "curious"
        ]
        
        # Doubt and clarification indicators
        doubt_indicators = [
            "clarify", "clarification", "explain", "not sure", "confused",
            "unclear", "help me understand", "i don't understand",
            "i'm confused", "what does", "what is", "what are",
            "how does", "how do", "how can", "how will", "how should",
            "why does", "why do", "why is", "why are", "why will",
            "can you explain", "could you explain", "would you explain",
            "can you clarify", "could you clarify", "would you clarify",
            "help me", "i need help", "i need clarification",
            "what do you mean", "what does this mean", "what is this",
            "i'm not sure", "i am not sure", "not certain", "uncertain"
        ]
        
        # Check if text starts with question words
        starts_with_question = any(text.startswith(qw) for qw in question_words)
        
        # Check if text contains question words (not just at start)
        contains_question_word = any(f" {qw} " in f" {text} " for qw in question_words)
        
        # Check for doubt indicators
        contains_doubt = any(indicator in text for indicator in doubt_indicators)
        
        # Check for interrogative patterns (question structure)
        # Patterns like "is/are/was/were + subject", "do/does/did + subject", etc.
        interrogative_patterns = [
            r"\bis\s+\w+", r"\bare\s+\w+", r"\bwas\s+\w+", r"\bwere\s+\w+",
            r"\bdo\s+\w+", r"\bdoes\s+\w+", r"\bdid\s+\w+",
            r"\bhave\s+\w+", r"\bhas\s+\w+", r"\bhad\s+\w+",
            r"\bwill\s+\w+", r"\bwould\s+\w+", r"\bcould\s+\w+", r"\bshould\s+\w+",
            r"\bcan\s+\w+", r"\bmay\s+\w+", r"\bmight\s+\w+"
        ]
        has_interrogative_pattern = any(re.search(pattern, text) for pattern in interrogative_patterns)
        
        # Determine if it's a question
        is_question = (
            has_question_mark or
            starts_with_question or
            (contains_question_word and (has_question_mark or has_interrogative_pattern)) or
            contains_doubt
        )
        
        # Determine suggested type
        # If there's existing agent context, it's more likely a clarification
        # Otherwise, it's general chat
        if agent_summary_context:
            suggested_type = "clarification"
        else:
            suggested_type = "general_chat"
        
        return (is_question, suggested_type)
    
    def _build_conversation_context_string(self, full_context: Optional[Dict] = None, max_messages: int = 10, include_accumulated: bool = True, include_clarifications: bool = True) -> str:
        """Build a formatted conversation context string from full_context for use in LLM prompts.
        
        Args:
            full_context: Full context dictionary containing conversation history and state information
            max_messages: Maximum number of recent messages to include (default: 10)
            include_accumulated: Whether to include accumulated context (default: True)
            include_clarifications: Whether to include clarification history (default: True)
            
        Returns:
            Formatted string containing conversation history and context, or empty string if no context
        """
        if not full_context:
            return ""
        
        context_parts = []
        
        # Add conversation history
        conversation_history = full_context.get("conversation_history", [])
        if conversation_history:
            history_text = "\n".join([
                f"{msg['role'].title()}: {msg['content']}" 
                for msg in conversation_history[-max_messages:]  # Last N messages
            ])
            context_parts.append(f"Previous Conversation History:\n{history_text}")
        
        # Add accumulated context if requested
        if include_accumulated:
            accumulated = full_context.get("accumulated_context", "")
            if accumulated:
                context_parts.append(f"Accumulated Context from Previous Interactions:\n{accumulated}")
        
        # Add clarification history if requested
        if include_clarifications:
            clarification_history = full_context.get("clarification_history", [])
            if clarification_history:
                clarifications = "\n".join([
                    f"Q: {item.get('question', '')}\nA: {item.get('answer', '')}"
                    for item in clarification_history[-5:]  # Last 5 clarifications
                ])
                context_parts.append(f"Previous Clarifications:\n{clarifications}")
        
        return "\n\n".join(context_parts) if context_parts else ""
    
    async def _analyze_user_input(self, user_input: str, state: Any = None, agent_summary_context: Dict = None, streaming_events: Optional[List] = None, extracted_requirements: Optional[Dict] = None, force_message_type: Optional[str] = None, full_context: Optional[Dict] = None) -> Dict:
        """Internal LLM-based combined analysis (used by the graph node)
        
        Args:
            force_message_type: If set to "agent_creation", override message_analysis.type to "agent_creation" after LLM call
            full_context: Full context dictionary containing conversation history and state information
        """
        
        # Build conversation history context using reusable helper
        conversation_context = self._build_conversation_context_string(full_context, max_messages=10)
        if conversation_context:
            conversation_context = f"\n            {conversation_context.replace(chr(10), chr(10) + '            ')}\n            "
        
        # Build extracted requirements context if provided
        extracted_reqs_context = ""
        if extracted_requirements:
            role = extracted_requirements.get('role', '')
            responsibility = extracted_requirements.get('responsibility', '')
            process_title = extracted_requirements.get('process_title', 'Main Process')
            process_steps = extracted_requirements.get('process_steps', [])
            tool_usage_title = extracted_requirements.get('tool_usage_title')
            tool_steps = extracted_requirements.get('tool_steps', [])
            extracted_reqs_context = f"""
            Pre-extracted Requirements (use these directly):
            Role: {role}
            Responsibility: {responsibility}
            Process Title: {process_title}
            Process Steps: {json.dumps(process_steps, indent=2)}
            Tool Usage Title: {tool_usage_title if tool_usage_title else 'None'}
            Tool Steps: {json.dumps(tool_steps, indent=2)}
            """
        
        combined_prompt = f""" 
        You are an expert AI assistant for agent configuration. Analyze this user message and provide a comprehensive response.
        
        {conversation_context}
        
        User message: "{user_input}"
        
        {extracted_reqs_context}
        
        Provide a complete analysis in this JSON format:
        {{
            "message_analysis": {{
                "type": "agent_creation|modification|clarification|general_chat",
                "modification_type": "add_step|remove_step|change_role|add_tool|remove_tool|add_requirement|null",
                "confidence": 0.8
            }},
            "extracted_requirements": {{
                "role": "extracted role",
                "responsibility": "extracted responsibility",
                "process_title": "Main Process",
                "process_steps": [
                    "Detailed implementation step 1",
                    "Detailed implementation step 2",
                    "Detailed implementation step 3"
                ],
                "tool_usage_title": "Tool-Specific Operations",
                "tool_steps": [
                    "Tool-specific implementation step 1",
                    "Tool-specific implementation step 2"
                ]
            }},
            "tool_categories": {{
                "primary_categories": ["category1", "category2"],
                "specific_keywords": ["keyword1", "keyword2"],
                "related_keywords": ["related1", "related2"],
                "tool_types": ["tool_type1", "tool_type2"],
                "app_categories": ["app_category1", "app_category2"],
                "required_tools": [
                    {{
                        "tool_name": "specific tool name",
                        "tool_type": "tool category",
                        "priority": "high|medium|low",
                        "reason": "why this tool is needed"
                    }}
                ],
                "search_queries": [
                    {{"query": "email management", "weight": 0.9}},
                    {{"query": "gmail integration", "weight": 0.8}},
                    {{"query": "communication tools", "weight": 0.7}}
                ],
                "integration_needs": ["integration1", "integration2"],
                "confidence_score": 0.85,
                "analysis_summary": "Brief comprehensive summary of the agent's role, purpose, and how it should operate"
            }},
            "clarification_questions": ["question1", "question2"],
            "needs_clarification": false
        }}
        
        Instructions:
        1. For message_analysis: Classify the user's intent into one of these types:
           
           - "general_chat": User asking general questions, greetings, or having casual conversation
             Examples: "Hello", "Hi", "What can you help me with?", "How does this work?", 
             "What are your capabilities?", "Can you explain what you do?", "What should I do?",
             "Help me understand...", "I don't know what to do", "Thanks", "Good morning",
             "How are you?", "What's this about?", "Tell me more", "I'm confused",
             "What does this mean?", "How do I use this?", "Can you clarify?",
             "I'm not sure what to do", "What is this agent for?", "How does the agent work?",
             "Why did you do that?", "What happened?", "Can you explain?"
           
           - "agent_creation": User describing what they want an agent to do (specific task/role)
             Examples: "Create an agent that sends emails", 
             "I need an agent to handle customer support",
             "Build an agent that monitors data", "Make an agent for processing invoices"
           
           - "modification": User wants to change existing agent configuration
             Examples: "Add a step to validate data", "Change the email template",
             "Update the agent to include...", "Remove the notification step"
             
             CRITICAL: "modification" requires EXPLICIT change requests with action verbs like:
             "add", "remove", "change", "update", "modify", "delete", "replace", "edit".
             DO NOT classify as "modification" if the user is:
             - Asking a question (even if it mentions existing configuration)
             - Seeking clarification or explanation
             - Expressing doubt or confusion
             - Using question words (what, how, when, why, can you, could you, etc.)
           
           - "clarification": User providing answers or clarifying previous questions, OR asking
             questions about existing agent configuration when agent_summary_context is provided
             Examples: "Yes, send it daily at 9 AM", "Use Gmail for emails",
             "The report should include sales data", "What does the current agent do?",
             "How does the agent handle errors?", "Can you explain the current workflow?",
             "I'm confused about what the agent does", "What tools does the agent use?"
           
           CRITICAL PRIORITY RULES FOR QUESTION DETECTION:
           
           1. QUESTION MARK PRIORITY: If the message contains a question mark (?), it is almost
              certainly a question. Classify as "general_chat" or "clarification" (not "modification").
           
           2. QUESTION WORD PRIORITY: If the message starts with or contains question words like:
              "what", "how", "when", "why", "where", "who", "which", "can you", "could you",
              "would you", "should I", "do you", "does it", "is it", "are you", "tell me",
              "explain", "help me", "clarify", etc., classify as "general_chat" or "clarification".
           
           3. DOUBT INDICATORS: If the message contains doubt indicators like:
              "not sure", "confused", "unclear", "help me understand", "I don't understand",
              "I'm confused", "clarify", "clarification", "explain", "what does", "what is",
              classify as "general_chat" or "clarification".
           
           4. MODIFICATION vs QUESTION: A message is ONLY "modification" if it contains:
              - Explicit action verbs: "add", "remove", "change", "update", "modify", "delete"
              - AND no question marks or question words
              - AND clear intent to change configuration (not just asking about it)
           
           5. CONTEXT-BASED CLASSIFICATION:
              - If agent_summary_context exists AND user asks a question → "clarification"
              - If no agent context AND user asks a question → "general_chat"
              - If user asks "what does the agent do?" or similar → "clarification" (if context exists)
           
           IMPORTANT: If the user is greeting, asking general questions about capabilities,
           or having casual conversation, ALWAYS classify as "general_chat".
           Only classify as "agent_creation" if they specifically describe wanting to create
           or build an agent for a particular task.
           
           When in doubt between "modification" and a question type, ALWAYS choose the question type
           ("general_chat" or "clarification"). It's better to treat a question as a question
           than to misclassify it as a modification request.
        2. For extracted_requirements: Extract agent configuration details from the message
           - role: Extract from the message or use agent_summary_context role/purpose if available
           - responsibility: Extract from the message or use agent_summary_context responsibility if available
           - process_steps: Generate implementation-oriented, actionable steps. If agent_summary_context is provided:
             * Use core_responsibilities and expected_workflow to create detailed, concrete process steps
             * Combine and transform these into specific actionable implementation steps
             * Each step should be clear, implementation-focused, and describe what the agent will actually do
             * Include at least 3-5 detailed steps based on the workflow and responsibilities
             Example: Instead of "Handle requests", use "Receive and validate incoming customer support requests"
        3. For tool_categories: Identify categories that would help find relevant tools
        4. For clarification_questions: Generate helpful clarification questions if needed
        5. Set needs_clarification to true only if message_analysis type is "agent_creation" AND critical information is missing
        
        Return ONLY valid JSON, no additional formatting.
        """
        
        # Pre-classification check: Use rule-based detection to identify questions
        is_question, suggested_type = self._is_question_or_doubt(user_input, agent_summary_context)
        
        # If it's clearly a question, add a hint to the prompt
        if is_question:
            question_hint = f"""
            
            IMPORTANT HINT: The user input appears to be a QUESTION or DOUBT based on rule-based detection.
            It contains question marks, question words, or doubt indicators.
            You should classify this as "{suggested_type}" (not "modification").
            The user is asking for clarification, explanation, or help - NOT requesting a configuration change.
            """
            combined_prompt = combined_prompt + question_hint
        
        # Log the prompt
        self._log_llm_prompt([HumanMessage(content=combined_prompt)], "Combined Analysis and Extraction")

        try:
            content = ""
            accumulated = ""
            with LLMTimingContext("Combined Analysis and Extraction", self):
                response = await self.llm.ainvoke([HumanMessage(content=combined_prompt)])
                if hasattr(response, 'content') and response.content:
                    content = response.content
                    accumulated = content
                    
                    # Stream as message_delta if streaming_events list is provided
                    if streaming_events is not None:
                        streaming_events.append({
                            "type": "message_delta",
                            "timestamp": time.time(),
                            "data": {
                                "delta": accumulated,
                                "accumulated": accumulated,
                                "role": "assistant",
                                "content": accumulated
                            }
                        })
                content = accumulated.strip()
            
            # Log the response
            self._log_llm_response(content, "Combined Analysis and Extraction")
            
            content = self._extract_json_from_content(content)
            
            result = json.loads(content)
            
            # Post-processing override: Check if LLM misclassified a question as "modification"
            # Use rule-based detection as fallback to correct misclassifications
            llm_classified_type = result.get("message_analysis", {}).get("type", "general_chat")
            
            # If LLM classified as "modification" but rule-based detector says it's a question, override it
            if llm_classified_type == "modification":
                is_question_override, suggested_type_override = self._is_question_or_doubt(user_input, agent_summary_context)
                if is_question_override:
                    logger.info(f"⚠️ Overriding LLM classification: 'modification' → '{suggested_type_override}' (rule-based detection identified question)")
                    if "message_analysis" not in result:
                        result["message_analysis"] = {}
                    result["message_analysis"]["type"] = suggested_type_override
                    # Update confidence to reflect the override
                    result["message_analysis"]["confidence"] = 0.9
            
            # Override message_analysis.type if force_message_type is set
            if force_message_type == "agent_creation":
                if "message_analysis" not in result:
                    result["message_analysis"] = {}
                result["message_analysis"]["type"] = "agent_creation"
            
            # Fix needs_clarification logic: only true when message_analysis type is agent_creation
            message_type = result.get("message_analysis", {}).get("type", "general_chat")
            if message_type == "agent_creation":
                # Keep the LLM's decision for agent_creation
                pass
            else:
                # For all other types, always set needs_clarification to false
                result["needs_clarification"] = False
            
            # Use provided extracted_requirements if available, otherwise use LLM-generated ones
            if extracted_requirements:
                result["extracted_requirements"] = extracted_requirements
            
            # Check if message_analysis type is "clarification" and generate response
            if message_type == "clarification":
                clarification_questions = result.get("clarification_questions", [])
                extracted_reqs = result.get("extracted_requirements", {})
                
                # Generate clarification response using LLM based on user input and extracted requirements
                if extracted_reqs:
                    clarification_response = await self._generate_clarification_response(
                        user_input=user_input,
                        extracted_requirements=extracted_reqs,
                        clarification_questions=clarification_questions,
                        streaming_events=streaming_events
                    )
                    result["clarification_response"] = clarification_response
            
            # Save current_instruction from extracted_requirements
            extracted_reqs = result.get("extracted_requirements", {})
            if extracted_reqs:
                # Store previous instruction before updating
                if isinstance(self.current_instruction, dict):
                    self.previous_instruction = self.current_instruction.copy()
                else:
                    try:
                        self.previous_instruction = self.current_instruction.dict()
                    except:
                        self.previous_instruction = {}
                
                # Update current_instruction with extracted requirements
                new_instruction = AgentInstruction(
                    role=extracted_reqs.get("role", ""),
                    responsibility=extracted_reqs.get("responsibility", ""),
                    key_tool=extracted_reqs.get("key_tool", None),
                    process_title=extracted_reqs.get("process_title", "Main Process"),
                    process_steps=extracted_reqs.get("process_steps", []),
                    tool_usage_title=extracted_reqs.get("tool_usage_title", None),
                    tool_steps=extracted_reqs.get("tool_steps", [])
                )
                
                # Update instance variable
                self.current_instruction = new_instruction
                
                # Update state if provided
                if state is not None:
                    state.current_instruction = new_instruction
                    print(f"✅ Updated state.current_instruction with {len(extracted_reqs.get('process_steps', []))} process steps", file=sys.stderr)
                
                print(f"✅ Updated current_instruction with {len(extracted_reqs.get('process_steps', []))} process steps", file=sys.stderr)
            
            return result
            
        except Exception as e:
            print(f"Combined analysis error: {e}")
            return {}
            
    


    @time_method("Get Instruction", log_args=True)
    async def get_instruction(self, current_instruction: Dict[str, Any], recommended_tools: List[Dict], streaming_events: Optional[List] = None, message_type: Optional[str] = None, extracted_requirements: Optional[Dict] = None, modification_type: Optional[str] = None, user_input: Optional[str] = None, full_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Combine current instruction with suggested tools to create comprehensive agent configuration.
        
        If message_type is 'modification', only update the specific part indicated by modification_type and extracted_requirements.
        
        Args:
            full_context: Full context dictionary containing conversation history and state information
        """
        
        # Helper function to convert instruction to dict
        def get_instruction_dict(instruction):
            """Convert instruction to dict format, handling both dict and object types"""
            if isinstance(instruction, dict):
                return instruction
            elif hasattr(instruction, 'model_dump'):
                return instruction.model_dump()
            elif hasattr(instruction, 'dict'):
                return instruction.dict()
            else:
                return {}
        
        current_instruction_dict = get_instruction_dict(current_instruction)
        
        # Build modification context if this is a modification request
        modification_context = ""
        fields_to_modify = []
        
        if message_type == "modification" and extracted_requirements:
            # Identify which specific fields need modification
            if modification_type in ["add_step", "remove_step"] or extracted_requirements.get("process_steps"):
                fields_to_modify.append("process_steps")
            if extracted_requirements.get("tool_steps"):
                fields_to_modify.append("tool_steps")
            if extracted_requirements.get("role"):
                fields_to_modify.append("role")
            if extracted_requirements.get("responsibility"):
                fields_to_modify.append("responsibility")
            
            # Build modification context
            modification_parts = []
            if "process_steps" in fields_to_modify:
                new_steps = extracted_requirements.get("process_steps", [])
                modification_parts.append(f"PROCESS_STEPS to update: {json.dumps(new_steps, indent=2)}")
            if "tool_steps" in fields_to_modify:
                new_tool_steps = extracted_requirements.get("tool_steps", [])
                modification_parts.append(f"TOOL_STEPS to update: {json.dumps(new_tool_steps, indent=2)}")
            if "role" in fields_to_modify:
                modification_parts.append(f"ROLE to update: {extracted_requirements.get('role', '')}")
            if "responsibility" in fields_to_modify:
                modification_parts.append(f"RESPONSIBILITY to update: {extracted_requirements.get('responsibility', '')}")
            
            modification_context = f"""
        
        ⚠️ MODIFICATION REQUEST DETECTED ⚠️
        Modification Type: {modification_type or "general_modification"}
        User Request: "{user_input}"
        
        IMPORTANT: You must update ONLY the following fields and keep ALL other fields EXACTLY as they are:
        {chr(10).join(modification_parts)}
        
        Current instruction fields that MUST remain unchanged:
        - Role: {current_instruction_dict.get('role', 'N/A')} ({"UPDATE" if "role" in fields_to_modify else "KEEP AS IS"})
        - Responsibility: {current_instruction_dict.get('responsibility', 'N/A')} ({"UPDATE" if "responsibility" in fields_to_modify else "KEEP AS IS"})
        - Process Steps: {len(current_instruction_dict.get('process_steps', []))} steps ({"UPDATE" if "process_steps" in fields_to_modify else "KEEP AS IS"})
        - Tool Steps: {len(current_instruction_dict.get('tool_steps', []))} steps ({"UPDATE" if "tool_steps" in fields_to_modify else "KEEP AS IS"})
        
        CRITICAL REQUIREMENTS:
        1. Keep ALL unchanged fields EXACTLY as they appear in the current instruction
        2. Update ONLY the fields listed above based on the extracted requirements
        3. Ensure all process_steps and tool_steps remain implementation-focused and actionable
        4. Each step should be clear, specific, and describe what the agent will actually do
        5. Do NOT regenerate or modify fields that are not in the modification list
        """
        
        # Build conversation history context using reusable helper
        conversation_context = self._build_conversation_context_string(full_context, max_messages=10)
        if conversation_context:
            # Format for uppercase headers style prompt
            conversation_context = f"\n        {conversation_context.replace('Previous Conversation History:', 'PREVIOUS CONVERSATION HISTORY:').replace('Accumulated Context from Previous Interactions:', 'ACCUMULATED CONTEXT FROM PREVIOUS INTERACTIONS:').replace('Previous Clarifications:', 'PREVIOUS CLARIFICATIONS:').replace(chr(10), chr(10) + '        ')}\n        "
        
        SYSTEM_PROMPT = f"""You are an AI assistant that creates comprehensive agent instructions by combining role descriptions with relevant tool suggestions.

        {conversation_context}
        CURRENT AGENT INSTRUCTION:
        {self._get_current_state_summary()}
        {modification_context}

        RECOMMENDED TOOLS:
        {json.dumps(recommended_tools, indent=2)}

        Generate a TECHNICAL response that:
        • References the conversation context and previous requests
        • Explains WHY certain tools are being selected based on established patterns
        • Mentions specific core tools by name with technical reasoning
        • Acknowledges iterative refinements from previous conversations
        • Shows awareness of established integrations and assistant type
        • Uses technical language appropriate for configuration management
        • Provides implementation details when relevant
        • Naturally integrates specific tool IDs into process steps for clear implementation guidance
        {"• Updates ONLY the specified fields for modification requests, keeping all other fields unchanged" if message_type == "modification" else ""}

        For tool_steps array, also integrate tool IDs naturally:
        "Tool-specific step description using tool_id"

        Examples of process steps with integrated tool IDs:
        - "Receive and parse incoming messages using slack_get_messages and slack_get_channel_info"
        - "Create calendar event with meeting details using google_calendar_create_event"
        - "Send confirmation email to participants using gmail_send_email"
        - "Update project status in database using airtable_create_record and airtable_update_record"
        - "Schedule follow-up reminder using google_calendar_create_event and slack_send_message"

        Examples of tool steps with integrated tool IDs:
        - "Authenticate with Slack API using slack_oauth"
        - "Configure webhook endpoints using slack_create_webhook"
        - "Use Send email to send meeting confirmations"
        - "Use Create Google Calendar Event for Google Calendar users"

        Examples of technical responses:
        - "Based on your previous scheduling assistant setup, I'm extending the Google Calendar integration with additional tools..."
        - "Analyzing your conversation pattern shows multi-integration requirements. Adding Gmail tools to complement the existing Slack integration..."
        - "Your iterative refinement pattern indicates you need granular control. I'll add specific tools: post_message, list_channels..."

        Be technical, contextual, and reference previous conversation elements.

        {self.parser.get_format_instructions()}

        """
        # Log the prompt
        self._log_llm_prompt([HumanMessage(content=SYSTEM_PROMPT)], "Get Enhanced Instruction")

        try:
            content = ""
            accumulated = ""
            with LLMTimingContext("Get Enhanced Instruction", self):
                response = await self.llm.ainvoke([HumanMessage(content=SYSTEM_PROMPT)])
                if hasattr(response, 'content') and response.content:
                    content = response.content
                    accumulated = content
                    
                    # Stream as message_delta if streaming_events list is provided
                    if streaming_events is not None:
                        streaming_events.append({
                            "type": "message_delta",
                            "timestamp": time.time(),
                            "data": {
                                "delta": accumulated,
                                "accumulated": accumulated,
                                "role": "assistant",
                                "content": accumulated
                            }
                        })
                content = accumulated.strip()
            
            # Log the response
            self._log_llm_response(content, "Get Enhanced Instruction")
            
            # Clean response content for Gemini
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
            
            # Parse the enhanced instruction
            enhanced_instruction = self.parser.parse(content)
            
            # Update the current instruction with the enhanced version
            self.current_instruction = enhanced_instruction
            
            # # Log the output
            # try:
            #     if hasattr(enhanced_instruction, 'model_dump'):
            #         instruction_dict = enhanced_instruction.model_dump()
            #     elif hasattr(enhanced_instruction, 'dict'):
            #         instruction_dict = enhanced_instruction.dict()
            #     elif isinstance(enhanced_instruction, dict):
            #         instruction_dict = enhanced_instruction
            #     else:
            #         instruction_dict = str(enhanced_instruction)
                
            #     logger.info(f"📋 get_instruction output: {json.dumps(instruction_dict, indent=2, default=str)}")
            # except Exception as log_error:
            #     logger.warning(f"⚠️ Failed to log get_instruction output: {log_error}")
            
            return enhanced_instruction
            
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ JSON parsing error in get_instruction: {e}")
            # Fallback: return current instruction as-is
            return current_instruction
        except Exception as e:
            logger.warning(f"⚠️ Error in get_instruction: {e}")
            # Fallback: return current instruction as-is
            return current_instruction
    

    async def _handle_modification(self, user_input: str, modification_type: str, recommended_tools: List[Dict], agent_id: Optional[str] = None, cookies: Optional[Dict[str, str]] = None, streaming_events: Optional[List] = None, full_context: Optional[Dict] = None) -> Dict:
        """Handle modifications to existing instruction
        
        Args:
            full_context: Full context dictionary containing conversation history and state information
        """
                    
        print(f"🔍 **** AGENT MODIFICATION ****")
        # Check if this is a role change request that needs confirmation
        if modification_type == "change_role":
            return self._handle_role_change_confirmation(user_input)
        
        # Check if this is a tool removal request
        if modification_type == "remove_tool":
            return await self._handle_tool_removal_request(user_input, agent_id, recommended_tools, cookies=cookies, streaming_events=streaming_events)
        
        # Capture the previous instruction before any modifications
        previous_instruction = (
            self.current_instruction.model_dump() 
            if hasattr(self.current_instruction, 'model_dump') 
            else self.current_instruction
        )
        # Update the instance variable for tracking
        # self.previous_instruction = self.current_instruction
        current_dict = (
            self.current_instruction.model_dump() 
            if hasattr(self.current_instruction, 'model_dump') 
            else self.current_instruction
        )
        
        modification_prompt = f"""
        Current agent instruction: {json.dumps(current_dict, indent=2)}
        
        User wants to modify: "{user_input}"
        Modification type: {modification_type}
        
        Update the agent instruction based on the user's request.
        
        {self.parser.get_format_instructions()}
        
        Return ONLY the JSON response, no additional formatting or markdown.
        """
        
        try:
            with LLMTimingContext("Handle Clarification", self):
                response = await self.llm.ainvoke([HumanMessage(content=modification_prompt)])
                content = response.content.strip() if hasattr(response, 'content') else ""
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
            
            updated_instruction = self.parser.parse(content)
            self.current_instruction = updated_instruction
            
            # Identify what changed
            changes = self._identify_changes(current_dict, updated_instruction.model_dump())
            
            message = f"""
Updated! Here's what changed:
{chr(10).join([f"• {change}" for change in changes])}
            """
            
# **Current Configuration:**
# **Role:** {self._get_instruction_attr(updated_instruction, 'role', '')}
# **Primary Responsibility:** {self._get_instruction_attr(updated_instruction, 'responsibility', '')}
# {f"**Key Tool:** {self._get_instruction_attr(updated_instruction, 'key_tool', '')}" if self._get_instruction_attr(updated_instruction, 'key_tool', '') else ""}

# **Process Steps:**
# {chr(10).join([f"• {step}" for step in self._get_instruction_attr(updated_instruction, 'process_steps', [])])}

# Anything else you'd like to adjust?

            # Check if this is a tool removal request
            tool_removal_info = self._handle_tool_removal_in_modification(user_input)
            removal_results = None
            instruction_update_results = None
            
            # If tool removal is requested, process it
            if tool_removal_info.get("has_removal_request"):
                if agent_id:
                    # Remove tools via API (pass cookies for authentication and streaming events)
                    removal_results = await self.remove_tools_via_api(agent_id, tool_removal_info, cookies=cookies, streaming_events=streaming_events)
                    
                    # Extract removed tool names for instruction update
                    removed_tool_names = []
                    if removal_results.get("success"):
                        for removed_tool in removal_results.get("removed_tools", []):
                            tool_name = removed_tool.get("tool_name", "")
                            if tool_name and tool_name != "Unknown":
                                removed_tool_names.append(tool_name)
                        
                        # Update instructions if tools were removed and referenced
                        if removed_tool_names:
                            instruction_update_results = await self.update_instructions_after_tool_removal(
                                agent_id, removed_tool_names
                            )
                            
                            # If instructions were updated, use the new instruction
                            if instruction_update_results.get("instructions_updated"):
                                updated_instruction_dict = instruction_update_results.get("updated_instruction")
                                if updated_instruction_dict:
                                    updated_instruction = self.parser.parse(json.dumps(updated_instruction_dict))
                                    self.current_instruction = updated_instruction
                else:
                    # Agent ID not available - inform user
                    removal_results = {
                        "success": False,
                        "message": "Agent ID not available. Cannot remove tools without agent context.",
                        "removed_tools": [],
                        "not_found_tools": tool_removal_info.get("tool_names", []),
                        "errors": ["Agent ID not available"]
                    }
            
            # Get tool suggestions for the modified instruction
            tool_suggestions = await self._get_tool_suggestions_for_instruction(updated_instruction, recommended_tools)
            
            # Build response message with tool removal info if applicable
            response_message = message
            if removal_results:
                if removal_results.get("success"):
                    removed_count = len(removal_results.get("removed_tools", []))
                    not_found_count = len(removal_results.get("not_found_tools", []))
                    
                    removal_summary = f"\n\n**Tool Removal:**\n"
                    if removed_count > 0:
                        removal_summary += f"✅ Successfully removed {removed_count} tool(s)\n"
                    if not_found_count > 0:
                        removal_summary += f"⚠️ {not_found_count} tool(s) not found in the tools list\n"
                    
                    if instruction_update_results and instruction_update_results.get("instructions_updated"):
                        removal_summary += f"📝 Updated instructions to remove references to removed tools\n"
                    
                    response_message += removal_summary
                else:
                    response_message += f"\n\n**Tool Removal:**\n❌ {removal_results.get('message', 'Failed to remove tools')}\n"
            
            return {
                "instruction": updated_instruction.model_dump(),
                "previous_instruction": previous_instruction,
                "needs_clarification": False,
                "questions": [],
                "recommended_tools": tool_suggestions,
                "removed_tools": tool_removal_info.get("tool_names", []),
                "removal_results": removal_results,
                "instruction_update_results": instruction_update_results,
                "response_message": response_message
            }
            
        except Exception as e:
            logger.info(f"Modification parsing error: {e}")
            # Use current_dict and previous_instruction if available, otherwise use fallback
            fallback_instruction = (
                current_dict if 'current_dict' in locals() 
                else (self.current_instruction.model_dump() if hasattr(self.current_instruction, 'model_dump') else self.current_instruction)
            )
            fallback_previous = (
                previous_instruction if 'previous_instruction' in locals()
                else (self.previous_instruction.model_dump() if hasattr(self.previous_instruction, 'model_dump') else self.previous_instruction)
            )
            return {
                "instruction": fallback_instruction,
                "previous_instruction": fallback_previous,
                "needs_clarification": True,
                "questions": []
            }
    
    async def _handle_agent_creation(self, user_input: str, extracted_requirements: Optional[Dict] = None, tool_categories: Optional[Dict] = None, streaming_events: Optional[List] = None, full_context: Optional[Dict] = None) -> Dict:
        """Handle agent creation workflow: get tools, semantic search, and generate instruction.
        
        Args:
            user_input: User's input describing the agent they want to create
            extracted_requirements: Optional pre-extracted requirements (role, responsibility, process_steps, etc.)
            full_context: Full context dictionary containing conversation history and state information
            tool_categories: Optional tool categories for searching tools
            streaming_events: Optional list to append streaming events to
            
        Returns:
            Dict with instruction, recommended_tools, needs_clarification, and questions
        """
        try:
            print(f"🔍 **** AGENT CREATION ****")
            # Step 1: Prepare current_instruction from extracted_requirements or use existing
            if extracted_requirements:
                # Create AgentInstruction from extracted_requirements
                current_instruction = AgentInstruction(
                    role=extracted_requirements.get("role", ""),
                    responsibility=extracted_requirements.get("responsibility", ""),
                    key_tool=extracted_requirements.get("key_tool", None),
                    process_title=extracted_requirements.get("process_title", "Main Process"),
                    process_steps=extracted_requirements.get("process_steps", []),
                    tool_usage_title=extracted_requirements.get("tool_usage_title", None),
                    tool_steps=extracted_requirements.get("tool_steps", [])
                )
                # Update instance variable
                self.current_instruction = current_instruction
            else:
                # Use existing current_instruction or create empty one
                if hasattr(self.current_instruction, 'model_dump'):
                    current_instruction = self.current_instruction
                elif isinstance(self.current_instruction, dict):
                    current_instruction = AgentInstruction(**self.current_instruction)
                else:
                    current_instruction = self.current_instruction
            
            # Convert to dict for method calls
            current_instruction_dict = (
                current_instruction.model_dump() 
                if hasattr(current_instruction, 'model_dump') 
                else current_instruction
            )
            
            # Step 2: Get tools from vector DB (always fetch fresh)
            if not tool_categories:
                # If no tool_categories provided, create empty dict (get_tools_from_categories will handle it)
                tool_categories = {}
            
            tools_result = self.get_tools_from_categories(
                tool_categories, 
                limit=50, 
                user_input=user_input
            )
            
            # Extract tools list from result
            tools_list = tools_result.get("tools_list", [])
            
            # Step 3: Perform semantic search to rank and filter tools
            recommended_tools = []
            if tools_list:
                semantic_result = await self.semantic_search_from_filtered_tools(
                    user_input,
                    current_instruction_dict,
                    tools_list,
                    streaming_events=streaming_events
                )
                recommended_tools = semantic_result.get("recommended_tools", [])
            else:
                logger.warning("No tools found from vector DB search")
            
            # Step 4: Get instruction with recommended tools
            instruction = await self.get_instruction(
                current_instruction_dict,
                recommended_tools,
                streaming_events=streaming_events,
                message_type="agent_creation",
                extracted_requirements=extracted_requirements
            )
            
            # Convert instruction to dict if needed
            if hasattr(instruction, 'model_dump'):
                instruction_dict = instruction.model_dump()
            elif hasattr(instruction, 'dict'):
                instruction_dict = instruction.dict()
            else:
                instruction_dict = instruction
            
            # Update current_instruction
            self.current_instruction = instruction
            
            # Return result (no response_message - let respond node handle it)
            return {
                "instruction": instruction_dict,
                "previous_instruction": {},
                "needs_clarification": False,
                "questions": [],
                "recommended_tools": recommended_tools
            }
            
        except Exception as e:
            logger.error(f"Error in _handle_agent_creation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return fallback result
            return {
                "instruction": self.current_instruction.model_dump() if hasattr(self.current_instruction, 'model_dump') else self.current_instruction,
                "previous_instruction": {},
                "needs_clarification": True,
                "questions": ["An error occurred during agent creation. Could you please try again?"],
                "recommended_tools": []
            }
    
    def _handle_tool_removal_in_modification(self, user_input: str) -> Dict[str, Any]:
        """Detect and extract tool removal requests from user input.
        
        Returns:
            Dict with 'tool_names' (list of tool names) and 'sub_tool_names' (dict mapping tool_name to sub_tool_name)
        """
        try:
            import re
            
            # Enhanced patterns to detect tool removal requests
            removal_patterns = [
                r'remove\s+(?:the\s+)?([\w\s\-_]+?)(?:\s+tool|\s+from|\s+$|,|\.)',
                r'delete\s+(?:the\s+)?([\w\s\-_]+?)(?:\s+tool|\s+from|\s+$|,|\.)',
                r'get\s+rid\s+of\s+(?:the\s+)?([\w\s\-_]+?)(?:\s+tool|\s+from|\s+$|,|\.)',
                r'don\'t\s+need\s+(?:the\s+)?([\w\s\-_]+?)(?:\s+tool|\s+from|\s+$|,|\.)',
                r'uninstall\s+(?:the\s+)?([\w\s\-_]+?)(?:\s+tool|\s+from|\s+$|,|\.)',
                r'disable\s+(?:the\s+)?([\w\s\-_]+?)(?:\s+tool|\s+from|\s+$|,|\.)',
            ]
            
            tool_names = []
            sub_tool_mappings = {}  # {tool_name: sub_tool_name}
            user_input_lower = user_input.lower()
            
            # Extract tool names
            for pattern in removal_patterns:
                matches = re.findall(pattern, user_input_lower, re.IGNORECASE)
                for match in matches:
                    tool_name = match.strip()
                    # Clean up common words
                    tool_name = re.sub(r'\s+(tool|from|the|a|an)\s+', ' ', tool_name, flags=re.IGNORECASE).strip()
                    if tool_name and len(tool_name) > 1:
                        tool_names.append(tool_name)
            
            # Detect sub-tool removal patterns (e.g., "remove send_message from slack")
            sub_tool_patterns = [
                r'remove\s+([\w\s\-_]+?)\s+from\s+([\w\s\-_]+?)(?:\s+tool|\s+$|,|\.)',
                r'delete\s+([\w\s\-_]+?)\s+from\s+([\w\s\-_]+?)(?:\s+tool|\s+$|,|\.)',
            ]
            
            for pattern in sub_tool_patterns:
                matches = re.findall(pattern, user_input_lower, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple) and len(match) == 2:
                        sub_tool_name = match[0].strip()
                        parent_tool_name = match[1].strip()
                        if sub_tool_name and parent_tool_name:
                            sub_tool_mappings[parent_tool_name] = sub_tool_name
            
            # Remove duplicates while preserving order
            seen = set()
            unique_tool_names = []
            for name in tool_names:
                name_lower = name.lower()
                if name_lower not in seen:
                    seen.add(name_lower)
                    unique_tool_names.append(name)
            
            return {
                "tool_names": unique_tool_names,
                "sub_tool_mappings": sub_tool_mappings,
                "has_removal_request": len(unique_tool_names) > 0 or len(sub_tool_mappings) > 0
            }
            
        except Exception as e:
            logger.error(f"Error detecting tool removal: {e}")
            return {
                "tool_names": [],
                "sub_tool_mappings": {},
                "has_removal_request": False
            }
    
    async def _handle_tool_removal_request(self, user_input: str, agent_id: Optional[str], recommended_tools: List[Dict], cookies: Optional[Dict[str, str]] = None, streaming_events: Optional[List] = None) -> Dict:
        """Handle tool removal request when modification_type is 'remove_tool'."""
        try:
            # Detect tool removal from user input
            tool_removal_info = self._handle_tool_removal_in_modification(user_input)
            
            if not tool_removal_info.get("has_removal_request"):
                return {
                    "instruction": self.current_instruction.model_dump() if hasattr(self.current_instruction, 'model_dump') else self.current_instruction,
                    "previous_instruction": self.previous_instruction.model_dump() if hasattr(self.previous_instruction, 'model_dump') else self.previous_instruction,
                    "needs_clarification": True,
                    "questions": ["Which tool(s) would you like to remove? Please specify the tool name(s)."],
                    "recommended_tools": recommended_tools,
                    "removed_tools": [],
                    "response_message": "I didn't detect any specific tools to remove. Could you please specify which tool(s) you'd like to remove?"
                }
            
            if not agent_id:
                return {
                    "instruction": self.current_instruction.model_dump() if hasattr(self.current_instruction, 'model_dump') else self.current_instruction,
                    "previous_instruction": self.previous_instruction.model_dump() if hasattr(self.previous_instruction, 'model_dump') else self.previous_instruction,
                    "needs_clarification": False,
                    "questions": [],
                    "recommended_tools": recommended_tools,
                    "removed_tools": [],
                    "response_message": "Agent ID not available. Cannot remove tools without agent context.",
                    "removal_results": {
                        "success": False,
                        "message": "Agent ID not available"
                    }
                }
            
            # Remove tools via API (pass cookies for authentication and streaming events)
            removal_results = await self.remove_tools_via_api(agent_id, tool_removal_info, cookies=cookies, streaming_events=streaming_events)
            
            # Extract removed tool names for instruction update
            removed_tool_names = []
            instruction_update_results = None
            
            if removal_results.get("success"):
                for removed_tool in removal_results.get("removed_tools", []):
                    tool_name = removed_tool.get("tool_name", "")
                    if tool_name and tool_name != "Unknown":
                        removed_tool_names.append(tool_name)
                
                # Update instructions if tools were removed and referenced
                if removed_tool_names:
                    instruction_update_results = await self.update_instructions_after_tool_removal(
                        agent_id, removed_tool_names
                    )
                    
                    # If instructions were updated, use the new instruction
                    if instruction_update_results.get("instructions_updated"):
                        updated_instruction_dict = instruction_update_results.get("updated_instruction")
                        if updated_instruction_dict:
                            updated_instruction = self.parser.parse(json.dumps(updated_instruction_dict))
                            self.current_instruction = updated_instruction
            
            # Build response message
            removed_count = len(removal_results.get("removed_tools", []))
            not_found_count = len(removal_results.get("not_found_tools", []))
            
            response_message = "**Tool Removal Results:**\n\n"
            if removal_results.get("success"):
                if removed_count > 0:
                    response_message += f"✅ Successfully removed {removed_count} tool(s):\n"
                    for rt in removal_results.get("removed_tools", []):
                        if rt.get("type") == "sub_tool":
                            response_message += f"  - {rt.get('sub_tool_name', 'Unknown')} from {rt.get('tool_name', 'Unknown')}\n"
                        else:
                            response_message += f"  - {rt.get('tool_name', 'Unknown')}\n"
                
                if not_found_count > 0:
                    response_message += f"\n⚠️ {not_found_count} tool(s) not found in the tools list:\n"
                    for nf in removal_results.get("not_found_tools", []):
                        response_message += f"  - {nf}\n"
                
                if instruction_update_results and instruction_update_results.get("instructions_updated"):
                    response_message += f"\n📝 Updated instructions to remove references to removed tools.\n"
            else:
                response_message += f"❌ {removal_results.get('message', 'Failed to remove tools')}\n"
                if removal_results.get("errors"):
                    response_message += "\nErrors:\n"
                    for error in removal_results.get("errors", []):
                        response_message += f"  - {error}\n"
            
            current_instruction = (
                self.current_instruction.model_dump() 
                if hasattr(self.current_instruction, 'model_dump') 
                else self.current_instruction
            )
            
            return {
                "instruction": current_instruction,
                "previous_instruction": self.previous_instruction.model_dump() if hasattr(self.previous_instruction, 'model_dump') else self.previous_instruction,
                "needs_clarification": False,
                "questions": [],
                "recommended_tools": recommended_tools,
                "removed_tools": tool_removal_info.get("tool_names", []),
                "removal_results": removal_results,
                "instruction_update_results": instruction_update_results,
                "response_message": response_message
            }
            
        except Exception as e:
            logger.error(f"Error handling tool removal request: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "instruction": self.current_instruction.model_dump() if hasattr(self.current_instruction, 'model_dump') else self.current_instruction,
                "previous_instruction": self.previous_instruction.model_dump() if hasattr(self.previous_instruction, 'model_dump') else self.previous_instruction,
                "needs_clarification": False,
                "questions": [],
                "recommended_tools": recommended_tools,
                "removed_tools": [],
                "response_message": f"An error occurred while processing tool removal: {str(e)}"
            }
    
    async def remove_tools_via_api(self, agent_id: str, tool_removal_info: Dict[str, Any], cookies: Optional[Dict[str, str]] = None, headers: Optional[Dict[str, str]] = None, streaming_events: Optional[List] = None) -> Dict[str, Any]:
        """Remove tools from agent configuration via API calls.
        
        Args:
            agent_id: Agent UUID
            tool_removal_info: Dict from _handle_tool_removal_in_modification containing tool_names and sub_tool_mappings
            cookies: Optional dict of cookies to include in requests (for authentication).
                    Required for authenticated API calls. Can be extracted from FastAPI Request using:
                    `cookies = dict(request.cookies)` or from state if stored there.
            headers: Optional dict of headers to include in requests (for authentication)
            
        Returns:
            Dict with removal results including removed_tools, not_found_tools, and errors
            
        Note:
            This method makes HTTP requests to internal APIs that require session authentication.
            If cookies are not provided, you will get a 401 Unauthorized error.
            To extract cookies from a FastAPI Request object:
            ```python
            cookies = dict(request.cookies)
            ```
        """
        try:
            from src.backend.environment import Config
            
            tool_names = tool_removal_info.get("tool_names", [])
            sub_tool_mappings = tool_removal_info.get("sub_tool_mappings", {})
            print(f"tool_names: {tool_names}")
            print(f"sub_tool_mappings: {sub_tool_mappings}")
            
            if not tool_names and not sub_tool_mappings:
                return {
                    "success": False,
                    "message": "No tools specified for removal",
                    "removed_tools": [],
                    "not_found_tools": [],
                    "errors": []
                }
            
            # Fetch tools from API (using async httpx)
            base_url = Config.FRONTEND_URL or "http://localhost:3000"
            fetch_url = f"{base_url}/api/v1/builder/{agent_id}/tools/get"
            
            # Prepare headers and cookies for authentication
            request_headers = headers.copy() if headers else {}
            request_headers.setdefault("Content-Type", "application/json")
            request_headers.setdefault("Accept", "application/json")
            
            try:
                async with httpx.AsyncClient(timeout=30.0, cookies=cookies, follow_redirects=True) as client:
                    fetch_response = await client.get(fetch_url, headers=request_headers)
                    fetch_response.raise_for_status()
                    fetch_data = fetch_response.json()
                
                if fetch_data.get("status") != "success":
                    return {
                        "success": False,
                        "message": f"Failed to fetch tools: {fetch_data.get('message', 'Unknown error')}",
                        "removed_tools": [],
                        "not_found_tools": tool_names,
                        "errors": [f"API returned status: {fetch_data.get('status')}"]
                    }
                
                tools = fetch_data.get("data", [])
                if not tools:
                    return {
                        "success": False,
                        "message": "No tools found in agent configuration",
                        "removed_tools": [],
                        "not_found_tools": tool_names,
                        "errors": ["No tools available"]
                    }
                
            except httpx.RequestError as e:
                logger.error(f"Error fetching tools from API: {e}")
                return {
                    "success": False,
                    "message": f"Failed to fetch tools from API: {str(e)}",
                    "removed_tools": [],
                    "not_found_tools": tool_names,
                    "errors": [f"API request failed: {str(e)}"]
                }
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error fetching tools from API: {e}")
                if e.response.status_code == 401:
                    error_msg = (
                        f"Authentication failed (401 Unauthorized). "
                        f"The API requires session cookies for authentication. "
                        f"Please ensure cookies are passed to remove_tools_via_api() method. "
                        f"Error: {str(e)}"
                    )
                    logger.warning(error_msg)
                else:
                    error_msg = f"HTTP error fetching tools: {e.response.status_code} - {str(e)}"
                return {
                    "success": False,
                    "message": error_msg,
                    "removed_tools": [],
                    "not_found_tools": tool_names,
                    "errors": [f"HTTP error: {e.response.status_code}"]
                }
            
            # Now process tool removals using the fetched tools
            removed_tools = []
            not_found_tools = []
            errors = []
            
            # Helper function to match tool names against API response structure
            def match_tool_name(search_name: str, tool: Dict) -> bool:
                """Check if search_name matches tool name, key, or id (case-insensitive, partial match)"""
                tool_name = tool.get("name", "").lower()
                tool_key = tool.get("key", "").lower()
                tool_id = tool.get("id", "").lower()
                search_lower = search_name.lower()
                
                # Exact match against name, key, or id
                if (search_lower == tool_name or 
                    search_lower == tool_key or 
                    search_lower == tool_id):
                    return True
                # Partial match
                if (search_lower in tool_name or tool_name in search_lower or
                    search_lower in tool_key or tool_key in search_lower):
                    return True
                # Check metadata name for connector_library
                if tool.get("tool_type") == "connector_library":
                    metadata_name = tool.get("metadata", {}).get("name", "").lower()
                    if search_lower == metadata_name or search_lower in metadata_name:
                        return True
                return False
            
            # Helper function to match sub-tool names
            def match_sub_tool_name(search_name: str, sub_tool: Dict) -> bool:
                """Check if search_name matches sub-tool key or name"""
                sub_tool_key = sub_tool.get("key", "").lower()
                sub_tool_name = sub_tool.get("name", "").lower()
                search_lower = search_name.lower()
                
                return (search_lower == sub_tool_key or 
                        search_lower == sub_tool_name or
                        search_lower in sub_tool_key or
                        search_lower in sub_tool_name)
            
            # Process tool removals - find tools to delete
            tools_to_delete = []  # List of (tool_id, payload) tuples
            for tool_name in tool_names:
                tool_found = False
                
                for tool in tools:
                    if match_tool_name(tool_name, tool):
                        tool_found = True
                        tool_id = tool.get("id")
                        if tool_id:
                            tools_to_delete.append((tool_id, {"function": ""}))
                            removed_tools.append({
                                "tool_name": tool.get("name", "Unknown"),
                                "tool_id": tool_id,
                                "type": "tool"
                            })
                
                if not tool_found:
                    not_found_tools.append(tool_name)
            
            # Process sub-tool removals
            for parent_tool_name, sub_tool_name in sub_tool_mappings.items():
                parent_tool_found = False
                
                for tool in tools:
                    if match_tool_name(parent_tool_name, tool):
                        parent_tool_found = True
                        tool_id = tool.get("id")
                        
                        # Check if this tool has sub-tools (dronahq_tools, mcp, composio)
                        if tool.get("tool_type") in ["dronahq_tools", "mcp", "composio"]:
                            sub_tools = tool.get("tools", [])
                            if not sub_tools:
                                errors.append(f"Tool '{parent_tool_name}' has no sub-tools")
                                break
                            
                            # Find the sub-tool to remove
                            sub_tool_to_remove = None
                            for sub_tool in sub_tools:
                                if match_sub_tool_name(sub_tool_name, sub_tool):
                                    sub_tool_to_remove = sub_tool
                                    break
                            
                            if sub_tool_to_remove and tool_id:
                                sub_tool_key = sub_tool_to_remove.get("key", "")
                                if sub_tool_key:
                                    tools_to_delete.append((tool_id, {"function": sub_tool_key}))
                                    removed_tools.append({
                                        "tool_name": tool.get("name", "Unknown"),
                                        "tool_id": tool_id,
                                        "sub_tool_name": sub_tool_to_remove.get("name", sub_tool_name),
                                        "sub_tool_key": sub_tool_key,
                                        "type": "sub_tool"
                                    })
                                else:
                                    errors.append(f"Sub-tool '{sub_tool_name}' has no key")
                            else:
                                not_found_tools.append(f"{sub_tool_name} from {parent_tool_name}")
                            break
                        else:
                            errors.append(f"Tool '{parent_tool_name}' does not support sub-tool removal")
                            break
                
                if not parent_tool_found:
                    not_found_tools.append(f"{sub_tool_name} from {parent_tool_name} (parent tool not found)")
            
            # Call delete API for each tool/sub-tool (using async httpx)
            delete_url_template = f"{base_url}/api/v1/builder/{agent_id}/tool/delete/{{tool_id}}"
            failed_deletions = []
            
            async with httpx.AsyncClient(timeout=30.0, cookies=cookies, follow_redirects=True) as client:
                for tool_id, payload in tools_to_delete:
                    try:
                        delete_url = delete_url_template.format(tool_id=tool_id)
                        # httpx.delete() doesn't support body parameters
                        # Use client.request() with method="DELETE" instead
                        delete_response = await client.request(
                            method="DELETE",
                            url=delete_url,
                            json=payload,
                            headers=request_headers
                        )
                        delete_response.raise_for_status()
                        logger.info(f"Successfully deleted tool {tool_id} with payload {payload}")
                        
                        # Emit streaming event for successful deletion
                        if streaming_events is not None:
                            streaming_events.append({
                                "type": "tool_deleted",
                                "timestamp": time.time(),
                                "data": {
                                    "tool_id": tool_id,
                                    "message": f"Tool {tool_id} deleted successfully",
                                    "agent_id": agent_id
                                }
                            })
                    except httpx.RequestError as e:
                        logger.error(f"Error deleting tool {tool_id}: {e}")
                        failed_deletions.append({
                            "tool_id": tool_id,
                            "error": str(e)
                        })
                        errors.append(f"Failed to delete tool {tool_id}: {str(e)}")
                    except httpx.HTTPStatusError as e:
                        logger.error(f"HTTP error deleting tool {tool_id}: {e}")
                        failed_deletions.append({
                            "tool_id": tool_id,
                            "error": f"HTTP {e.response.status_code}: {str(e)}"
                        })
                        errors.append(f"Failed to delete tool {tool_id}: HTTP {e.response.status_code}")
            
            # Filter out tools that failed to delete from removed_tools
            if failed_deletions:
                failed_tool_ids = {f["tool_id"] for f in failed_deletions}
                removed_tools = [
                    rt for rt in removed_tools 
                    if rt.get("tool_id") not in failed_tool_ids
                ]
            
            # Emit tools_removed event after all deletions are complete (if any tools were successfully removed)
            # This event will be streamed to the frontend to trigger tools list refresh
            if removed_tools and streaming_events is not None:
                streaming_events.append({
                    "type": "tools_removed",
                    "timestamp": time.time(),
                    "data": {
                        "message": "Tools have been removed from the agent configuration",
                        "removed_tools": removed_tools,
                        "agent_id": agent_id,
                        "action": "refresh_tools"
                    }
                })
                logger.info(f"✅ Emitted tools_removed event for {len(removed_tools)} tools after API deletion")
            
            if failed_deletions:
                return {
                    "success": False,
                    "message": f"Some tools could not be deleted. {len(removed_tools)} removed, {len(failed_deletions)} failed.",
                    "removed_tools": removed_tools,
                    "not_found_tools": not_found_tools,
                    "errors": errors
                }
            
            return {
                "success": True,
                "message": f"Successfully removed {len(removed_tools)} tool(s)",
                "removed_tools": removed_tools,
                "not_found_tools": not_found_tools,
                "errors": errors
            }
            
        except Exception as e:
            logger.error(f"Error removing tools via API: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "message": f"Error removing tools: {str(e)}",
                "removed_tools": [],
                "not_found_tools": tool_names,
                "errors": [str(e)]
            }
    
    async def update_instructions_after_tool_removal(self, agent_id: str, removed_tool_names: List[str]) -> Dict[str, Any]:
        """Update instructions if removed tools are referenced in them.
        
        Args:
            agent_id: Agent UUID
            removed_tool_names: List of tool names that were removed
            
        Returns:
            Dict with update results including whether instructions were updated
        """
        try:
            if not removed_tool_names:
                return {
                    "success": True,
                    "instructions_updated": False,
                    "message": "No tools removed, no instruction update needed"
                }
            
            # Get current instruction
            current_instruction_dict = (
                self.current_instruction.model_dump() 
                if hasattr(self.current_instruction, 'model_dump') 
                else self.current_instruction
            )
            
            if not current_instruction_dict:
                return {
                    "success": True,
                    "instructions_updated": False,
                    "message": "No current instruction to update"
                }
            
            # Check if any removed tools are referenced in the instruction
            instruction_text = json.dumps(current_instruction_dict, indent=2)
            removed_tools_lower = [name.lower() for name in removed_tool_names]
            
            # Check various fields for tool references
            process_steps = current_instruction_dict.get("process_steps", [])
            tool_steps = current_instruction_dict.get("tool_steps", [])
            key_tool = current_instruction_dict.get("key_tool", "")
            
            has_reference = False
            referenced_tools = []
            
            # Check process_steps
            for step in process_steps:
                step_lower = str(step).lower()
                for tool_name in removed_tools_lower:
                    if tool_name in step_lower:
                        has_reference = True
                        if tool_name not in referenced_tools:
                            referenced_tools.append(tool_name)
            
            # Check tool_steps
            for step in tool_steps:
                step_lower = str(step).lower()
                for tool_name in removed_tools_lower:
                    if tool_name in step_lower:
                        has_reference = True
                        if tool_name not in referenced_tools:
                            referenced_tools.append(tool_name)
            
            # Check key_tool
            if key_tool:
                key_tool_lower = str(key_tool).lower()
                for tool_name in removed_tools_lower:
                    if tool_name in key_tool_lower:
                        has_reference = True
                        if tool_name not in referenced_tools:
                            referenced_tools.append(tool_name)
            
            if not has_reference:
                return {
                    "success": True,
                    "instructions_updated": False,
                    "message": "Removed tools were not referenced in instructions"
                }
            
            # Update instruction using LLM to remove references to removed tools
            update_prompt = f"""
The following tools have been removed from the agent configuration:
{json.dumps(removed_tool_names, indent=2)}

Current agent instruction:
{json.dumps(current_instruction_dict, indent=2)}

Please update the instruction to remove all references to the removed tools. Specifically:
- Remove any mentions of these tools from process_steps
- Remove any mentions from tool_steps
- If key_tool references a removed tool, set it to an empty string or remove it

Maintain the overall structure and intent of the instruction, just remove references to the removed tools.

{self.parser.get_format_instructions()}

Return ONLY the JSON response, no additional formatting or markdown.
"""
            
            try:
                with LLMTimingContext("Update Instructions After Tool Removal", self):
                    response = await self.llm.ainvoke([HumanMessage(content=update_prompt)])
                    content = response.content.strip() if hasattr(response, 'content') else ""
                
                # Clean response content
                if content.startswith("```json"):
                    content = content.replace("```json", "").replace("```", "").strip()
                elif content.startswith("```"):
                    content = content.replace("```", "").strip()
                
                # Parse the updated instruction
                updated_instruction = self.parser.parse(content)
                self.current_instruction = updated_instruction
                
                # Save updated instruction to database
                formatted_instruction = self._format_enhanced_instruction(updated_instruction.model_dump())
                await self.save_instruction_to_db(agent_id, formatted_instruction)
                
                return {
                    "success": True,
                    "instructions_updated": True,
                    "message": f"Updated instructions to remove references to {len(referenced_tools)} removed tool(s)",
                    "referenced_tools": referenced_tools,
                    "updated_instruction": updated_instruction.model_dump()
                }
                
            except Exception as e:
                logger.error(f"Error updating instructions with LLM: {e}")
                return {
                    "success": False,
                    "instructions_updated": False,
                    "message": f"Failed to update instructions: {str(e)}",
                    "error": str(e)
                }
                
        except Exception as e:
            logger.error(f"Error in update_instructions_after_tool_removal: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "instructions_updated": False,
                "message": f"Error updating instructions: {str(e)}",
                "error": str(e)
            }
    
    async def _get_tool_suggestions_for_instruction(self, instruction, recommended_tools: List[Dict]) -> List[Dict]:
        """Get tool suggestions based on the current instruction"""
        try:
            all_tools = recommended_tools

            # Create a prompt to analyze what tools are needed for this instruction
            analysis_prompt = f"""
            Analyze this agent instruction and recommend the most suitable tools:
            
            Role: {self._get_instruction_attr(instruction, 'role', '')}
            Primary Responsibility: {self._get_instruction_attr(instruction, 'responsibility', '')}
            Key Tool: {self._get_instruction_attr(instruction, 'key_tool', '') if self._get_instruction_attr(instruction, 'key_tool', '') else 'None specified'}
            Process Steps: {', '.join(self._get_instruction_attr(instruction, 'process_steps', []))}
            
            Available tools: 
            {json.dumps(all_tools, indent=2)}
            
            Based on the agent's role and responsibilities, recommend the most relevant tools.
            Focus on tools that directly support the agent's primary functions.
            
            Return ONLY the JSON response.
            """
            
            with LLMTimingContext("Analyze Modification Request", self):
                response = await self.llm.ainvoke([HumanMessage(content=analysis_prompt)])
                content = response.content.strip() if hasattr(response, 'content') else ""
            
            # Clean response content
            content = self._extract_json_from_content(content)
            
            suggestions = json.loads(content)
            
            # Handle both list and dict responses from LLM
            if isinstance(suggestions, list):
                return suggestions
            elif isinstance(suggestions, dict):
                return suggestions.get("recommended_tools", [])
            else:
                return []
            
            # Validate and enhance the suggestions
            # validated_tools = []
            # for tool in suggestions.get("recommended_tools", []):
            #     if isinstance(tool, dict) and "tool_name" in tool:
            #         # Find the full tool data from available tools
            #         full_tool = next((t for t in all_tools if t["tool_name"] == tool["tool_name"]), None)
            #         if full_tool:
            #             enhanced_tool = {
            #                 "tool_name": full_tool["tool_name"],
            #                 "description": full_tool["description"],
            #                 "logo_url": full_tool.get("logo_url", "🔧"),
            #                 "app_name": full_tool.get("app_name", "general"),
            #                 "status": "draft",
            #                 "relevance_reason": tool.get("relevance_reason", "Recommended for this agent")
            #             }
            #             validated_tools.append(enhanced_tool)
            
            # return validated_tools
            
        except Exception as e:
            logger.error(f"Error")
            return []
    
    def _handle_role_change_confirmation(self, user_input: str) -> Dict:
        """Handle role change requests with confirmation"""
        try:
            # Check if this is a confirmation response (yes/no)
            user_input_lower = user_input.lower().strip()
            
            if user_input_lower in ['yes', 'y', 'confirm', 'proceed']:
                # User confirmed, proceed with role change
                # Get the pending role change from conversation context
                pending_change = getattr(self, '_pending_role_change', None)
                if pending_change:
                    # Apply the role change
                    return self._apply_role_change(pending_change)
                else:
                    return {
                        "instruction": self.current_instruction.model_dump(),
                        "needs_clarification": True,
                        "questions": ["What role would you like the agent to have?"]
                    }
            
            elif user_input_lower in ['no', 'n', 'cancel', 'abort']:
                # User declined, cancel role change
                self._pending_role_change = None
                return {
                    "instruction": self.current_instruction.model_dump(),
                    "needs_clarification": False,
                    "questions": []
                }
            
            else:
                # This is the initial role change request, ask for confirmation
                # Store the requested change for later confirmation
                self._pending_role_change = user_input
                
                # Extract the new role from user input
                role_prompt = f"""
                Extract the new role from this user request: "{user_input}"
                
                Current role: {self._get_instruction_attr(self.current_instruction, 'role', '')}
                
                What new role is the user requesting? Provide a clear, concise role description.
                """
                
                try:
                    with LLMTimingContext("Extract New Role", self):
                        response = self.llm.invoke([HumanMessage(content=role_prompt)])
                    new_role = response.content.strip()
                    
                    confirmation_message = f"""
**Role Change Confirmation Required**

**Current Role:** {self._get_instruction_attr(self.current_instruction, 'role', '')}
**Proposed New Role:** {new_role}

⚠️ **Warning:** Changing the role will significantly modify the agent's behavior and may require different tools and capabilities.

**Do you want to proceed with this role change?**
- Type **"yes"** to confirm and change the role
- Type **"no"** to cancel and keep the current role
                    """
                    
                    return {
                        "instruction": self.current_instruction.model_dump(),
                        "needs_clarification": True,
                        "questions": ["Do you want to proceed with this role change? (yes/no)"],
                        "pending_role_change": new_role
                    }
                    
                except Exception as e:
                    logger.error(f"Error")
                    return {
                        "instruction": self.current_instruction.model_dump(),
                        "needs_clarification": True,
                        "questions": ["What specific role would you like the agent to have?"]
                    }
                    
        except Exception as e:
            logger.error(f"Error")
            return {
                "instruction": (
                    self.current_instruction.model_dump() 
                    if hasattr(self.current_instruction, 'model_dump') 
                    else self.current_instruction
                ),
                "needs_clarification": False,
                "questions": []
            }
    
    async def _apply_role_change(self, role_change_request: str) -> Dict:
        """Apply the confirmed role change"""
        try:
            # Capture the previous instruction before any modifications
            previous_instruction = self.current_instruction.model_dump()
            current_dict = self.current_instruction.model_dump()
            
            modification_prompt = f"""
            Current agent instruction: {json.dumps(current_dict, indent=2)}
            
            User wants to change the role: "{role_change_request}"
            
            Update the agent instruction to reflect the new role. This is a significant change that may require:
            - New role
            - Updated responsibility
            - Modified process_steps
            
            {self.parser.get_format_instructions()}
            
            Return ONLY the JSON response, no additional formatting or markdown.
            """
            
            with LLMTimingContext("Handle Instruction Modification", self):
                response = await self.llm.ainvoke([HumanMessage(content=modification_prompt)])
                content = response.content.strip() if hasattr(response, 'content') else ""
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
            
            updated_instruction = self.parser.parse(content)
            self.current_instruction = updated_instruction
            
            # Clear the pending role change
            self._pending_role_change = None
            
            # Identify what changed
            changes = self._identify_changes(current_dict, updated_instruction.model_dump())
            
            message = f"""
✅ **Role Changed Successfully!**

Here's what changed:
{chr(10).join([f"• {change}" for change in changes])}

**New Configuration:**
**Role:** {self._get_instruction_attr(updated_instruction, 'role', '')}
**Primary Responsibility:** {self._get_instruction_attr(updated_instruction, 'responsibility', '')}
{f"**Key Tool:** {self._get_instruction_attr(updated_instruction, 'key_tool', '')}" if self._get_instruction_attr(updated_instruction, 'key_tool', '') else ""}

**Process Steps:**
{chr(10).join([f"• {step}" for step in self._get_instruction_attr(updated_instruction, 'process_steps', [])])}

The agent now has a completely new role and may need different tools. Would you like me to suggest appropriate tools for this new role?
            """
            
            # Check if this is a tool removal request
            removed_tools = self._handle_tool_removal_in_modification(role_change_request)
            
            # Get tool suggestions for the modified instruction
            # For role changes, we start with an empty list and let the system suggest new tools
            tool_suggestions = self._get_tool_suggestions_for_instruction(updated_instruction, [])
            
            return {
                "instruction": updated_instruction.model_dump(),
                "previous_instruction": previous_instruction,
                "needs_clarification": False,
                "questions": [],
                "recommended_tools": tool_suggestions,
                "removed_tools": removed_tools,
                "role_changed": True
            }
            
        except Exception as e:
            logger.error(f"Error")
            # Clear the pending role change on error
            self._pending_role_change = None
            return {
                "instruction": self.current_instruction.model_dump(),
                "needs_clarification": False,
                "questions": []
            }
    

    def _format_enhanced_instruction(self, instruction: Dict) -> str:
        """Format the enhanced instruction that combines role with suggested tools"""
        if not instruction:
            print("🔍 DEBUG: _format_enhanced_instruction - No instruction provided")
            return ""
        
        print(f"🔍 DEBUG: _format_enhanced_instruction - Input instruction type: {type(instruction)}")
        print(f"🔍 DEBUG: _format_enhanced_instruction - Input instruction keys: {list(instruction.keys()) if isinstance(instruction, dict) else 'Not a dict'}")
        
        # Handle both dict and AgentInstruction objects
        role = self._get_instruction_attr(instruction, "role", "")
        responsibility = self._get_instruction_attr(instruction, "responsibility", "")
        # key_tool = self._get_instruction_attr(instruction, "key_tool", "")
        process_title = self._get_instruction_attr(instruction, "process_title", "Main Process")
        process_steps = self._get_instruction_attr(instruction, "process_steps", [])
        tool_usage_title = self._get_instruction_attr(instruction, "tool_usage_title", "")
        tool_steps = self._get_instruction_attr(instruction, "tool_steps", [])
        
        # Build the enhanced instruction text
        instruction_text = "" ### \n\n🎯 **Enhanced Agent Configuration**
        
        # Role and responsibility
        if role:
            instruction_text += f"**Role:** {role}\n"
        if responsibility:
            instruction_text += f"**Primary Responsibility:** {responsibility}\n"
        # if key_tool:
        #     instruction_text += f"**Key Tool:** {key_tool}\n"
        instruction_text += "\n"
        
        # Main process steps
        if process_steps:
            instruction_text += f"**{process_title}:**\n"
            for step in process_steps:
                instruction_text += f"• {step}\n"
            instruction_text += "\n"
        
        # Tool-specific steps (if any)
        if tool_steps:
            title = tool_usage_title if tool_usage_title else "Tool-Specific Operations"
            instruction_text += f"**{title}:**\n"
            for step in tool_steps:
                instruction_text += f"• {step}\n"
            instruction_text += "\n"
        
        # print(f"🔍 DEBUG: _format_enhanced_instruction - Final formatted instruction length: {len(instruction_text)}")
        # print(f"🔍 DEBUG: _format_enhanced_instruction - Final formatted instruction preview: {instruction_text[:300]}...")
        
        return instruction_text
    
    def _format_assistant_response(self, assistant_response: Dict) -> str:
        """Format the assistant reply into a user-friendly message"""
        if not assistant_response:
            return ""
        
        # acknowledge_intent = assistant_response.get("acknowledge_intent", "")
        explain_changes = assistant_response.get("explain_changes", "")
        follow_up_questions = assistant_response.get("follow_up_questions", [])
        suggest_next_steps = assistant_response.get("suggest_next_steps", "")
        
        # Build the assistant reply text
        reply_text = "\n" ## 💬 **Assistant Response**
        
        # if acknowledge_intent:
        #     reply_text += f"**{acknowledge_intent}**\n\n"
        
        if explain_changes:
            reply_text += f"{explain_changes}\n\n"
        
        # if follow_up_questions:
        #     reply_text += "**Questions for you:**\n"
        #     for question in follow_up_questions:
        #         reply_text += f"• {question}\n"
        #     reply_text += "\n"
        
        # if suggest_next_steps:
        #     reply_text += f"**Next steps:** {suggest_next_steps}\n"
        
        return reply_text

    async def _handle_clarification(self, user_input: str, recommended_tools: List[Dict]) -> Dict:
        """Handle clarification responses"""
        # Check if this is a role change confirmation response
        if hasattr(self, '_pending_role_change') and self._pending_role_change:
            return self._handle_role_change_confirmation(user_input)
        
        # Update instruction based on clarification
        return await self._handle_modification(user_input, "clarification", recommended_tools, agent_id=None)
    
    def _handle_completion_request(self) -> Dict:
        """Generate final agent instruction"""
        final_instruction = self._format_final_instruction()
        
        message = f"""
🎉 Here's your complete agent instruction:

{final_instruction}
        """
        
        return {
            "instruction": self.current_instruction.model_dump(),
            "final_instruction": final_instruction,
            "is_complete": True,
            "needs_clarification": False,
            "questions": []
        }
    
    

    def _identify_changes(self, old_dict: Dict, new_dict: Dict) -> List[str]:
        """Identify what changed between versions with detailed information"""
        changes = []
        
        # Ensure both parameters are dictionaries
        if not isinstance(old_dict, dict):
            old_dict = {}
        if not isinstance(new_dict, dict):
            new_dict = {}
        
        for key, new_value in new_dict.items():
            old_value = old_dict.get(key)
            if old_value != new_value:
                if isinstance(new_value, list) and isinstance(old_value, list):
                    if len(new_value) > len(old_value):
                        added_items = [item for item in new_value if item not in old_value]
                        if added_items:
                            changes.append(f"Added to {key.replace('_', ' ')}: {', '.join(added_items[:3])}{'...' if len(added_items) > 3 else ''}")
                        else:
                            changes.append(f"Added items to {key.replace('_', ' ')}")
                    elif len(new_value) < len(old_value):
                        removed_items = [item for item in old_value if item not in new_value]
                        if removed_items:
                            changes.append(f"Removed from {key.replace('_', ' ')}: {', '.join(removed_items[:3])}{'...' if len(removed_items) > 3 else ''}")
                        else:
                            changes.append(f"Removed items from {key.replace('_', ' ')}")
                    else:
                        changes.append(f"Modified {key.replace('_', ' ')}")
                else:
                    # For non-list values, show the actual change
                    if key in ['role', 'responsibility', 'key_tool']:
                        changes.append(f"Updated {key.replace('_', ' ')}: '{old_value}' → '{new_value}'")
                    else:
                        changes.append(f"Updated {key.replace('_', ' ')}")
        
        return changes if changes else ["Made improvements to the configuration"]
    
    def _get_instruction_attr(self, instruction, attr_name, default=None):
        """Safely get attribute from instruction (handles both Pydantic models and dictionaries)"""
        if hasattr(instruction, attr_name):
            return getattr(instruction, attr_name)
        elif isinstance(instruction, dict):
            return instruction.get(attr_name, default)
        else:
            return default

    def _get_current_state_summary(self) -> str:
        """Get a summary of current agent state"""
        # Handle both Pydantic models and dictionaries for cached instances
        role = self._get_instruction_attr(self.current_instruction, 'role', '')
        responsibility = self._get_instruction_attr(self.current_instruction, 'responsibility', '')
        process_title = self._get_instruction_attr(self.current_instruction, 'process_title', '')
        steps = self._get_instruction_attr(self.current_instruction, 'process_steps', [])
        tool_usage_title = self._get_instruction_attr(self.current_instruction, 'tool_usage_title', '')
        tool_steps = self._get_instruction_attr(self.current_instruction, 'tool_steps', [])

        return (
            f"Role: {role}, Responsibility: {responsibility}, Process Title: {process_title}, "
            f"Process Steps: {steps}, Tool Usage Title: {tool_usage_title}, Tool Steps: {tool_steps}"
        )
    
    def _estimate_conversation_tokens(self) -> int:
        """Estimate total tokens in conversation history."""
        total = 0
        for msg in self.conversation_history:
            # Handle both dict format and LangChain message objects
            if isinstance(msg, dict):
                content = msg.get("content", "")
            elif hasattr(msg, 'content'):
                content = msg.content
            else:
                content = str(msg)
            total += estimate_tokens(str(content))
        return total
    
    def _summarize_conversation_history(self):
        """
        Summarize conversation history when it exceeds thresholds.
        Strategy: Extract instruction changes + preserve recent messages.
        """
        if len(self.conversation_history) <= self.preserve_recent_messages:
            return
        
        # Split into older and recent messages
        older_messages = self.conversation_history[:-self.preserve_recent_messages]
        recent_messages = self.conversation_history[-self.preserve_recent_messages:]
        
        # Extract key information from older messages
        instruction_changes = []
        for i in range(0, len(older_messages), 2):  # Process user-assistant pairs
            if i + 1 < len(older_messages):
                # Handle both dict format and LangChain message objects
                user_msg_obj = older_messages[i]
                assistant_msg_obj = older_messages[i + 1]
                
                if isinstance(user_msg_obj, dict):
                    user_msg = user_msg_obj.get("content", "")
                elif hasattr(user_msg_obj, 'content'):
                    user_msg = user_msg_obj.content
                else:
                    user_msg = str(user_msg_obj)
                
                if isinstance(assistant_msg_obj, dict):
                    assistant_msg = assistant_msg_obj.get("content", "")
                elif hasattr(assistant_msg_obj, 'content'):
                    assistant_msg = assistant_msg_obj.content
                else:
                    assistant_msg = str(assistant_msg_obj)
                
                # Extract instruction-related information
                if any(keyword in user_msg.lower() for keyword in 
                       ["create", "add", "modify", "change", "update", "remove", "tool"]):
                    summary = f"User requested: {user_msg}..."
                    if len(assistant_msg) > 0:
                        summary += f" | Response: {assistant_msg}..."
                    instruction_changes.append(summary)
        
        # Create compact summary
        if instruction_changes:
            self.conversation_summary = "Previous conversation summary:\n" + "\n".join(
                f"- {change}" for change in instruction_changes
            )
        
        # Replace conversation history with recent messages only
        self.conversation_history = recent_messages
        
        logger.info(f"Conversation summarized: {len(older_messages)} messages compressed, "
                    f"{len(recent_messages)} recent messages preserved")
    
    def _check_and_summarize_if_needed(self):
        """Check if conversation needs summarization and apply it."""
        message_count = len(self.conversation_history)
        token_count = self._estimate_conversation_tokens()
        
        should_summarize = (
            message_count >= self.max_conversation_messages or 
            token_count >= self.max_conversation_tokens
        )
        
        if should_summarize:
            logger.info(f"Triggering summarization: {message_count} messages, "
                        f"~{token_count} tokens")
            self._summarize_conversation_history()
        
    def reset_conversation(self):
        """Reset for a new conversation"""
        self.conversation_history = []
        self.conversation_summary = ""
        self.current_instruction = AgentInstruction(
            role="",
            responsibility="",
            key_tool=None,
            process_title="",
            process_steps=[],
            tool_usage_title=None,
            tool_steps=[]
        )
        self.previous_instruction = AgentInstruction(
            role="",
            responsibility="",
            key_tool=None,
            process_title="",
            process_steps=[],
            tool_usage_title=None,
            tool_steps=[]
        )

    async def sync_instructions_from_db(self, agent_id: str):
        """
        Sync current_instruction with latest system_instructions from database.
        Database value always wins (manual edits override chat updates).
        """
        try:
            from src.backend.postgres import get_builder_db
            from src.backend.postgres.builder.models import AgentSettings
            
            # Get database session
            db_gen = get_builder_db()
            db = next(db_gen)
            
            try:
                # Query latest instructions from database
                agent_settings = (
                    db.query(AgentSettings)
                    .filter(AgentSettings.agent_uuid == agent_id)
                    .first()
                )
                
                if agent_settings and agent_settings.system_instructions:
                    db_instructions = agent_settings.system_instructions.strip()
                    
                    # Parse database instructions into AgentInstruction format
                    if db_instructions:
                        # Extract instruction components from text
                        parsed_instruction = self._parse_text_to_instruction(db_instructions)
                        
                        # Update current_instruction with database value
                        self.current_instruction = parsed_instruction
                        
                        logger.info(f"✅ Synced instructions from database for agent {agent_id}")
                        logger.info(f"   Instruction length: {len(db_instructions)} chars")
                else:
                    logger.info(f"ℹ️ No instructions found in database for agent {agent_id}")
                    
            finally:
                db.close()
                
        except Exception as e:
            logger.warning(f"⚠️ Could not sync instructions from database: {e}")
            # Continue with existing instructions if sync fails

    def _parse_text_to_instruction(self, text: str) -> AgentInstruction:
        """
        Parse plain text instructions into AgentInstruction format.
        Handles both structured (with role/steps) and unstructured text.
        """
        try:
            # Try to extract structured components
            role_match = re.search(r'\*\*Role:\*\*\s*(.+?)(?:\n|$)', text)
            resp_match = re.search(r'\*\*Primary Responsibility:\*\*\s*(.+?)(?:\n|$)', text)
            steps_match = re.findall(r'•\s*(.+?)(?:\n|$)', text)
            
            return AgentInstruction(
                role=role_match.group(1).strip() if role_match else "",
                responsibility=resp_match.group(1).strip() if resp_match else "",
                process_title="Main Process",
                process_steps=steps_match if steps_match else []
            )
        except Exception as e:
            logger.warning(f"Could not parse structured instruction: {e}")
            # Fallback: use entire text as primary responsibility
            return AgentInstruction(
                role="",
                responsibility=text,
                process_title="Main Process",
                process_steps=[]
            )

    async def save_instruction_to_db(self, agent_id: str, formatted_instruction: str):
        """Save formatted instruction back to database."""
        try:
            from src.backend.postgres import get_builder_db
            from src.backend.postgres.builder.models import AgentSettings
            
            db_gen = get_builder_db()
            db = next(db_gen)
            
            try:
                agent_settings = (
                    db.query(AgentSettings)
                    .filter(AgentSettings.agent_uuid == agent_id)
                    .first()
                )
                
                if agent_settings:
                    agent_settings.system_instructions = formatted_instruction
                    db.commit()
                    logger.info(f"✅ Saved instruction to database for agent {agent_id}")
                else:
                    logger.warning(f"⚠️ No agent settings found for agent {agent_id}")
            finally:
                db.close()
                
        except Exception as e:
            logger.warning(f"⚠️ Could not save instruction to database: {e}")

    async def reset_all_configuration(self, thread_id):
        """Reset all agent configuration including state, checkpointer, and related data"""
        print(f"🔄 CONFIGURATION RESET: Starting reset for assistant instance {id(self)}")
        logger.info("🔄 Resetting all agent configuration...")
        
        # First, clear the actual checkpointer data from the database
        print(f"   Clearing checkpointer data from database...")
        checkpointer_cleared = await self.clear_checkpointer_state(thread_id)
        if checkpointer_cleared:
            print(f"   ✅ Checkpointer data cleared from database")
        else:
            print(f"   ⚠️ Warning: Checkpointer data may not have been fully cleared")
        
        # Log current state before reset
        print(f"   Current thread_id: {thread_id}")
        print(f"   Current checkpointer: {self.checkpointer}")
        print(f"   Conversation history length: {len(self.conversation_history)}")
        print(f"   Current instruction: {self.current_instruction}")
        
        # Reset conversation state
        print(f"   Resetting conversation state...")
        self.reset_conversation()
        
        # Reset thread ID to force new checkpointer state
        old_thread_id = thread_id
        self.thread_id = generate_thread_id()
        
        # Properly close checkpointer before setting to None
        if self.checkpointer:
            await self.close_checkpointer()
        else:
            self.checkpointer = None
            
        print(f"   Thread ID changed: {old_thread_id} -> {thread_id}")
        print(f"   Checkpointer cleared: {self.checkpointer}")
        
        # Reset timing and performance data
        self.llm_total_time = 0.0
        self.vector_search_time = 0.0
        self.total_processing_time = 0.0
        print(f"   Timing data reset")
        
        # Reset conversation summarization settings to defaults
        self.max_conversation_messages = 10
        self.max_conversation_tokens = 3000
        self.preserve_recent_messages = 4
        print(f"   Conversation settings reset to defaults")
        
        print(f"✅ CONFIGURATION RESET COMPLETE: Assistant instance {id(self)} reset successfully")
        logger.info("✅ All agent configuration reset successfully")
        
        return {
            "success": True,
            "message": "All agent configuration has been reset",
            "reset_data": {
                "conversation_history": True,
                "current_instruction": True,
                "previous_instruction": True,
                "thread_id": True,
                "checkpointer": True,
                "checkpointer_data_cleared": checkpointer_cleared,
                "timing_data": True,
                "conversation_summary": True,
                "messages_cleared": True,
                "state_emptied": True
            }
        }

    def create_empty_state(self):
        """Create a completely empty AgentConfigState with no messages"""
        from src.backend.agent_configuration.state import AgentConfigState
        
        return AgentConfigState(
            messages=[],  # Empty messages list
            current_instruction={},
            previous_instruction={},
            tool_categories={},
            recommended_tools=[],
            available_tools=[],
            needs_clarification=False,
            questions=[],
            message_analysis={},
            user_input="",
            agent_id="",
            message_type="",
            extraction_result={},
            assistant_response="",
            current_node="",
            streaming_events=[],
            vector_search_results=[],
            llm_ranked_tools=[],
            tool_confidence_scores={},
            processing_stage="initialized",
            node_timing_history={},
            retry_attempts={},
            original_agent_request="",
            clarification_history=[],
            accumulated_context=""
        )

    def get_empty_state(self):
        """Get a completely empty AgentConfigState with no messages for reset operations"""
        return self.create_empty_state()

    async def clear_checkpointer_state(self, thread_id):
        """Clear checkpointer state for the current thread using graph deleteState method"""
        try:
            await self._initialize_checkpointer()
            
            if self.checkpointer:
                print(f"🗑️ CLEARING CHECKPOINTER STATE: For thread_id: {thread_id}")
                
                self.graph = compile_agent_config_graph(self, self.checkpointer)

                # Use graph's deleteState method to clear all state for the thread
                config = {"configurable": {"thread_id": thread_id}}
                
                # Delete all state for this thread using the graph's deleteState method
                await self.graph.adelete_state(config)
                
                print(f"✅ CHECKPOINTER STATE CLEARED: All state removed for thread: {thread_id}")
                logger.info(f"✅ Cleared checkpointer state for thread: {thread_id}")
                return True
            else:
                logger.info(f"ℹ️ No checkpointer to clear for thread: {thread_id}")
                return True
        except Exception as e:
            print(f"❌ ERROR clearing checkpointer state: {e}")
            logger.warning(f"⚠️ Could not clear checkpointer state: {e}")
            return False

    @time_method("Process Clarification Answers", log_args=True)
    async def process_clarification_answers(self, user_input: str, current_instruction: Dict, clarification_questions: List[str]) -> Dict:
        """Process user input as answers to clarification questions and generate updated instruction."""
        try:
            # Create prompt for the LLM to process the clarification answers
            clarification_prompt = f"""
You are an AI assistant helping to refine agent configuration based on user clarification answers.

Current agent instruction:
{json.dumps(current_instruction, indent=2)}

Original clarification questions that were asked:
{json.dumps(clarification_questions, indent=2)}

User's answers/clarifications:
{user_input}

Your task:
1. Use the user's input as answers to fill in missing elements or clarify unclear requirements in the current instruction
2. Generate an updated, more complete instruction based on the user's clarifications
3. Determine if the instruction is now complete and clear enough to proceed
4. If still unclear, generate new clarification questions for remaining gaps

Return a JSON response with:
{{
    "updated_instruction": {{
        "role": "extracted role",
        "responsibility": "extracted responsibility",
        "process_steps": ["step1", "step2"]
    }},
    "clarification_questions": ["question1", "question2"],
    "needs_clarification": false,
    "confidence_score": 0.8,
    "analysis": "Brief analysis of what was clarified and what remains unclear"
}}

Instructions:
- If the user's answers provide sufficient detail to create a complete instruction, set needs_clarification to false
- If there are still gaps or ambiguities, set needs_clarification to true and provide specific questions
- The updated_instruction should incorporate all the user's clarifications
- Be specific and actionable in the instruction details
- Only ask for clarification on truly essential missing information
"""

            # Call the LLM to process the clarification using ainvoke
            with LLMTimingContext("Process Clarification Answers", self):
                response = await self.llm.ainvoke([HumanMessage(content=clarification_prompt)])
                response_content = response.content.strip() if hasattr(response, 'content') else ""
            
            # Clean response content for Gemini
            response_content = self._extract_json_from_content(response_content)
                          
            return json.loads(response_content)
            
        except Exception as e:
            logger.error(f"Error processing clarification answers: {e}")
            # Return fallback response
            return {
                "updated_instruction": current_instruction,
                "clarification_questions": ["Could you provide more specific details about your requirements?"],
                "needs_clarification": True,
                "confidence_score": 0.1,
                "analysis": f"Error processing clarification: {str(e)}"
            }

    def _extract_json_from_content(self, content: str) -> str:
        """Extract JSON content from ```json code blocks if present."""
        if "```json" in content and "```" in content:
            # Find the start and end of the JSON code block
            start_marker = "```json"
            end_marker = "```"
            
            start_idx = content.find(start_marker)
            if start_idx != -1:
                # Move past the ```json marker
                start_idx += len(start_marker)
                # Find the closing ```
                end_idx = content.find(end_marker, start_idx)
                if end_idx != -1:
                    # Extract the JSON content between the markers
                    json_content = content[start_idx:end_idx].strip()
                    return json_content
        
        return content
    
def time_llm_call(method_name: str = None):
        """Decorator to time LLM calls and log the duration"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                method_display_name = method_name or func.__name__
                
                # print(f"🕐 Starting LLM call: {method_display_name}")
                
                try:
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    # print(f"✅ LLM call completed: {method_display_name} - Duration: {duration:.2f}s")
                    return result
                    
                except Exception as e:
                    end_time = time.time()
                    duration = end_time - start_time
                    print(f"❌ LLM call failed: {method_display_name} - Duration: {duration:.2f}s - Error: {str(e)}", file=sys.stderr)
                    raise
                    
            return wrapper
        return decorator

    
