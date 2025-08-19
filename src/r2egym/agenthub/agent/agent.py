import os
import re
import copy
import yaml
import json
import time
import uuid
import hmac
import hashlib
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel

import litellm
from openai import OpenAI, AzureOpenAI
import openai
from azure.identity import (
    ChainedTokenCredential,
    AzureCliCredential,
    DefaultAzureCredential,
    get_bearer_token_provider,
)
from tenacity import (
    Retrying,
    retry_if_not_exception_type,
    retry_if_exception_message,
    stop_after_attempt,
    wait_random_exponential,
)
from dotenv import load_dotenv

from r2egym.agenthub.action import Action
from r2egym.agenthub.utils.log import get_logger
from r2egym.agenthub.environment.env import RepoEnv
from r2egym.agenthub.runtime.docker import DockerRuntime
from r2egym.agenthub.trajectory import TrajectoryStep, Trajectory
from anthropic import Anthropic, AnthropicVertex  # Add Anthropic Vertex import
from r2egym.agenthub.tools import (
    r2egym_bash_execute_tool,
    search_tool,
    file_editor,
    finish_tool,
    str_replace_editor_tool,
    execute_bash_tool,
    submit_tool,
)
import traceback
logger = get_logger(__name__)  # Logger for this module
MAX_CONTEXT_TOKENS = 65536

# Azure LLM Model supported models from SWE-agent
AZURE_SUPPORTED_MODELS = [
    "gpt-4o",
    "o3",
    "o3-mini",
    "o4-mini",
    "gpt-4.1",
    "gpt-4.5-preview",
    "o1",
    "gpt-4.1-mini"
]

# CopilotClaudeModel supported models from SWE-agent
COPILOT_CLAUDE_SUPPORTED_MODELS = [
    "claude-sonnet-4",
    "gpt-4.1-2025-04-14",
    "gpt-3.5-turbo-0613",
    "gpt-4o-mini-2024-07-18",
    "gpt-4-0613",
    "gpt-4-0125-preview",
    "gpt-4o-2024-11-20",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "o3-mini-2025-01-31",
    "o3-mini-paygo",
    "gpt-4o-copilot",
    "text-embedding-3-small",
    "claude-3.5-sonnet",
    "claude-3.7-sonnet",
    "claude-3.7-sonnet-thought",
    "claude-opus-4",
    "claude-opus-41",
    "gemini-2.0-flash-001",
    "o3-2025-04-16",
    "o4-mini-2025-04-16",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano-2025-04-14",
    "oswe-vscode",
    "gpt-4.1-oswe-control",
]

##############################################################################
# AgentArgs Dataclass
##############################################################################
@dataclass
class AgentArgs:
    system_prompt: str
    instance_prompt: str
    command_files: List[Path]
    llm_name: str
    llm_base_url: Optional[str] = "http://localhost:8000/v1"  # None
    demo_file: Optional[Path] = None
    use_demo: Optional[bool] = False
    other_args: Optional[Dict[str, Any]] = None  # To handle extra configurations

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "AgentArgs":
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)
        return cls(**config)


##############################################################################
# Agent Class
##############################################################################
class Agent:
    """Agent handles the behavior of the model and how it interacts with the environment."""

    def __init__(self, name: str, args: AgentArgs, logger=None):
        self.name = name
        self.args = args
        # self.trajectory_steps: List[TrajectoryStep] = []
        if logger is None:
            self.logger = get_logger(name)  # initialize logger from the agent name
        else:
            self.logger = logger
        self.llm_name = args.llm_name

        self.llm_base_url = (
            # "http://localhost:8000/v1"
            os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
            if ("openai/" in self.llm_name) or ("hosted_vllm" in self.llm_name)
            else None
        )
        self.system_prompt_template = args.system_prompt
        self.instance_prompt_template = args.instance_prompt
        self.command_files = args.command_files
        self.other_args = args.other_args or {}
        self.logger.info(f"Initialized Agent: {name} with LLM: {args.llm_name}")
        self.max_retries = self.other_args.get("max_retries", 5)
        self.llm_timeout = self.other_args.get("timeout", 3000)



    def prepare_system_message(
        self, problem_statement: str, structure: str, command_docs: str, demo: str
    ) -> str:
        """Prepare the system prompt by filling in placeholders."""
        system_prompt = self.system_prompt_template.format(
            # problem_statement=problem_statement,
            # structure=structure,
            command_docs=command_docs,
            demo=demo,
        )
        return system_prompt

    def prepare_instance_prompt(
        self, agent_history: str, command_docs: str, steps_remaining: int
    ) -> str:
        """Prepare the instance prompt by filling in placeholders."""
        instance_prompt = self.instance_prompt_template.format(
            agent_history=agent_history,
            command_docs=command_docs,
        )
        # self.logger.info(isinstance(steps_remaining, int))
        # Add steps remaining message
        if steps_remaining > 0:
            stepcount_message = f"Steps Remaining: {steps_remaining}"
        else:
            stepcount_message = "You have reached the maximum number of steps. Please submit your answer NOW."
        instance_prompt += f"\n{stepcount_message}"
        self.logger.info(stepcount_message)  # Log the steps remaining message
        return instance_prompt

    def prepare_history_message(self, include_all_obs=False) -> str:
        """Prepare the agent's message history as a string."""
        history = ""
        for idx, step in enumerate(self.trajectory_steps):
            thought = step.thought
            action = step.action
            observation = step.observation
            # history += f'THOUGHT:\n```\n{thought}\n```\n'
            # history += f'ACTION:\n```\n{action}\n```\n'
            action_template = """
            {thought}
            ```
            {action}
            ```
            """
            history += action_template.format(thought=thought, action=action)
            if idx == len(self.trajectory_steps) - 1 or include_all_obs:
                history += f"\nOBSERVATION:\n```\n{observation}\n```\n"
            # add a separator
            history += "-" * 50 + "\n"
        return history

    def reset(self):
        """Reset the agent's trajectory."""
        self.trajectory_steps = []
        self.history = []

    def _count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Counts the tokens for a list of messages using the litellm library.
        Adjust as needed depending on the model and library.
        """
        token_count = litellm.token_counter(model=self.llm_name, messages=messages)
        self.logger.info(f"Total tokens in conversation: {token_count}")
        return token_count

    def model_query(
        self, messages: List[Dict[str, str]], temperature: float = 0, k_responses: int = 1) -> Tuple[List[Dict[str, Any]], float]:
        """Query the LLM with the messages and measure execution time. 
        
        Args:
            messages: List of message dictionaries for the conversation
            temperature: Temperature for response generation
            k_responses: Number of responses to generate (default=1 for backward compatibility)
            
        Returns:
            Tuple of (list of responses, execution time). When k_responses=1, returns ([single_response], exec_time)
        """
        response = None
        retries = 0
        tools = None

        if self.use_fn_calling:
            if self.scaffold == "r2egym":
                tools = [search_tool, file_editor, r2egym_bash_execute_tool, finish_tool]
            elif self.scaffold == "openhands" or self.scaffold == "sweagent":
                tools = [str_replace_editor_tool, execute_bash_tool, submit_tool]
            if "vertex" not in self.llm_name.lower():
                self.logger.warning(f"using prompt caching for {self.llm_name}")
                # vertex is not supported yet: https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude-prompt-caching
                # litellm might need dev install with vertex: https://github.com/BerriAI/litellm/issues/6898
                # add prompt caching for anthropic
                tools[-1]["function"]["cache_control"] = {"type": "ephemeral"}
                breakpoints_remaining = 3  # remaining 1 for system/tool (above)
                for message in reversed(messages):
                    if message["role"] in ("user", "tool"):
                        if breakpoints_remaining > 0:
                            message["cache_control"] = {"type": "ephemeral"}
                            breakpoints_remaining -= 1
                        else:
                            break

        # Start timer
        start_time = time.time()
        # check if using locally hosted models
        using_local = "openai/" in self.llm_name or "hosted" in self.llm_name
        if using_local:
            litellm.api_key = None

        messages_ = copy.deepcopy(messages)
        total_tokens = self._count_tokens(messages_)
        if total_tokens > MAX_CONTEXT_TOKENS:
            logger.warning(f"Total tokens: {total_tokens} > {MAX_CONTEXT_TOKENS}")
            raise ValueError(f"Total tokens: {total_tokens} > {MAX_CONTEXT_TOKENS}")
        
        # query the model with retries
        while retries < self.max_retries:
            try:
                kwargs = {
                    "tool_choice": "none",
                    "function_call": None,
                }
                if tools:
                    kwargs = {}
                if "o3" not in self.llm_name and "o4" not in self.llm_name:
                    kwargs["temperature"] = temperature
                # Handle prefix-based routing
                if self.llm_name.startswith("trapi-"):
                    # Extract model name after trapi- prefix
                    actual_model = self.llm_name[6:]  # Remove "trapi-" prefix
                    assert actual_model in AZURE_SUPPORTED_MODELS, f"Model '{actual_model}' not found in Azure supported models after removing 'trapi-' prefix"
                    response = self._azure_api_call(
                        model=actual_model,
                        tools=tools,
                        messages=messages_,
                        timeout=self.llm_timeout,
                        api_base=self.llm_base_url,
                        **kwargs,
                    )
                elif self.llm_name.startswith("capi-"):
                    # Extract model name after capi- prefix
                    actual_model = self.llm_name[5:]  # Remove "capi-" prefix
                    assert actual_model in COPILOT_CLAUDE_SUPPORTED_MODELS, f"Model '{actual_model}' not found in Copilot Claude supported models after removing 'capi-' prefix"
                    response = self._copilot_claude_api_call(
                        model=actual_model,
                        tools=tools,
                        messages=messages_,
                        timeout=self.llm_timeout,
                        api_base=self.llm_base_url,
                        **kwargs,
                    )
                else:
                    response = litellm.completion(
                        model=self.llm_name,
                        tools=tools,
                        messages=messages_,
                        timeout=self.llm_timeout,
                        api_base=self.llm_base_url,
                        # max_tokens=3000,
                        **kwargs,
                    )
                self.logger.warning(f"Querying LLM complete")
                break
            except Exception as e:
                self.logger.error(f"LLM query failed @ {retries}: {e}")
                retries += 1
                if "RateLimitError" in str(e):
                    time.sleep(60)
                if retries >= self.max_retries:
                    raise e

        # End timer, calculate total execution time, and include in response
        exec_time = time.time() - start_time
        
        # For k_responses > 1, generate additional responses
        responses = [response]
        if k_responses > 1:
            for i in range(k_responses - 1):
                additional_retries = 0
                while additional_retries < self.max_retries:
                    try:
                        # Use same parameters as the first response
                        kwargs = {
                            "tool_choice": "none",
                            "function_call": None,
                        }
                        if tools:
                            kwargs = {}
                        if "o3" not in self.llm_name and "o4" not in self.llm_name:
                            kwargs["temperature"] = temperature
                        
                        # Handle prefix-based routing for additional responses
                        if self.llm_name.startswith("trapi-"):
                            # Extract model name after trapi- prefix
                            actual_model = self.llm_name[6:]  # Remove "trapi-" prefix
                            assert actual_model in AZURE_SUPPORTED_MODELS, f"Model '{actual_model}' not found in Azure supported models after removing 'trapi-' prefix"
                            additional_response = self._azure_api_call(
                                model=actual_model,
                                tools=tools,
                                messages=messages_,
                                timeout=self.llm_timeout,
                                api_base=self.llm_base_url,
                                **kwargs,
                            )
                        elif self.llm_name.startswith("capi-"):
                            # Extract model name after capi- prefix
                            actual_model = self.llm_name[5:]  # Remove "capi-" prefix
                            assert actual_model in COPILOT_CLAUDE_SUPPORTED_MODELS, f"Model '{actual_model}' not found in Copilot Claude supported models after removing 'capi-' prefix"
                            additional_response = self._copilot_claude_api_call(
                                model=actual_model,
                                tools=tools,
                                messages=messages_,
                                timeout=self.llm_timeout,
                                api_base=self.llm_base_url,
                                **kwargs,
                            )
                        else:
                            additional_response = litellm.completion(
                                model=self.llm_name,
                                tools=tools,
                                messages=messages_,
                                timeout=self.llm_timeout,
                                api_base=self.llm_base_url,
                                **kwargs,
                            )
                        responses.append(additional_response)
                        self.logger.warning(f"Generated additional response {i+2}/{k_responses}")
                        break
                    except Exception as e:
                        self.logger.error(f"Additional LLM query {i+2} failed @ {additional_retries}: {e}")
                        additional_retries += 1
                        if "RateLimitError" in str(e):
                            time.sleep(60)
                        if additional_retries >= self.max_retries:
                            self.logger.error(f"Failed to generate additional response {i+2}/{k_responses} after {self.max_retries} retries")
                            # Continue without this response rather than failing completely
                            break
        
        return responses, exec_time

    def _get_copilot_client(self):
        """Get or create a cached GitHub Copilot client"""
        if not hasattr(self, '_copilot_client') or self._copilot_client is None:
            self._copilot_client = self._create_copilot_client()
        return self._copilot_client

    def _create_copilot_client(self):
        """Create GitHub Copilot client with proper authentication"""
        def create_request_hmac(hmac_secret: str) -> str:
            """Create HMAC for request authentication"""
            current = str(int(time.time()))
            signature = hmac.new(
                hmac_secret.encode("utf-8"), current.encode("utf-8"), hashlib.sha256
            ).hexdigest()
            return f"{current}.{signature}"

        def fetch_token() -> str:
            """Fetch GitHub Copilot token using Node.js script"""
            try:
                vscode_copilot_dir = (
                    os.environ.get("VSCODE_COPILOT_DIR", os.path.expanduser("~/repo/vscode-copilot"))
                )
                vscode_copilot_dir = os.path.expanduser(vscode_copilot_dir)
                if not os.path.exists(vscode_copilot_dir):
                    raise ValueError(f"vscode-copilot directory not found at: {vscode_copilot_dir}")
                
                result = subprocess.run(
                    ["npx", "tsx", "src/util/node/fetch-token-standalone.js"],
                    capture_output=True,
                    text=True,
                    cwd=vscode_copilot_dir,
                )
                
                if result.returncode != 0:
                    raise ValueError(f"Failed to fetch token: {result.stderr}")
                
                token = result.stdout.strip()
                if not token:
                    raise ValueError("fetch-token.js returned empty output")
                
                return token
            except Exception as e:
                raise ValueError(f"Failed to get Copilot token: {e}")

        # Load environment variables
        vscode_copilot_dir = (
            os.environ.get("VSCODE_COPILOT_DIR", os.path.expanduser("~/repo/vscode-copilot"))
        )
        env_file_path = os.path.expanduser(os.path.join(vscode_copilot_dir, ".env"))

        if not os.environ.get("HMAC_SECRET") and os.path.exists(env_file_path):
            try:
                load_dotenv(dotenv_path=env_file_path)
            except Exception as e:
                self.logger.warning("Failed to load .env file: %s", e)

        hmac_secret = os.environ.get("HMAC_SECRET")
        if not hmac_secret:
            raise ValueError("HMAC_SECRET not found in environment variables")
        
        bearer_token = fetch_token()
        hmac_value = create_request_hmac(hmac_secret)

        # Create OpenAI client
        client = OpenAI(
            api_key=bearer_token,
            base_url=self.llm_base_url or "https://api.enterprise.githubcopilot.com",
            default_headers={
                "X-Interaction-Type": "conversation-agent",
                "OpenAI-Intent": "conversation-agent",
                "X-GitHub-Api-Version": "2025-05-01",
                "Copilot-Integration-Id": "vscode-chat-dev",
                "VScode-SessionId": "r2egym-session",
                "VScode-MachineId": "r2egym-machine",
                "X-Interaction-Id": str(uuid.uuid4()),
                "X-Initiator": "agent",
                "Editor-Version": "r2egym/1.0",
                "Editor-Plugin-Version": "r2egym/1.0",
                "Request-Hmac": hmac_value,
            },
            timeout=self.llm_timeout,
        )
        return client

    def _copilot_claude_api_call(self, model: str, messages: List[Dict[str, str]], 
                                 tools: Optional[List[Dict]] = None, timeout: int = 60, 
                                 api_base: Optional[str] = None, **kwargs) -> Any:
        """
        Call the GitHub Copilot Claude API using the cached client with retry logic for HMAC timestamp errors.
        """
        def retry_warning(retry_state):
            exception = retry_state.outcome.exception() if retry_state.outcome else None
            if exception:
                self.logger.warning(
                    f"Retrying Copilot Claude query (attempt {retry_state.attempt_number}) due to {exception.__class__.__name__}: {exception}"
                )
            # Special handling for HMAC timestamp errors - clear client to force new token
            if isinstance(exception, openai.AuthenticationError) and "HMAC timestamp out of range" in str(exception):
                self.logger.info("Refreshing client due to HMAC timestamp error")
                self._copilot_client = None  # Clear client to force recreation with new HMAC

        # Retry logic for HMAC timestamp errors
        for attempt in Retrying(
            stop=stop_after_attempt(20),  # Default retry count from SWE-agent
            wait=wait_random_exponential(min=10, max=120),  # Default wait times from SWE-agent  
            reraise=True,
            retry=retry_if_not_exception_type((
                # Don't retry these errors
                KeyboardInterrupt,
                openai.BadRequestError,
            )) | retry_if_exception_message(match="HMAC timestamp out of range"),
            before_sleep=retry_warning,
        ):
            with attempt:
                client = self._get_copilot_client()

                # Build chat request
                request_kwargs = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": 8192,
                }
                
                # Add temperature and top_p for models that support them
                not_temperature_models = [
                    "o3-mini-2025-01-31",
                    "o3-mini-paygo", 
                    "o3-2025-04-16",
                    "o4-mini-2025-04-16",
                ]
                if model not in not_temperature_models:
                    if "temperature" in kwargs:
                        request_kwargs["temperature"] = kwargs["temperature"]
                    if "top_p" in kwargs:
                        request_kwargs["top_p"] = kwargs["top_p"]
                
                if tools:
                    request_kwargs["tools"] = tools
                    request_kwargs["tool_choice"] = "auto"

                # Make the API call
                response = client.chat.completions.create(**request_kwargs)
                
                return response

    def _get_azure_client(self, model: str):
        """Get or create a cached Azure OpenAI client for the specified model"""
        # Create a cache key based on the model to handle different models
        cache_key = f"_azure_client_{model}"
        if not hasattr(self, cache_key) or getattr(self, cache_key) is None:
            setattr(self, cache_key, self._create_azure_client(model))
        return getattr(self, cache_key)

    def _create_azure_client(self, model: str):
        """Create Azure OpenAI client with proper authentication"""
        # Model metadata mapping from SWE-agent
        model_meta = {
            "gpt-4o": ("2024-11-20", "msrne/shared", "2024-10-21"),
            "o3": ("2025-04-16", "msrne/shared", "2025-04-01-preview"),
            "o3-mini": ("2025-01-31", "msrne/shared", "2025-04-01-preview"),
            "o4-mini": ("2025-04-16", "msrne/shared", "2025-04-01-preview"),
            "gpt-4.1": ("2025-04-14", "gcr/shared", "2025-04-01-preview"),
            "gpt-4.5-preview": ("2025-02-27", "msrne/shared", "2025-04-01-preview"),
            "o1": ("2024-12-17", "msrne/shared", "2025-04-01-preview"),
            "gpt-4.1-mini": ("2025-04-14", "msrne/shared", "2025-04-01-preview"),
        }
        
        if model not in model_meta:
            raise ValueError(f"{model} not in supported Azure models {AZURE_SUPPORTED_MODELS}")
            
        version, instance, api_version = model_meta[model]
        deployment_name = re.sub(r"[^a-zA-Z0-9._-]", "", f"{model}_{version}")
        endpoint = f"https://trapi.research.microsoft.com/{instance}"

        # Set up Azure credentials
        credential = get_bearer_token_provider(
            ChainedTokenCredential(
                AzureCliCredential(),
                DefaultAzureCredential(
                    exclude_cli_credential=True,
                    exclude_environment_credential=True,
                    exclude_shared_token_cache_credential=True,
                    exclude_developer_cli_credential=True,
                    exclude_powershell_credential=True,
                    exclude_interactive_browser_credential=True,
                    exclude_visual_studio_code_credentials=True,
                    managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
                ),
            ),
            "api://trapi/.default",
        )

        # Create Azure OpenAI client
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=credential,
            api_version=api_version,
        )
        
        # Store deployment name for API calls (using model-specific attribute)
        setattr(self, f"_azure_deployment_name_{model}", deployment_name)
        
        return client

    def _azure_api_call(self, model: str, messages: List[Dict[str, str]], 
                        tools: Optional[List[Dict]] = None, timeout: int = 60, 
                        api_base: Optional[str] = None, **kwargs) -> Any:
        """
        Call the Azure OpenAI API using the cached client.
        """
        client = self._get_azure_client(model)

        # Build Azure request arguments
        deployment_name = getattr(self, f"_azure_deployment_name_{model}")
        azure_kwargs = {
            "model": deployment_name,
            "messages": messages,
        }
        
        # Models that don't support custom temperature or top_p
        not_temperature_models = ["o1", "o3", "o3-mini", "o4-mini"]
        if model not in not_temperature_models:
            if "temperature" in kwargs:
                azure_kwargs["temperature"] = kwargs["temperature"]
            if "top_p" in kwargs:
                azure_kwargs["top_p"] = kwargs["top_p"]
        
        if tools:
            azure_kwargs["tools"] = tools

        # Make the API call
        response = client.chat.completions.create(**azure_kwargs)
        
        return response

    def parse_response(self, response: Dict[str, Any]) -> Tuple[str, Action]:
        """
        Parse the response from the LLM.
        """
        """
        Extracts:
        - thought: first thing in <think>...</think> block
        - action: the entire first <function=...></function> block
        Returns (thought, action).
        """
        # Regex to match (non-greedily) from `<think>` up to the first `</think>`
        pattern_thought = re.compile(r"(?s)(<think>.*?</think>)")
        pattern_action = re.compile(r"(?s)(<function=.*?</function>)")
        match_thought = pattern_thought.search(response)
        match_action = pattern_action.search(response)

        if match_thought:
            thought = match_thought.group(1)  # The entire <think>...</think> block
        else:
            thought = ""
        if match_action:
            action = match_action.group(1)  # The entire <function=...></function> block
        else:
            action = ""
        # Strip leading/trailing whitespace
        thought = thought.strip()
        action = action.strip()

        # convert action to Action object
        action = Action.from_string(action)

        return thought, action

    def parse_response_v2(self, response_text: str) -> Tuple[str, Action]:
        """
        Extracts:
        - thought: everything before the first <function=...> block
        - action: the entire first <function=...></function> block
        Returns (thought, action).
        """
        # Regex to match (non-greedily) from `<function=` up to the first `</function>`
        pattern = re.compile(r"(?s)(<function=.*?</function>)")
        match = pattern.search(response_text)

        if match:
            action = match.group(1)  # The entire <function=...></function> block
            thought = response_text[: match.start()]  # Everything before the block
        else:
            # If no match, treat entire text as "thought"
            thought = response_text
            action = ""

        # Strip leading/trailing whitespace
        thought = thought.strip()
        action = action.strip()

        # convert action to Action object
        action = Action.from_string(action)

        return thought, action

    def custom_parser(self, response):
        thought = response.choices[0].message.content
        if not thought:
            thought = ""

        try:
            function_name = response.choices[0].message.tool_calls[0].function.name
            parameters = json.loads(
                response.choices[0].message.tool_calls[0].function.arguments
            )
            action = Action(function_name=function_name, parameters=parameters)
        except:
            action = Action(function_name="", parameters={})

        return thought, action

    def run(
        self,
        env: "RepoEnv",  # env: RepoEnv
        use_fn_calling: bool = True,
        # step limits TODO: maybe add these limits in the agent args
        max_steps: int = 10,
        max_steps_absolute: int = 50,
        # token limits
        max_token_limit: int = 65536,  # 64k tokens
        # time limits
        max_exec_time: int = 90,  # 5 mins per env execution
        max_total_time: int = 50000,  # 20 minutes overall agent run limit
        max_llm_time: int = 7200,  # 2 mins per LLM timeout (note this is per query exlcuding retries | not enforcing hard limit since llm might hit rate limits etc)
        # temperature
        temperature=0,
        # additional metadata e.g. for hints / additional inputs etc
        metadata: Optional[Dict[str, Any]] = {},
        scaffold: str = "r2egym",
        # k responses support
        k_responses: int = 1,
    ):
        assert scaffold in ["r2egym", "openhands", "sweagent"], "Scaffold must be either r2egym or openhands or sweagent"
        self.scaffold = scaffold
        # get the start time
        start_time = time.time()
        self.llm_timeout = max_llm_time

        # if self.llm_name is not gpt or sonnet, disable fn calling
        support_fn_calling = (
            "gpt" in self.llm_name
            or "sonnet" in self.llm_name
            or "o3" in self.llm_name
            or "o4" in self.llm_name
            and "qwen" not in self.llm_name
        )
        self.use_fn_calling = use_fn_calling and support_fn_calling
        self.logger.warning(f"Using fn calling: {self.use_fn_calling}")

        # Log the environment and agent
        self.logger.info(f"Running agent {self.name} in environment {env}.")

        # Reset the environment and the agent
        env.reset()
        env.add_commands(self.command_files)
        self.reset()

        # Prepare problem_statement and structure from the environment
        problem_statement = env.runtime.get_task_instruction()
        self.logger.info(f"Problem Statement: {problem_statement}")
        
        # Get GT patch - handle different modes (regular, swesmith, 1r1m)
        if hasattr(env.runtime, 'commit') and env.runtime.commit is not None:
            gt_patch = env.runtime.commit.get_patch(test_file=True, non_test_file=False)
        else:
            # For swesmith or 1r1m mode where commit object doesn't exist
            gt_patch = ""

        # get system and instance prompts
        system_prompt = self.system_prompt_template
        user_prompt = self.instance_prompt_template.format(
            problem_statement=problem_statement,
            gt_patch=gt_patch,
            working_dir='/testbed',
            # base_commit=env.runtime.ds['base_commit'],
            test_patch_hint=metadata.get("test_patch_hint", ""),
            candidate_patch=metadata.get("candidate_patch", ""),
            candidate_patch_correctness=(
                "correct"
                if metadata.get("candidate_patch_correctness", False)
                else "incorrect"
            ),
        )
        self.logger.info(f"User Prompt: {user_prompt}")

        if self.args.use_demo:
            with open(self.args.demo_file, "r") as file:
                demo = file.read()
            user_prompt = f"{demo}\n\n{user_prompt}"
        self.logger.info(f"User Prompt with demo: {user_prompt}")

        # initialize the history
        self.history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # initialize the parameters
        obs = None
        done = False
        step_count = 0
        total_time_traj = 0
        self.trajectory_steps: List[TrajectoryStep] = []

        # agent loop
        while not done:
            # Prepare the agent's message history
            # self.logger.info(isinstance(steps_remaining, int))
            # Add steps remaining message
            steps_remaining = max_steps - step_count
            if steps_remaining > 0:
                stepcount_message = f"Steps Remaining: {steps_remaining}"
            else:
                stepcount_message = "You have reached the maximum number of steps. Please submit your answer NOW."
            self.history[-1][
                "content"
            ] += f"\n{stepcount_message}"  # postpend stepcount message
            self.logger.info(stepcount_message)

            # Query the LLM
            messages = copy.deepcopy(self.history)
            try:
                responses, llm_exec_time = self.model_query(messages, temperature, k_responses)
                response = responses[0]  # Use first response for execution
                alternative_responses = responses[1:] if len(responses) > 1 else None  # Store alternatives
            except Exception as e:
                self.logger.error(f"Error querying LLM: {e}")
                self.logger.error(f"Error querying LLM: {traceback.format_exc()}")
                done = True
                exit_reason = "llm_query_error"
                break

            # Log total tokens in the response
            if hasattr(response, "usage"):
                usage = response.usage
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)
                total_tokens = getattr(usage, "total_tokens", 0)

                prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
                self.logger.warning(f"Prompt Token Details: {prompt_tokens_details}")
                self.logger.info(
                    f"Prompt Tokens: {prompt_tokens}\nCompletion Tokens: {completion_tokens}\nTotal Tokens: {total_tokens}"
                )
            else:
                completion_tokens = -1
                prompt_tokens = -1
                total_tokens = -1
                total_tokens =  self._count_tokens(messages)
                self.logger.warning(
                    "No token usage information available in the response."
                )

            # Parse the LLM response to get 'thought' and 'action'
            self.response = response  # for debugging
            assistant_message = response.choices[0].message.content
            self.logger.info(f"Assistant's message:\n{assistant_message}\n")

            if self.use_fn_calling:
                thought, action = self.custom_parser(response)
            else:
                thought, action = self.parse_response(assistant_message)
            
            # Parse alternative responses for structured data
            parsed_alternative_responses = []
            if alternative_responses:
                for i, alt_resp in enumerate(alternative_responses):
                    try:
                        if self.use_fn_calling:
                            alt_thought, alt_action = self.custom_parser(alt_resp)
                        else:
                            alt_message = alt_resp.choices[0].message.content if hasattr(alt_resp, 'choices') and alt_resp.choices else ""
                            alt_thought, alt_action = self.parse_response(alt_message)
                        
                        parsed_alternative_responses.append({
                            "thought": alt_thought,
                            "action": alt_action.to_xml_string()
                        })
                    except Exception as e:
                        self.logger.error(f"Failed to parse alternative response {i+2}: {e}")
                        parsed_alternative_responses.append({
                            "thought": "",
                            "action": "",
                            "parse_error": str(e)
                        })

            # action_str = action.to_xml_string()
            self.logger.info(f"THOUGHT:\n{thought}\n")
            self.logger.info(f"ACTION:\n{action.to_bashcmd()}\n")

            # Send the action to the environment
            try:
                obs, reward, done, info = env.step(action, timeout=max_exec_time)
                # env.runtime.commit_after_step(step_count)
            except Exception as e:
                obs = str(e)
                self.logger.error(f"Error during environment step: {obs}")

            env_exec_time = info["total_time"]
            total_step_time = llm_exec_time + env_exec_time
            total_time_traj += total_step_time
            step_count += 1  # Increment the step count

            if self.use_fn_calling:
                assistant_response = response.choices[0].message.dict()
                if assistant_response.get("tool_calls", None):
                    assistant_response["tool_calls"] = assistant_response["tool_calls"][
                        :1
                    ]  # only keep the first tool call
                self.history.append(assistant_response)
                # add tool response / user response to history
                try:
                    function_name = (
                        response.choices[0].message.tool_calls[0].function.name
                    )
                    function_id = response.choices[0].message.tool_calls[0].id
                    self.history.append(
                        {
                            "role": "tool",
                            "content": str(obs),
                            "name": function_name,
                            "tool_call_id": function_id,
                        }
                    )
                    self.logger.warning("logging fn response as a tool call")
                    self.logger.warning(
                        f"number of fn calls: {len(response.choices[0].message.tool_calls)}"
                    )
                except Exception as e:
                    self.logger.error(f"Error logging tool response: {e}")
                    self.logger.warning("fallback: logging fn response as a tool call")
                    self.history.append({"role": "user", "content": str(obs)})
            else:
                self.logger.warning("logging fn response as a user message")
                assistant_message = f"{thought}\n\n{action.to_xml_string()}"
                # assistant_message = f"{thought}\n\n{original_xml_str}"
                self.history.append({"role": "assistant", "content": assistant_message})
                self.history.append({"role": "user", "content": str(obs)})

            # Log the thought, action, and observation
            self.logger.info(f"OBSERVATION:\n{obs}\n")
            self.logger.info("-" * 50)

            # Check if the agent has reached limits or done
            # check if agent has finished naturally i.e. the agent uses the finish tool
            if done:
                if steps_remaining > 0:
                    self.logger.info(
                        f"Agent has finished naturally before step limit. current step count: {step_count}. max steps: {max_steps}."
                    )
                    exit_reason = "agent"
                elif steps_remaining == 0:
                    self.logger.info(
                        f"Agent finised on reaching the maximum number of steps: {max_steps}. current step count: {step_count}."
                    )
                    exit_reason = "max_step_limit"
                else:
                    self.logger.info(
                        f"Agent has finished after continuing past the max steps: {max_steps}. current step count: {step_count}."
                    )
                    exit_reason = "agent_max_step_limit"
            # check for token limit
            elif total_tokens >= max_token_limit:
                self.logger.info(
                    f"Agent reached max tokens: {max_token_limit}. Current token count: {total_tokens}. Exiting."
                )
                exit_reason = "token_limit"
                done = True
            # check for absolute step limit | note that the max steps is just indicative but the absolute step limit is the hard limit
            elif step_count >= max_steps_absolute:
                self.logger.info(
                    f"Agent reached max steps: {max_steps_absolute}. Exiting."
                )
                exit_reason = "abs_step_limit"
                done = True

            elif total_time_traj >= max_total_time:
                self.logger.info(f"Agent reached max time: {max_total_time}. Exiting.")
                exit_reason = "traj_time_limit"
                done = True

            # Create a TrajectoryStep object and append to the list
            trajectory_step = TrajectoryStep(
                # key parts
                step_idx=step_count - 1,
                thought=thought,
                action=action.to_xml_string(),
                observation=str(obs),
                done=done,
                info=info,  # also store the info to be safe
                # tokens
                token_usage_prompt=prompt_tokens,
                token_usage_completion=completion_tokens,
                token_usage_total=total_tokens,
                # metadata (current step stats)
                llm_exec_time=llm_exec_time,
                env_exec_time=env_exec_time,
                total_step_time=total_step_time,
                total_time_traj=total_time_traj,
                step_count=step_count,
                # k_responses support - store alternative responses
                alternative_responses=parsed_alternative_responses if parsed_alternative_responses else None,
            )
            self.trajectory_steps.append(trajectory_step)

        # get the output patch
        # output_patch, _ = env.runtime.run(f"git diff {initial_commit} HEAD")
        # output_patch, _ = env.runtime.run(f"git diff {initial_commit} HEAD -- . ':(exclude)pyproject.toml'")
        # env.runtime.soft_git_reset()

        # compute output patch cummulatively from the start using git diff from the initial commit
        output_patch = env.runtime.get_patch()

        # Create a Trajectory object
        self.trajectory = Trajectory(
            trajectory_steps=[
                traj_step.model_dump() for traj_step in self.trajectory_steps
            ],
            problem_statement=problem_statement,
            docker_image=env.runtime.docker_image,
            agent_args=asdict(self.args),
            env_args=asdict(env.args),
            max_steps=max_steps,
            max_steps_absolute=max_steps_absolute,
            max_token_limit=max_token_limit,
            max_llm_time=max_llm_time,
            max_exec_time=max_exec_time,
            max_total_time=max_total_time,
            exit_reason=exit_reason,  # reason for exiting. must be one of the [agent, max_step_limit, agent_max_step_limit, abs_step_limit, token_limit, traj_time_limit, llm_query_error]
            output_patch=output_patch,
        )

        self.logger.info(f"Agent completed in {time.time() - start_time} seconds.")
        return self.trajectory