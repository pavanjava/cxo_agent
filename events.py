from llama_index.core.workflow import Event
from typing import List, Any


class PrefixMessageEvent(Event):
    prefix_messages: List[Any]


class AgentEvent(Event):
    agent: Any
