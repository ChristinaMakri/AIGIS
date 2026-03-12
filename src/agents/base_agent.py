"""
Abstract Base Agent Class
All agents inherit from this class

The Perceive → Decide → Act cycle implemented here is the canonical
BDI (Belief-Desire-Intention) agent execution loop:

  Rao, A.S. & Georgeff, M.P. (1995).
  "BDI agents: From theory to practice."
  Proceedings of ICMAS-95, pp. 312–319. AAAI Press.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np
from ..message import Message


class Agent(ABC):
    """Abstract base class for all agents in the system"""

    def __init__(self, agent_id: str, position: Tuple[float, float]):
        """
        Args:
            agent_id: Unique identifier for the agent
            position: Initial (lat, lon) position
        """
        self.agent_id = agent_id
        self.position = position  # (lat, lon)
        self.grid_position = None  # (row, col) - set by environment
        self.messages_inbox: List[Message] = []
        self.messages_outbox: List[Message] = []
        self.is_active = True

    @abstractmethod
    def perceive(self, environment) -> None:
        """
        Perceive the environment and update internal state.

        Args:
            environment: The simulation environment
        """
        pass

    @abstractmethod
    def decide(self) -> None:
        """
        Make decisions based on perceptions and internal state.
        This is where the agent's reasoning happens.
        """
        pass

    @abstractmethod
    def act(self, environment) -> None:
        """
        Execute actions in the environment.

        Args:
            environment: The simulation environment
        """
        pass

    def send_message(self, message: Message) -> None:
        """Add message to outbox"""
        self.messages_outbox.append(message)

    def receive_message(self, message: Message) -> None:
        """Receive message into inbox"""
        self.messages_inbox.append(message)

    def clear_messages(self) -> None:
        """Clear processed messages"""
        self.messages_inbox.clear()

    def get_outbox_messages(self) -> List[Message]:
        """Get and clear outbox"""
        messages = self.messages_outbox.copy()
        self.messages_outbox.clear()
        return messages

    def update(self, environment) -> None:
        """
        Main update loop: perceive -> decide -> act

        Args:
            environment: The simulation environment
        """
        if not self.is_active:
            return

        self.perceive(environment)
        self.decide()
        self.act(environment)
