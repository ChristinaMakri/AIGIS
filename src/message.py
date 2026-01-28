"""
FIPA-ACL Message Implementation for Agent Communication
"""
from typing import Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Message:
    """
    FIPA-ACL compliant message structure for agent communication.

    Performatives:
        - INFORM: Inform receiver of some fact
        - REQUEST: Request action from receiver
        - CFP: Call For Proposal (used in Contract Net Protocol)
        - PROPOSE: Propose to perform action (response to CFP)
        - ACCEPT_PROPOSAL: Accept a proposal
        - REJECT_PROPOSAL: Reject a proposal
        - REFUSE: Refuse to perform action
        - CONFIRM: Confirm that action was performed
        - QUERY: Query for information
    """
    sender: str
    receiver: str
    performative: str
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    conversation_id: str = ""
    reply_to: str = ""

    def __post_init__(self):
        """Validate performative type"""
        valid_performatives = {
            'INFORM', 'REQUEST', 'CFP', 'PROPOSE', 'ACCEPT_PROPOSAL',
            'REJECT_PROPOSAL', 'REFUSE', 'CONFIRM', 'QUERY'
        }
        if self.performative not in valid_performatives:
            raise ValueError(f"Invalid performative: {self.performative}")

    def __repr__(self):
        return (f"Message({self.performative}: {self.sender} -> {self.receiver}, "
                f"content={self.content})")
