from typing import List, Optional, ClassVar
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from Utils.LLM import LlamaSSHLLM

class Agent(BaseModel):
    
    name: str
    role: str
    llm_name: str
    messages: List[BaseMessage] = []
    system_message: Optional[SystemMessage] = None
    description: str = ''
    
    agents_registry: ClassVar[List['Agent']] = []
    
    def __init__(self, **data):
        super().__init__(**data)
        Agent.agents_registry.append(self)
    

    def add_message(self, message: BaseMessage):
        self.messages.append(message)
        
        for agent in Agent.agents_registry:
            if agent != self:
                agent.messages.append(message)

    def get_message_history(self) -> List[BaseMessage]:
        return self.messages

    def set_system_message(self, content: str):
        self.system_message = SystemMessage(content=content)

    
    def generate_response(self) -> AIMessage:
        """
        Generate a response using the language model based on the message history.
        """
        # Ensure system message is included
        if self.llm_name == 'llama':
             llm = LlamaSSHLLM()
        else:
             from langchain_openai import ChatOpenAI
             llm = ChatOpenAI(model=self.llm_name, temperature=0)
        prompt_messages = [self.system_message] if self.system_message else []
        prompt_messages.extend(self.messages)

        response = llm.invoke(prompt_messages)
        response.name = self.name
        self.add_message(response)
        return response

class Debater(Agent):
    """
    A class representing a debater with a specific stance on a topic.
    """
    stance: str

   

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nStance: {self.stance}\nDescription: {self.description}\n"

class Moderator(Agent):
    """
    A class representing the moderator of the debate.
    """


    def evaluate_debate(self, round_number: int) -> AIMessage:
        return self.generate_response()

    def decide_to_continue(self, evaluation_content: str) -> bool:
        return "yes" not in evaluation_content.lower()

class Judge(Agent):
    """
    A class representing the judge who provides the final verdict.
    """


    def provide_verdict(self) -> AIMessage:
        return self.generate_response()