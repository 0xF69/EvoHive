from pydantic import BaseModel


class ThinkerDNA(BaseModel):
    """一个Thinker的'思维基因'"""
    persona: str
    knowledge_bias: str
    constraint: str


class Thinker(BaseModel):
    """一个独立的LLM Agent"""
    id: str
    dna: ThinkerDNA
    model: str = "deepseek/deepseek-chat"
