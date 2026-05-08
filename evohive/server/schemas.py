from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProviderModelDiscoverRequest(APIModel):
    provider: str = Field(min_length=1, max_length=64)
    api_key: str = Field(min_length=1, max_length=4096)


class RunEstimateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    providers: list[str] = Field(default_factory=list, max_length=20)
    provider_models: dict[str, list[str]] = Field(default_factory=dict)
    total: int = Field(default=60, ge=1, le=2000)
    gens: int = Field(default=3, ge=1, le=50)
    mode: str = Field(default="fast", max_length=32)
    budget: float = Field(default=0.5, gt=0, le=100000)
    token_budget_control: str = Field(default="off", max_length=32)
    token_budget_multiplier: float = Field(default=1.0, ge=0.1, le=10.0)
    enable_token_budget_control: bool = False
    enable_search: bool = False
    allow_budget_override: bool = False


class ProviderModelCheckRequest(APIModel):
    provider: str = Field(min_length=1, max_length=64)
    api_key: str = Field(default="", max_length=4096)
    models: list[str] = Field(default_factory=list, max_length=100)


class ProviderPreflightRequest(APIModel):
    providers: list[str] = Field(default_factory=list, max_length=20)
    api_keys: dict[str, str] = Field(default_factory=dict)
