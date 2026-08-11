from __future__ import annotations

from pydantic import BaseModel, Field


class ImpressionRequest(BaseModel):
    device_type: str = Field(examples=["flagship_android"])
    os_version: str = Field(examples=["v13"])
    region: str = Field(examples=["south"])
    ad_category: str = Field(examples=["gaming"])
    content_type: str = Field(examples=["lockscreen"])
    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    historical_ctr: float = Field(ge=0.0, le=1.0)
    session_length_sec: float = Field(ge=0.0)


class CTRPrediction(BaseModel):
    model_config = {"protected_namespaces": ()}

    click_probability: float
    model_used: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
