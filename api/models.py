"""API request/response models for the RAG service.

Extracted verbatim from api/search.py (Tier 1 refactor) — field types,
defaults and Field constraints are preserved exactly. No behaviour change.
"""
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query:      str       = Field(..., min_length=1, max_length=1000)
    top_k:      int       = Field(5, ge=1, le=10)
    visitor_id: str       = Field(default="v_prod_guest")
    namespace:  str       = Field(default="all")
    source:     str       = Field(default="api")
    user_agent: str       = Field(default="unknown")
    last_query:  str       = Field(default="")
    last_intent: str       = Field(default="")
    country:     str       = Field(default="")
    company:     str       = Field(default="")


class Source(BaseModel):
    filename:        str
    clean_name:      str
    page:            str
    pdf_url:         str
    page_url:        str = ""
    resource_type:   str = ""
    preview:         str
    relevance_score: float


class SearchResponse(BaseModel):
    query:     str
    answer:    str
    sources:   list[Source]
    followups: list[str]
    blocked:   bool = False
    cached:    bool = False
    intent:            str       = "general"
    detected_products: list[str] = []
    detected_use_case: str       = ""
    rewritten_query:   str       = ""
    confidence:        float     = 0.0
    inherited:         bool      = False
    similarity:        float     = 0.0
    visitor_history:   list      = []
    lead_quality_tag:  str       = "EARLY_EXPLORER"
    resource_types:    list[str] = []
    detected_workloads: list[str] = []


class IdentifyRequest(BaseModel):
    visitor_id: str
    email:      str
    name:       str = ""
    source:     str = "commercial_nudge"
    products:   str = ""
    company:    str = ""
    country:    str = ""


class SummariseRequest(BaseModel):
    filename: str = Field(..., min_length=1, max_length=500)


class SummariseResponse(BaseModel):
    filename:       str
    summary:        str = ""
    key_topics:     list[str] = []
    suggested_name: str = ""
    suggested_type: str = ""
    cached:         bool = False
    error:          str = ""
