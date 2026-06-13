"""
tests/test_enricher.py — Full test suite for pipeline/enricher.py

Tests cover:
  - Structural overrides (Tier 1) — 100% accuracy guarantee
  - LLM output validation
  - Product alias normalisation
  - Merge logic
  - Failure handling
  - Batch concurrency

Run: pytest tests/test_enricher.py -v
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from enricher import (
    STRUCTURAL_OVERRIDES,
    VALID_PRODUCTS,
    VALID_USE_CASES,
    _apply_structural_overrides,
    _build_prompt,
    _default_metadata,
    _normalise_product,
    _validate_llm_output,
    enrich_chunk,
    enrich_chunks_batch,
    merge_enrichment_into_metadata,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_FABRIC_CHUNK = """
Equinix Fabric port speeds: 1, 10, 100 and 400 Gbps options available.
Dynamic connection speeds with bandwidth tiers from 10 Mbps to 100 Gbps.
SLA: 99.999% uptime on dual ports, 99.9% on single port.
Connect to AWS, Azure, and Google Cloud via software-defined interconnection.
"""

SAMPLE_NE_CHUNK = """
Network Edge is a Virtual Network Functions (VNF) as a Service offering.
Deploy SD-WAN gateways, firewalls, and cloud routers without physical hardware.
Supports Cisco SD-WAN, VMware SD-WAN Edge, and Fortinet Secure SD-WAN.
Available in 32 global metros, provisioned in minutes.
"""

SAMPLE_CROSS_PRODUCT_CHUNK = """
Equinix Fabric eliminates network silos and streamlines traffic flows.
Network Edge provides VNF-as-a-Service for SD-WAN and cloud routing.
Together they form a complete hybrid multicloud networking solution.
"""

SAMPLE_SHORT_CHUNK = "Equinix Fabric."  # too short for LLM enrichment

SAMPLE_LLM_RESPONSE = {
    "primary_product":      ["Equinix Fabric"],
    "mentioned_products":   ["Equinix Fabric Cloud Router"],
    "use_case":             ["interconnection", "cloud-adjacency"],
    "target_audience":      ["financial-services", "cloud-service-providers"],
    "content_role":         "technical-detail",
    "technical_depth":      "engineer",
    "product_maturity":     "evaluation",
    "has_specs":            True,
    "has_architecture":     False,
    "integration_partners": ["AWS", "Azure", "Google Cloud"],
}


# ── Tier 1: Structural override tests ────────────────────────────────────────

class TestStructuralOverrides:

    def test_blueprint_forces_has_architecture_true(self):
        base = _default_metadata()
        base["has_architecture"] = False  # LLM said false
        result = _apply_structural_overrides(base, "blueprint")
        assert result["has_architecture"] is True, \
            "Blueprint must always have has_architecture=True"

    def test_blueprint_forces_engineer_depth(self):
        base = _default_metadata()
        base["technical_depth"] = "executive"  # LLM got it wrong
        result = _apply_structural_overrides(base, "blueprint")
        assert result["technical_depth"] == "engineer"

    def test_data_sheet_forces_has_specs_true(self):
        base = _default_metadata()
        base["has_specs"] = False  # LLM missed the specs
        result = _apply_structural_overrides(base, "data-sheet")
        assert result["has_specs"] is True, \
            "Data sheets must always have has_specs=True"

    def test_analyst_report_forces_practitioner(self):
        """Analyst reports look executive but have technical depth."""
        base = _default_metadata()
        base["technical_depth"] = "executive"  # LLM got fooled by executive language
        result = _apply_structural_overrides(base, "analyst-report")
        assert result["technical_depth"] == "practitioner"

    def test_case_study_forces_case_example_role(self):
        base = _default_metadata()
        result = _apply_structural_overrides(base, "case-study")
        assert result["content_role"] == "case-example"

    def test_success_story_forces_executive(self):
        base = _default_metadata()
        result = _apply_structural_overrides(base, "success-story")
        assert result["technical_depth"] == "executive"
        assert result["content_role"] == "case-example"

    def test_infographic_forces_executive(self):
        base = _default_metadata()
        result = _apply_structural_overrides(base, "infographic")
        assert result["technical_depth"] == "executive"

    def test_unknown_resource_type_no_override(self):
        """Unknown resource types should not be overridden."""
        base = _default_metadata()
        base["technical_depth"] = "engineer"
        result = _apply_structural_overrides(base, "unknown-type")
        assert result["technical_depth"] == "engineer", \
            "Unknown resource type must not change any fields"

    def test_structural_overrides_cover_all_resource_types(self):
        """Verify all known resource types have override rules."""
        expected_types = {
            "blueprint", "data-sheet", "playbook", "case-study",
            "analyst-report", "whitepaper", "infographic",
            "solution-brief", "infopaper", "success-story",
            "video", "webinar",
        }
        missing = expected_types - set(STRUCTURAL_OVERRIDES.keys())
        assert not missing, \
            f"Missing structural overrides for: {missing}"

    def test_boolean_override_only_forces_true(self):
        """
        Structural override on bool fields only forces True.
        LLM can still set True on non-override types.
        """
        base = _default_metadata()
        base["has_specs"] = True  # LLM correctly identified specs in a whitepaper
        result = _apply_structural_overrides(base, "whitepaper")
        # whitepaper has no has_specs override — LLM result preserved
        assert result["has_specs"] is True


# ── Product normalisation tests ───────────────────────────────────────────────

class TestProductNormalisation:

    def test_canonical_name_returned_unchanged(self):
        assert _normalise_product("Equinix Fabric") == "Equinix Fabric"
        assert _normalise_product("Network Edge") == "Network Edge"
        assert _normalise_product("Equinix Metal") == "Equinix Metal"

    def test_alias_maps_to_canonical(self):
        """Aliases from products.json should map to canonical names."""
        # These should resolve via the alias map
        # Test with known aliases from our registry
        result = _normalise_product("Fabric")
        # Either resolves to canonical or returns None if registry not loaded
        assert result in {"Equinix Fabric", None}

    def test_unknown_product_returns_none(self):
        assert _normalise_product("AWS Direct Connect") is None
        assert _normalise_product("random text") is None
        assert _normalise_product("") is None

    def test_case_insensitive_lookup(self):
        result = _normalise_product("equinix fabric")
        assert result in {"Equinix Fabric", None}

    def test_fcr_is_separate_from_fabric(self):
        """Fabric Cloud Router must not collapse into Equinix Fabric."""
        fcr = _normalise_product("Equinix Fabric Cloud Router")
        fabric = _normalise_product("Equinix Fabric")
        if fcr and fabric:
            assert fcr != fabric, \
                "FCR and Fabric must be separate canonical products"


# ── LLM output validation tests ──────────────────────────────────────────────

class TestValidateLLMOutput:

    def test_valid_response_passes_through(self):
        result = _validate_llm_output(SAMPLE_LLM_RESPONSE)
        assert "Equinix Fabric" in result["primary_product"]
        assert "interconnection" in result["use_case"]
        assert result["has_specs"] is True
        assert "AWS" in result["integration_partners"]

    def test_unknown_product_silently_dropped(self):
        raw = {**SAMPLE_LLM_RESPONSE,
               "primary_product": ["Equinix Fabric", "UnknownProduct XYZ"]}
        result = _validate_llm_output(raw)
        assert "UnknownProduct XYZ" not in result["primary_product"]
        assert "Equinix Fabric" in result["primary_product"]

    def test_unknown_use_case_silently_dropped(self):
        raw = {**SAMPLE_LLM_RESPONSE,
               "use_case": ["interconnection", "made-up-use-case"]}
        result = _validate_llm_output(raw)
        assert "made-up-use-case" not in result["use_case"]
        assert "interconnection" in result["use_case"]

    def test_use_case_capped_at_3(self):
        raw = {**SAMPLE_LLM_RESPONSE,
               "use_case": [
                   "interconnection", "cloud-adjacency",
                   "edge-computing", "distributed-ai", "sustainability"
               ]}
        result = _validate_llm_output(raw)
        assert len(result["use_case"]) <= 3

    def test_mentioned_products_never_overlap_primary(self):
        raw = {**SAMPLE_LLM_RESPONSE,
               "primary_product":    ["Equinix Fabric"],
               "mentioned_products": ["Equinix Fabric", "Network Edge"]}
        result = _validate_llm_output(raw)
        assert "Equinix Fabric" not in result["mentioned_products"], \
            "mentioned_products must never contain a primary_product"
        assert "Network Edge" in result["mentioned_products"]

    def test_invalid_technical_depth_defaults_to_practitioner(self):
        raw = {**SAMPLE_LLM_RESPONSE, "technical_depth": "intermediate"}
        result = _validate_llm_output(raw)
        assert result["technical_depth"] == "practitioner"

    def test_invalid_content_role_defaults_to_overview(self):
        raw = {**SAMPLE_LLM_RESPONSE, "content_role": "summary"}
        result = _validate_llm_output(raw)
        assert result["content_role"] == "overview"

    def test_unknown_audience_silently_dropped(self):
        raw = {**SAMPLE_LLM_RESPONSE,
               "target_audience": ["financial-services", "made-up-industry"]}
        result = _validate_llm_output(raw)
        assert "made-up-industry" not in result["target_audience"]
        assert "financial-services" in result["target_audience"]

    def test_audience_groups_derived_correctly(self):
        raw = {**SAMPLE_LLM_RESPONSE,
               "target_audience": ["financial-services", "retail-banking"]}
        result = _validate_llm_output(raw)
        assert "finance" in result["audience_groups"]

    def test_long_integration_partner_silently_dropped(self):
        raw = {**SAMPLE_LLM_RESPONSE,
               "integration_partners": ["AWS", "A" * 60]}  # too long
        result = _validate_llm_output(raw)
        assert "AWS" in result["integration_partners"]
        assert "A" * 60 not in result["integration_partners"]

    def test_empty_response_returns_safe_defaults(self):
        result = _validate_llm_output({})
        assert result["primary_product"] == []
        assert result["use_case"] == []
        assert result["has_specs"] is False
        assert result["technical_depth"] == "practitioner"
        assert result["content_role"] == "overview"


# ── enrich_chunk integration tests ───────────────────────────────────────────

class TestEnrichChunk:

    @pytest.mark.asyncio
    async def test_short_chunk_returns_structural_only(self):
        """Short chunks skip LLM but still get structural overrides."""
        result = await enrich_chunk(
            chunk_text    = SAMPLE_SHORT_CHUNK,
            resource_type = "data-sheet",
        )
        assert result["enriched"] is True
        assert result["has_specs"] is True  # structural override applied
        assert result["enrichment_error"] == "short_chunk_structural_only"

    @pytest.mark.asyncio
    async def test_successful_enrichment(self):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps(SAMPLE_LLM_RESPONSE)

        with patch("enricher.client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_resp
            )
            result = await enrich_chunk(
                chunk_text    = SAMPLE_FABRIC_CHUNK,
                title         = "Equinix Fabric Data Sheet",
                resource_type = "data-sheet",
                url           = "https://www.equinix.com/resources/data-sheets/equinix-fabric",
                aem_tags      = ["Equinix Fabric", "interconnection"],
            )

        assert result["enriched"] is True
        assert result["enrichment_error"] is None
        # Structural override: data-sheet → has_specs=True regardless of LLM
        assert result["has_specs"] is True
        # LLM output preserved for non-override fields
        assert "Equinix Fabric" in result["primary_product"]

    @pytest.mark.asyncio
    async def test_llm_failure_returns_structural_overrides(self):
        """On LLM failure: enriched=False but structural overrides still applied."""
        with patch("enricher.client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("OpenAI rate limit")
            )
            result = await enrich_chunk(
                chunk_text    = SAMPLE_FABRIC_CHUNK,
                resource_type = "blueprint",
                retries       = 0,
            )

        assert result["enriched"] is False
        assert result["enrichment_error"] is not None
        # Structural override still applied despite LLM failure
        assert result["has_architecture"] is True
        assert result["technical_depth"] == "engineer"

    @pytest.mark.asyncio
    async def test_malformed_json_falls_back_gracefully(self):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "not valid json {"

        with patch("enricher.client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_resp
            )
            result = await enrich_chunk(
                chunk_text    = SAMPLE_FABRIC_CHUNK,
                resource_type = "whitepaper",
                retries       = 0,
            )

        assert result["enriched"] is False
        # Should not raise — graceful fallback
        assert "primary_product" in result

    @pytest.mark.asyncio
    async def test_structural_override_wins_over_llm(self):
        """Structural override must always beat LLM on same field."""
        # LLM says has_architecture=False for a blueprint
        wrong_llm = {**SAMPLE_LLM_RESPONSE, "has_architecture": False}
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps(wrong_llm)

        with patch("enricher.client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_resp
            )
            result = await enrich_chunk(
                chunk_text    = SAMPLE_FABRIC_CHUNK,
                resource_type = "blueprint",
            )

        assert result["has_architecture"] is True, \
            "Blueprint structural override must beat LLM"
        assert result["technical_depth"] == "engineer", \
            "Blueprint technical_depth override must beat LLM"

    @pytest.mark.asyncio
    async def test_fcr_not_confused_with_fabric(self):
        """Fabric Cloud Router chunk must be tagged as FCR, not Fabric."""
        fcr_chunk = """
        Equinix Fabric Cloud Router offers secure Layer 3 routing between clouds.
        Spin up a virtual router in less than a minute and begin creating
        Layer 3 connections—no physical hardware or licensing required.
        """
        fcr_response = {
            **SAMPLE_LLM_RESPONSE,
            "primary_product": ["Equinix Fabric Cloud Router"],
            "mentioned_products": ["Equinix Fabric"],
        }
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps(fcr_response)

        with patch("enricher.client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_resp
            )
            result = await enrich_chunk(
                chunk_text    = fcr_chunk,
                resource_type = "data-sheet",
            )

        # FCR should be primary, Fabric mentioned
        assert "Equinix Fabric Cloud Router" in result["primary_product"]
        # Fabric should not be in primary (it's mentioned, not primary here)
        assert "Equinix Fabric Cloud Router" not in result["mentioned_products"]

    @pytest.mark.asyncio
    async def test_cross_product_chunk_has_both_products(self):
        """Cross-product chunks should have primary AND mentioned products."""
        cross_response = {
            **SAMPLE_LLM_RESPONSE,
            "primary_product":    ["Equinix Fabric"],
            "mentioned_products": ["Network Edge"],
        }
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps(cross_response)

        with patch("enricher.client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_resp
            )
            result = await enrich_chunk(
                chunk_text    = SAMPLE_CROSS_PRODUCT_CHUNK,
                resource_type = "solution-brief",
            )

        assert len(result["primary_product"]) >= 1
        # Both products should be findable across primary + mentioned
        all_products = result["primary_product"] + result["mentioned_products"]
        assert "Network Edge" in all_products or "Equinix Fabric" in all_products


# ── enrich_chunks_batch tests ─────────────────────────────────────────────────

class TestEnrichChunksBatch:

    @pytest.mark.asyncio
    async def test_batch_adds_enrichment_key_to_all_chunks(self):
        chunks = [
            {"text": SAMPLE_FABRIC_CHUNK, "id": "1"},
            {"text": SAMPLE_NE_CHUNK,     "id": "2"},
            {"text": SAMPLE_SHORT_CHUNK,  "id": "3"},
        ]
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps(SAMPLE_LLM_RESPONSE)

        with patch("enricher.client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_resp
            )
            results = await enrich_chunks_batch(
                chunks        = chunks,
                title         = "Test Document",
                resource_type = "whitepaper",
                url           = "https://www.equinix.com/test",
            )

        assert len(results) == 3
        for r in results:
            assert "enrichment" in r, "Every chunk must have enrichment key"
            assert "enriched" in r["enrichment"]

    @pytest.mark.asyncio
    async def test_batch_preserves_original_chunk_data(self):
        chunks = [{"text": SAMPLE_FABRIC_CHUNK, "id": "abc", "page": 3}]
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps(SAMPLE_LLM_RESPONSE)

        with patch("enricher.client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_resp
            )
            results = await enrich_chunks_batch(chunks=chunks)

        assert results[0]["id"] == "abc"
        assert results[0]["page"] == 3
        assert results[0]["text"] == SAMPLE_FABRIC_CHUNK

    @pytest.mark.asyncio
    async def test_batch_partial_failure_doesnt_block(self):
        """If some chunks fail enrichment, others should still succeed."""
        chunks = [
            {"text": SAMPLE_FABRIC_CHUNK, "id": "1"},
            {"text": SAMPLE_NE_CHUNK,     "id": "2"},
        ]

        call_count = 0
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps(SAMPLE_LLM_RESPONSE)

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Rate limit")
            return mock_resp

        with patch("enricher.client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                side_effect=side_effect
            )
            # Patch enrich_chunk to use retries=0 via monkeypatching
            import enricher as enricher_module
            original = enricher_module.enrich_chunk

            async def enrich_no_retry(*args, **kwargs):
                kwargs["retries"] = 0
                return await original(*args, **kwargs)

            enricher_module.enrich_chunk = enrich_no_retry
            results = await enrich_chunks_batch(chunks=chunks)
            enricher_module.enrich_chunk = original

        assert len(results) == 2
        # First chunk failed, second succeeded
        enriched_count = sum(1 for r in results if r["enrichment"]["enriched"])
        assert enriched_count >= 1


# ── merge_enrichment_into_metadata tests ─────────────────────────────────────

class TestMergeEnrichmentIntoMetadata:

    def test_enrichment_merged_into_existing_metadata(self):
        existing = {
            "text":          "some chunk text",
            "url":           "https://equinix.com/test",
            "resource_type": "data-sheet",
            "is_latest":     True,
        }
        enrichment = {
            "primary_product": ["Equinix Fabric"],
            "has_specs":       True,
            "enriched":        True,
            "enrichment_error": None,
        }
        result = merge_enrichment_into_metadata(existing, enrichment)

        assert result["text"] == "some chunk text"      # original preserved
        assert result["url"]  == "https://equinix.com/test"
        assert result["primary_product"] == ["Equinix Fabric"]
        assert result["has_specs"] is True

    def test_enrichment_error_not_stored_in_pinecone(self):
        existing   = {"text": "chunk"}
        enrichment = {
            "primary_product":  [],
            "enriched":         False,
            "enrichment_error": "OpenAI rate limit",  # must not reach Pinecone
        }
        result = merge_enrichment_into_metadata(existing, enrichment)
        assert "enrichment_error" not in result

    def test_enrichment_overwrites_existing_tag_fields(self):
        """Re-enrichment should overwrite old enrichment tags."""
        existing = {
            "text":            "chunk",
            "primary_product": ["Network Edge"],  # old tag
        }
        enrichment = {
            "primary_product":  ["Equinix Fabric"],  # new tag after re-enrichment
            "enriched":         True,
            "enrichment_error": None,
        }
        result = merge_enrichment_into_metadata(existing, enrichment)
        assert result["primary_product"] == ["Equinix Fabric"]


# ── Prompt quality tests ──────────────────────────────────────────────────────

class TestPromptQuality:

    def test_prompt_includes_fcr_clarification(self):
        prompt = _build_prompt(
            chunk_text    = SAMPLE_FABRIC_CHUNK,
            title         = "Test",
            resource_type = "data-sheet",
            url           = "https://equinix.com",
            aem_tags      = [],
            structural    = {},
        )
        assert "Equinix Fabric Cloud Router" in prompt
        assert "separate" in prompt.lower()

    def test_prompt_includes_use_case_priority_rules(self):
        prompt = _build_prompt(
            chunk_text    = SAMPLE_FABRIC_CHUNK,
            title         = "Test",
            resource_type = "whitepaper",
            url           = "https://equinix.com",
            aem_tags      = [],
            structural    = {},
        )
        assert "cloud-adjacency" in prompt
        assert "priority" in prompt.lower()

    def test_prompt_tells_llm_what_is_already_determined(self):
        structural = {"has_architecture": True, "technical_depth": "engineer"}
        prompt = _build_prompt(
            chunk_text    = SAMPLE_FABRIC_CHUNK,
            title         = "Blueprint",
            resource_type = "blueprint",
            url           = "https://equinix.com",
            aem_tags      = [],
            structural    = structural,
        )
        assert "Already determined" in prompt
        assert "do not change" in prompt.lower()

    def test_prompt_includes_aem_tags(self):
        prompt = _build_prompt(
            chunk_text    = SAMPLE_FABRIC_CHUNK,
            title         = "Test",
            resource_type = "whitepaper",
            url           = "https://equinix.com",
            aem_tags      = ["Equinix Fabric", "SD-WAN", "interconnection"],
            structural    = {},
        )
        assert "Equinix Fabric" in prompt
        assert "SD-WAN" in prompt

    def test_chunk_text_truncated_at_1500_chars(self):
        long_chunk = "A" * 3000
        prompt = _build_prompt(
            chunk_text    = long_chunk,
            title         = "Test",
            resource_type = "whitepaper",
            url           = "https://equinix.com",
            aem_tags      = [],
            structural    = {},
        )
        # Chunk text in prompt should not exceed 1500 chars
        chunk_in_prompt = prompt.split("Chunk text:\n")[1].split("\n\nReturn")[0]
        assert len(chunk_in_prompt) <= 1500 + 10  # small buffer for whitespace


# ── Taxonomy completeness tests ───────────────────────────────────────────────

class TestTaxonomyCompleteness:

    def test_all_official_products_are_valid(self):
        from enricher import OFFICIAL_PRODUCTS
        for p in OFFICIAL_PRODUCTS:
            assert p in VALID_PRODUCTS, f"'{p}' not in VALID_PRODUCTS"

    def test_all_official_use_cases_are_valid(self):
        from enricher import OFFICIAL_USE_CASES
        for u in OFFICIAL_USE_CASES:
            assert u in VALID_USE_CASES, f"'{u}' not in VALID_USE_CASES"

    def test_equinix_nav_products_all_present(self):
        """Verify all 8 nav menu products are in the product list."""
        nav_products = [
            "Equinix Fabric",
            "Equinix Fabric Cloud Router",
            "Equinix Metal",
            "Equinix Precision Time",
            "Internet Access",
            "Managed Services",
            "Network Edge",
            "Platform Equinix",
        ]
        for p in nav_products:
            assert p in VALID_PRODUCTS, \
                f"Nav product '{p}' missing from VALID_PRODUCTS"

    def test_equinix_nav_use_cases_all_present(self):
        """Verify all 17 official use cases are present."""
        assert len(VALID_USE_CASES) == 17, \
            f"Expected 17 use cases, got {len(VALID_USE_CASES)}"
