import builtins
import importlib
import runpy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import server
from evohive.server import providers as provider_service


async def _fast_sleep(_: float):
    return None


def test_root_exposes_backend_metadata():
    client = TestClient(server.app)

    response = client.get("/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "EvoHive Backend"
    assert payload["endpoints"]["status"] == "/api/status"


def test_backend_status_exposes_service_metadata():
    client = TestClient(server.app)

    response = client.get("/api/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "EvoHive Backend"
    assert payload["status"] == "ok"
    assert payload["endpoints"]["status"] == "/api/status"
    assert payload["endpoints"]["websocket"] == "/ws"


def test_server_main_starts_without_static_app_reference(monkeypatch):
    calls = []

    def _fake_uvicorn_run(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(server.uvicorn, "run", _fake_uvicorn_run)

    runpy.run_path(str(Path(server.__file__)), run_name="__main__")

    assert calls
    assert calls[0]["kwargs"]["host"] == "0.0.0.0"
    assert calls[0]["kwargs"]["port"] == 8080


def test_websocket_mock_run_returns_structured_results_and_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_has_real_keys", lambda: False)
    monkeypatch.setattr(server, "_has_real_keys_for_config", lambda _config: False)
    monkeypatch.setattr(server, "ARTIFACT_ROOT", tmp_path / "evohive_runs")
    monkeypatch.setattr(server.asyncio, "sleep", _fast_sleep)

    client = TestClient(server.app)
    config = {
        "problem": "Design a pricing strategy for an AI code review tool.",
        "providers": ["deepseek", "gemini"],
        "total": 8,
        "gens": 2,
        "mode": "fast",
        "budget": 0.5,
        "enable_search": False,
        "api_keys": {"deepseek": "deepseek-secret", "gemini": "gemini-secret"},
    }

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start_war", "config": config})

        final_message = None
        seen_types = []
        for _ in range(256):
            message = ws.receive_json()
            seen_types.append(message["type"])
            if message["type"] == "war_complete":
                final_message = message
                break

    assert final_message is not None, f"war_complete not received; saw {seen_types}"

    results = final_message["results"]
    structured = results["structured_result"]
    artifact = results["artifact"]
    telemetry = results["telemetry"]

    assert {
        "executive_summary",
        "action_plan",
        "risks",
        "winner_reason",
        "recommendation",
        "next_steps",
        "decision",
        "alternatives",
    }.issubset(structured.keys())
    assert isinstance(structured["action_plan"], list)
    assert isinstance(structured["risks"], list)
    assert isinstance(structured["alternatives"], list)
    assert isinstance(structured["next_steps"], list)
    assert results["problem"] == config["problem"]
    assert results["event_count"] > 0
    assert results["resource_report"]["version"] == "resource-report.v1"
    assert results["resource_report"]["total_tokens"] > 0
    assert results["token_budget_report"]["version"] == "token-budget-report.v1"
    assert results["trajectory_summary"]["event_count"] > 0
    assert results["trajectory_log"][0]["action"]
    assert results["trajectory_replay"]["version"] == "trajectory-replay.v1"
    assert results["claim_verification_report"]["version"] == "claim-verification-loop.v1"
    assert results["answer_graph"]["version"] == "answer-graph.v1"
    assert results["answer_graph"]["summary"]["claim_node_count"] > 0
    assert any(node["type"] == "claim" for node in results["answer_graph"]["nodes"])
    assert telemetry["event_count"] == results["event_count"]
    assert "phase_stats" in telemetry
    assert "model_roster" in telemetry
    assert "war_started" in seen_types
    assert "generation_summary" in seen_types

    run_dir = Path(artifact["dir"])
    json_path = Path(artifact["json_path"])
    events_path = Path(artifact["events_path"])
    replay_path = Path(artifact["replay_path"])
    telemetry_path = Path(artifact["telemetry_path"])
    report_path = Path(artifact["report_path"])

    assert run_dir.is_dir()
    assert json_path.is_file()
    assert events_path.is_file()
    assert replay_path.is_file()
    assert telemetry_path.is_file()
    assert report_path.is_file()
    assert str(run_dir).startswith(str(tmp_path))
    assert artifact["run_id"].startswith("mock-")

    replay_resp = client.get(f"/api/runs/{artifact['run_id']}/replay")
    assert replay_resp.status_code == 200
    replay_payload = replay_resp.json()
    assert isinstance(replay_payload, list)
    assert replay_payload

    graph_resp = client.get(f"/api/runs/{artifact['run_id']}/answer-graph")
    assert graph_resp.status_code == 200
    graph_payload = graph_resp.json()
    assert graph_payload["version"] == "answer-graph.v1"
    assert graph_payload["summary"]["node_count"] > 0

    trajectory_resp = client.get(f"/api/runs/{artifact['run_id']}/trajectory-replay")
    assert trajectory_resp.status_code == 200
    assert trajectory_resp.json()["version"] == "trajectory-replay.v1"

    claim_resp = client.get(f"/api/runs/{artifact['run_id']}/claim-verification")
    assert claim_resp.status_code == 200
    assert claim_resp.json()["version"] == "claim-verification-loop.v1"

    run_resp = client.get(f"/api/runs/{artifact['run_id']}")
    assert run_resp.status_code == 200
    run_payload = run_resp.json()
    assert run_payload["run_id"] == artifact["run_id"]
    assert "telemetry" in run_payload
    assert run_payload["config"]["api_keys"] == {
        "deepseek": "[REDACTED]",
        "gemini": "[REDACTED]",
    }

    list_resp = client.get("/api/runs")
    assert list_resp.status_code == 200
    listed = list_resp.json()["runs"]
    assert any(item["run_id"] == artifact["run_id"] for item in listed)


def test_model_pool_supports_single_provider_single_model():
    providers = server._normalize_selected_providers(["deepseek"])
    model_pool = server._build_litellm_model_pool(
        providers,
        {"deepseek": ["deepseek-chat"]},
    )

    assert providers == ["deepseek"]
    assert model_pool == ["deepseek/deepseek-chat"]
    assert server._build_litellm_model_pool(["deepseek"], {"deepseek": ["deepseek/deepseek-chat"]}) == [
        "deepseek/deepseek-chat"
    ]


def test_websocket_mock_run_supports_single_provider_population(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_has_real_keys", lambda: False)
    monkeypatch.setattr(server, "_has_real_keys_for_config", lambda _config: False)
    monkeypatch.setattr(server, "ARTIFACT_ROOT", tmp_path / "evohive_runs")
    monkeypatch.setattr(server.asyncio, "sleep", _fast_sleep)

    client = TestClient(server.app)
    config = {
        "problem": "Find the healthiest simple lunch plan.",
        "providers": ["deepseek"],
        "provider_models": {"deepseek": ["deepseek-chat"]},
        "total": 7,
        "gens": 1,
        "mode": "fast",
        "budget": 0.5,
        "enable_search": False,
        "api_keys": {"deepseek": "deepseek-secret"},
    }

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start_war", "config": config})

        final_message = None
        war_started = None
        for _ in range(160):
            message = ws.receive_json()
            if message["type"] == "war_started":
                war_started = message
            if message["type"] == "war_complete":
                final_message = message
                break

    assert war_started is not None
    assert war_started["config"]["providers"] == ["deepseek"]
    assert war_started["config"]["single_model_mode"] is True
    assert final_message is not None
    assert final_message["results"]["champion"]["provider"] == "deepseek"
    assert final_message["results"]["telemetry"]["model_roster"]["providers"][0]["provider"] == "deepseek"


def test_api_key_helpers_redact_session_keys():
    normalized = server._normalize_api_keys({
        "openai": "  session-openai  ",
        "anthropic": "",
        "gemini": None,
        "unknown": "ignored",
    })
    assert normalized == {"openai": "session-openai"}

    redacted = server._redact_config({
        "providers": ["openai", "anthropic"],
        "api_keys": {"openai": "secret-1", "anthropic": "secret-2"},
        "search_api_keys": {"tavily": "search-secret-1", "serper": "search-secret-2"},
    })
    assert redacted["api_keys"] == {
        "openai": "[REDACTED]",
        "anthropic": "[REDACTED]",
    }
    assert redacted["search_api_keys"] == {
        "tavily": "[REDACTED]",
        "serper": "[REDACTED]",
    }


def test_real_mode_detection_accepts_session_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    assert server._has_real_keys_for_config({"api_keys": {"openai": "session-openai"}}) is True
    assert server._has_real_keys_for_config({"api_keys": {"openai": "   "}}) is False


def test_list_runs_skips_corrupt_manifests(monkeypatch, tmp_path):
    root = tmp_path / "evohive_runs"
    bad_dir = root / "bad-123-corrupt"
    bad_dir.mkdir(parents=True)
    (bad_dir / "run.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(server, "ARTIFACT_ROOT", root)

    client = TestClient(server.app)
    response = client.get("/api/runs")

    assert response.status_code == 200
    assert response.json() == {"runs": []}


def test_get_run_rejects_invalid_run_id():
    client = TestClient(server.app)

    response = client.get("/api/runs/bad$id")

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid run_id"


def test_checkpoint_endpoints(monkeypatch, tmp_path):
    from evohive.engine.checkpoint import save_checkpoint
    from evohive.engine.elimination_memory import EvolutionMemory
    from evohive.models import EvolutionConfig, Solution

    checkpoint_dir = tmp_path / "checkpoints"
    monkeypatch.setattr(server, "CHECKPOINT_DIR", checkpoint_dir)
    save_checkpoint(
        "run_ckpt_001",
        2,
        [Solution(content="checkpoint solution")],
        EvolutionMemory(),
        EvolutionConfig(problem="checkpoint problem", mode="fast"),
        checkpoint_dir=str(checkpoint_dir),
    )

    client = TestClient(server.app)
    list_response = client.get("/api/checkpoints")
    assert list_response.status_code == 200
    assert list_response.json()["checkpoints"][0]["run_id"] == "run_ckpt_001"

    get_response = client.get("/api/checkpoints/run_ckpt_001")
    assert get_response.status_code == 200
    assert get_response.json()["generation"] == 2


def test_discover_provider_models_endpoint(monkeypatch):
    monkeypatch.setattr(
        server,
        "_discover_provider_models_with_source",
        _fake_discover_provider_models_with_source,
    )
    client = TestClient(server.app)
    response = client.post(
        "/api/provider-models/discover",
        json={"provider": "openai", "api_key": "session-openai"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "openai"
    assert payload["models"] == ["gpt-5.4", "gpt-5.4-mini"]
    assert payload["source"] == "live"


def test_discover_provider_models_validates_required_fields():
    client = TestClient(server.app)

    response = client.post("/api/provider-models/discover", json={"provider": "openai"})

    assert response.status_code == 422


def test_provider_model_check_endpoint(monkeypatch):
    async def _fake_manual_probe(provider: str, api_key: str, model_ids: list[str]):
        assert provider == 'anthropic'
        assert api_key == 'session-anthropic'
        assert model_ids == ['claude-opus-4.7', 'claude-sonnet-4']
        return {
            'provider': provider,
            'ok': False,
            'mode': 'verified',
            'source': 'live',
            'valid': ['claude-sonnet-4'],
            'invalid': ['claude-opus-4.7'],
            'unchecked': [],
        }

    monkeypatch.setattr(server, '_probe_manual_models', _fake_manual_probe)
    client = TestClient(server.app)
    response = client.post(
        '/api/providers/model-check',
        json={
            'provider': 'anthropic',
            'api_key': 'session-anthropic',
            'models': ['claude-opus-4.7', 'claude-sonnet-4'],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload['mode'] == 'verified'
    assert payload['invalid'] == ['claude-opus-4.7']


def test_provider_preflight_endpoint(monkeypatch):
    async def _fake_probe(provider: str, api_key: str):
        if provider == "anthropic":
            return {
                "provider": provider,
                "ok": False,
                "error_code": "quota_exhausted",
                "message": "quota exhausted",
                "models": [],
            }
        return {
            "provider": provider,
            "ok": True,
            "error_code": None,
            "message": "ready",
            "models": ["gpt-5.4"],
        }

    monkeypatch.setattr(server, "_probe_provider_access", _fake_probe)
    client = TestClient(server.app)
    response = client.post(
        "/api/providers/preflight",
        json={
            "providers": ["openai", "anthropic"],
            "api_keys": {"openai": "session-openai", "anthropic": "session-anthropic"},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is True
    assert payload["ready_count"] == 1
    assert len(payload["providers"]) == 2
    assert any(item["error_code"] == "quota_exhausted" for item in payload["providers"])


def test_provider_preflight_reports_auth_failures(monkeypatch):
    async def _raise_auth(_provider: str, _api_key: str):
        request = provider_service.httpx.Request("GET", "https://example.test/models")
        response = provider_service.httpx.Response(401, request=request)
        raise provider_service.httpx.HTTPStatusError("unauthorized", request=request, response=response)

    monkeypatch.setattr(provider_service, "discover_provider_models_live", _raise_auth)
    client = TestClient(server.app)

    response = client.post(
        "/api/providers/preflight",
        json={"providers": ["openai"], "api_keys": {"openai": "bad-key"}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is False
    assert payload["providers"][0]["error_code"] == "auth_failed"


def test_discover_provider_models_falls_back_to_registry(monkeypatch):
    async def _raise_live_error(_provider: str, _api_key: str):
        raise RuntimeError("live discovery unavailable")

    monkeypatch.setattr(provider_service, "discover_provider_models_live", _raise_live_error)
    monkeypatch.setattr(provider_service, "registry_models_for_provider", lambda _provider: ["fallback-model"])
    client = TestClient(server.app)

    response = client.post(
        "/api/provider-models/discover",
        json={"provider": "openai", "api_key": "session-openai"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "fallback"
    assert payload["models"] == ["fallback-model"]


def test_estimate_run_endpoint_returns_budget_risk():
    client = TestClient(server.app)
    response = client.post(
        "/api/runs/estimate",
        json={
            "providers": ["anthropic", "openai"],
            "provider_models": {"anthropic": ["claude-sonnet-4"], "openai": ["gpt-5.4"]},
            "total": 180,
            "gens": 6,
            "mode": "deep",
            "budget": 0.1,
            "enable_search": True,
            "token_budget_control": "strict",
            "token_budget_multiplier": 0.7,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "estimate" in payload
    assert payload["estimate"]["risk"] in {"soft", "hard"}
    assert payload["guarded"] is False
    assert payload["token_budget_control"]["mode"] == "strict"
    assert payload["token_budget_control"]["enabled"] is True
    assert payload["token_budget_control"]["multiplier"] == 0.7


def test_token_budget_settings_support_user_switches():
    assert server._resolve_token_budget_settings({"token_budget_control": "off"})["enabled"] is False
    assert server._resolve_token_budget_settings({"token_budget_control": "strict"})["mode"] == "strict"
    assert server._resolve_token_budget_settings({"enable_token_budget_control": True})["mode"] == "auto"
    assert server._resolve_token_budget_settings({
        "token_budget_control": "off",
        "enable_token_budget_control": True,
    })["enabled"] is False


def test_websocket_budget_guard_returns_structured_error(monkeypatch):
    monkeypatch.setattr(server, "_has_real_keys_for_config", lambda _config: True)

    async def _budget_guard(*_args, **_kwargs):
        raise server.BudgetGuardError("Estimated cost $1.20-$2.40 may exceed the configured budget ($1.00).")

    monkeypatch.setattr(server, "run_real_evolution", _budget_guard)

    client = TestClient(server.app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({
            "type": "start_war",
            "config": {
                "problem": "Launch a pricing strategy.",
                "providers": ["anthropic", "openai"],
                "provider_models": {"anthropic": ["claude-sonnet-4"], "openai": ["gpt-5.4"]},
                "total": 180,
                "gens": 6,
                "mode": "deep",
                "budget": 0.1,
                "api_keys": {"anthropic": "session-anthropic", "openai": "session-openai"},
            },
        })
        message = ws.receive_json()

    assert message["type"] == "war_error"
    assert message["error_code"] == "budget_exceeded"
    assert message["retryable"] is True
    assert "error_id" in message
    assert "traceback" not in message
    assert "budget_estimate" in message
    assert message["budget_estimate"]["risk"] in {"soft", "hard"}


def test_websocket_rejects_malformed_json_without_traceback():
    client = TestClient(server.app)
    with client.websocket_connect("/ws") as ws:
        ws.send_text("{not json")
        message = ws.receive_json()

    assert message["type"] == "war_error"
    assert message["error_code"] == "invalid_message"
    assert message["retryable"] is False
    assert "error_id" in message
    assert "traceback" not in message


def test_websocket_rejects_non_object_start_config_without_traceback():
    client = TestClient(server.app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start_war", "config": "not-an-object"})
        message = ws.receive_json()

    assert message["type"] == "war_error"
    assert message["error_code"] == "invalid_message"
    assert message["retryable"] is False
    assert "error_id" in message
    assert "traceback" not in message


def test_llm_provider_does_not_patch_global_print():
    original_print = builtins.print

    importlib.import_module("evohive.llm.provider")

    assert builtins.print is original_print


def test_llm_session_api_keys_are_context_local():
    llm_provider = importlib.import_module("evohive.llm.provider")

    token = llm_provider.set_session_api_keys({
        "openai": "session-openai",
        "together": "session-together",
    })
    try:
        assert llm_provider._resolve_model("openai/gpt-4o")["api_key"] == "session-openai"
        assert llm_provider._resolve_model("together_ai/meta-llama/example")["api_key"] == "session-together"
    finally:
        llm_provider.reset_session_api_keys(token)

    assert "api_key" not in llm_provider._resolve_model("openai/gpt-4o")


@pytest.mark.asyncio
async def test_web_search_uses_session_keys_without_environment(monkeypatch):
    search_module = importlib.import_module("evohive.engine.web_search")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    captured = {}

    async def _fake_tavily(_query: str, api_key: str, _max_results: int):
        captured["api_key"] = api_key
        return [{"title": "ok", "snippet": "session key used", "url": "https://example.test"}]

    monkeypatch.setattr(search_module, "_search_tavily", _fake_tavily)
    token = search_module.set_session_search_api_keys({"tavily": "session-tavily"})
    try:
        results = await search_module.web_search("test query")
    finally:
        search_module.reset_session_search_api_keys(token)

    assert captured["api_key"] == "session-tavily"
    assert results[0]["title"] == "ok"


async def _fake_discover_provider_models_with_source(provider: str, api_key: str):
    assert provider == "openai"
    assert api_key == "session-openai"
    return ["gpt-5.4", "gpt-5.4-mini"], "live"
