from pathlib import Path

from fastapi.testclient import TestClient

import server


async def _fast_sleep(_: float):
    return None


def test_websocket_mock_run_returns_structured_results_and_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_has_real_keys", lambda: False)
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

    run_resp = client.get(f"/api/runs/{artifact['run_id']}")
    assert run_resp.status_code == 200
    run_payload = run_resp.json()
    assert run_payload["run_id"] == artifact["run_id"]
    assert "telemetry" in run_payload

    list_resp = client.get("/api/runs")
    assert list_resp.status_code == 200
    listed = list_resp.json()["runs"]
    assert any(item["run_id"] == artifact["run_id"] for item in listed)
