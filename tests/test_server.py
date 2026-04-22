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

    assert set(structured) == {
        "executive_summary",
        "action_plan",
        "risks",
        "winner_reason",
        "alternatives",
    }
    assert isinstance(structured["action_plan"], list)
    assert isinstance(structured["risks"], list)
    assert isinstance(structured["alternatives"], list)
    assert results["problem"] == config["problem"]
    assert results["event_count"] > 0
    assert "war_started" in seen_types
    assert "generation_summary" in seen_types

    run_dir = Path(artifact["dir"])
    json_path = Path(artifact["json_path"])
    events_path = Path(artifact["events_path"])
    report_path = Path(artifact["report_path"])

    assert run_dir.is_dir()
    assert json_path.is_file()
    assert events_path.is_file()
    assert report_path.is_file()
    assert str(run_dir).startswith(str(tmp_path))
    assert artifact["run_id"].startswith("mock-")
