from fastapi.testclient import TestClient

from mena_mlops.control_plane import app


def test_control_plane_health() -> None:
    response = TestClient(app).get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
