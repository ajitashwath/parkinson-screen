from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from backend.inference import (
    ImageDecodeError,
    TremorModel,
    decode_image_base64,
    effective_grid,
    normalize_drop,
    occluded_batch,
    to_model_input,
)
from backend.main import create_app, screening_label
from backend.security import RateLimiter
from backend.settings import DISCLAIMER, Settings
from common.preprocess import PreprocessConfig, preprocess_image
from tremor.synthetic import generate_synthetic_spiral

API = "/api/v1"
DEVICE = "device-abcdef12"
OTHER_DEVICE = "device-zzzzzz99"
MODEL_DIR = Path(__file__).resolve().parent.parent / "backend" / "models"


def make_png(size: int = 300) -> bytes:
    image = Image.new("L", (size, size), 255)
    ImageDraw.Draw(image).arc((40, 40, size - 40, size - 40), 0, 300, fill=0, width=4)
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def settings_with(**overrides) -> Settings:
    values = {"database_url": "sqlite://", "rate_limit_per_minute": 0, **overrides}
    return Settings(**values)


def stroke_payload(tremor_hz: float | None = 6.0, amplitude: float = 5.0) -> list[dict]:
    points = generate_synthetic_spiral(
        duration_s=5.0, sample_rate_hz=100.0, tremor_freq_hz=tremor_hz,
        tremor_amplitude=amplitude, seed=4,
    )
    return [
        {"x": p.x, "y": p.y, "t_ms": p.t_ms, "pressure": p.pressure,
         "pointer_type": p.pointer_type, "pen_lift": p.pen_lift}
        for p in points
    ]


def session_body(**overrides) -> dict:
    body = {
        "device_id": DEVICE, "kind": "spiral", "label": "lower_concern", "probability": 0.31,
        "tremor_score": 0.4, "spiral_deviation": 0.08, "pointer_type": "pen",
        "model_version": "m@1", "features": {"tremor_band_ratio": 0.3},
    }
    return {**body, **overrides}


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(settings_with())) as test_client:
        yield test_client


@pytest.fixture
def empty_client(tmp_path):
    with TestClient(create_app(settings_with(model_dir=tmp_path))) as test_client:
        yield test_client


def test_health_reports_both_models_loaded(client) -> None:
    body = client.get(f"{API}/health").json()
    assert body["status"] == "ok"
    assert body["imageModelLoaded"] and body["tremorModelLoaded"]
    assert body["disclaimer"] == DISCLAIMER


def test_health_reports_missing_models_instead_of_failing(empty_client) -> None:
    response = empty_client.get(f"{API}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "unavailable"
    assert response.json()["imageModelLoaded"] is False


def test_predict_returns_503_when_the_image_model_is_missing(empty_client) -> None:
    response = empty_client.post(
        f"{API}/predict", json={"image_base64": b64(make_png()), "kind": "spiral"}
    )
    assert response.status_code == 503
    assert response.json()["disclaimer"] == DISCLAIMER


def test_predict_returns_probability_label_and_occlusion_map(client) -> None:
    response = client.post(f"{API}/predict", json={"image_base64": b64(make_png()), "kind": "wave"})
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["label"] in {"lower_concern", "higher_concern", "uncertain"}
    assert body["kind"] == "wave"
    assert body["disclaimer"] == DISCLAIMER
    occlusion = body["occlusion_map"]
    assert occlusion["method"] == "occlusion"
    assert "Grad-CAM" not in occlusion["description"]
    assert occlusion["grid_size"] == 8 and len(occlusion["values"]) == 8
    flat = [v for row in occlusion["values"] for v in row]
    assert all(0.0 <= v <= 1.0 for v in flat)


def test_predict_accepts_a_data_url(client) -> None:
    payload = "data:image/png;base64," + b64(make_png())
    response = client.post(f"{API}/predict", json={"image_base64": payload, "kind": "spiral"})
    assert response.status_code == 200


def test_predict_is_deterministic(client) -> None:
    body = {"image_base64": b64(make_png()), "kind": "spiral"}
    first = client.post(f"{API}/predict", json=body).json()
    second = client.post(f"{API}/predict", json=body).json()
    assert first["probability"] == second["probability"]
    assert first["occlusion_map"] == second["occlusion_map"]


def test_predict_rejects_invalid_base64(client) -> None:
    body = {"image_base64": "!!!not base64!!!!!", "kind": "spiral"}
    response = client.post(f"{API}/predict", json=body)
    assert response.status_code == 422
    assert response.json()["code"] == "http_422"


def test_predict_rejects_bytes_that_are_not_an_image(client) -> None:
    response = client.post(
        f"{API}/predict", json={"image_base64": b64(b"this is not a png at all"), "kind": "spiral"}
    )
    assert response.status_code == 422


def test_predict_rejects_an_unknown_kind(client) -> None:
    body = {"image_base64": b64(make_png()), "kind": "circle"}
    response = client.post(f"{API}/predict", json=body)
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


def test_predict_rejects_a_missing_field(client) -> None:
    assert client.post(f"{API}/predict", json={"kind": "spiral"}).status_code == 422


def test_predict_rejects_an_image_over_the_size_limit() -> None:
    with TestClient(create_app(settings_with(max_image_bytes=100))) as small:
        response = small.post(
            f"{API}/predict", json={"image_base64": b64(make_png()), "kind": "spiral"}
        )
    assert response.status_code == 422
    assert "larger" in response.json()["detail"]


def test_tremor_returns_features_score_and_contributions(client) -> None:
    response = client.post(
        f"{API}/tremor", json={"points": stroke_payload(), "pointer_type": "pen"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["features"]["tremor_band_ratio"] > 0
    assert 0.0 <= body["score"] <= 1.0
    assert 1 <= len(body["top_contributions"]) <= 5
    assert body["trained_on"] == "synthetic"
    assert "simulated" in body["provenance"]
    assert body["disclaimer"] == DISCLAIMER


def test_tremor_scores_a_strong_tremor_above_a_clean_spiral(client) -> None:
    def score(hz, amplitude):
        body = {"points": stroke_payload(hz, amplitude), "pointer_type": "pen"}
        return client.post(f"{API}/tremor", json=body).json()["score"]

    assert score(6.0, 8.0) > score(None, 0.0)


def test_tremor_returns_warnings_not_errors_for_short_input(client) -> None:
    points = [{"x": float(i), "y": 0.0, "t_ms": i * 10.0} for i in range(10)]
    response = client.post(f"{API}/tremor", json={"points": points, "pointer_type": "mouse"})
    assert response.status_code == 200
    assert response.json()["warnings"]


def test_tremor_accepts_an_empty_point_list(client) -> None:
    response = client.post(f"{API}/tremor", json={"points": [], "pointer_type": "mouse"})
    assert response.status_code == 200
    assert response.json()["features"] == {} and response.json()["score"] is None


def test_tremor_rejects_too_many_points() -> None:
    with TestClient(create_app(settings_with(max_points=5))) as small:
        response = small.post(f"{API}/tremor", json={"points": stroke_payload()[:6]})
    assert response.status_code == 422


def test_tremor_rejects_out_of_range_values(client) -> None:
    bad_pressure = [{"x": 0, "y": 0, "t_ms": 0, "pressure": 5.0}]
    assert client.post(f"{API}/tremor", json={"points": bad_pressure}).status_code == 422
    negative_time = [{"x": 0, "y": 0, "t_ms": -1}]
    assert client.post(f"{API}/tremor", json={"points": negative_time}).status_code == 422
    huge = [{"x": 1e12, "y": 0, "t_ms": 0}]
    assert client.post(f"{API}/tremor", json={"points": huge}).status_code == 422


def test_tremor_works_without_the_tremor_model(empty_client) -> None:
    response = empty_client.post(f"{API}/tremor", json={"points": stroke_payload()})
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is False and body["score"] is None
    assert body["features"]


def test_session_create_list_and_delete(client) -> None:
    created = client.post(f"{API}/sessions", json=session_body())
    assert created.status_code == 201
    assert created.json()["probability"] == 0.31

    listed = client.get(f"{API}/sessions", params={"device_id": DEVICE}).json()
    assert [s["kind"] for s in listed["sessions"]] == ["spiral"]
    assert listed["sessions"][0]["pointerType"] == "pen"
    assert listed["disclaimer"] == DISCLAIMER

    deleted = client.delete(f"{API}/sessions", params={"device_id": DEVICE}).json()
    assert deleted["deleted"] == 1
    assert client.get(f"{API}/sessions", params={"device_id": DEVICE}).json()["sessions"] == []


def test_sessions_are_isolated_per_device(client) -> None:
    client.post(f"{API}/sessions", json=session_body(device_id=DEVICE))
    client.post(f"{API}/sessions", json=session_body(device_id=OTHER_DEVICE))
    client.delete(f"{API}/sessions", params={"device_id": DEVICE})
    remaining = client.get(f"{API}/sessions", params={"device_id": OTHER_DEVICE}).json()
    assert len(remaining["sessions"]) == 1
    client.delete(f"{API}/sessions", params={"device_id": OTHER_DEVICE})


def test_sessions_reject_bad_input(client) -> None:
    assert client.post(f"{API}/sessions", json=session_body(device_id="short")).status_code == 422
    assert client.post(f"{API}/sessions", json=session_body(probability=1.5)).status_code == 422
    assert client.post(f"{API}/sessions", json=session_body(label="sick")).status_code == 422
    assert client.get(f"{API}/sessions", params={"device_id": "../etc"}).status_code == 422
    assert client.get(f"{API}/sessions").status_code == 422
    assert client.delete(f"{API}/sessions").status_code == 422


def test_sessions_store_features_and_scores_only() -> None:
    from backend.database import SessionRow

    columns = {c.name for c in SessionRow.__table__.columns}
    assert not columns & {"image", "image_base64", "points", "strokes", "drawing"}


def test_sessions_are_capped_per_device() -> None:
    with TestClient(create_app(settings_with(max_sessions_per_device=3))) as capped:
        for _ in range(5):
            capped.post(f"{API}/sessions", json=session_body())
        listed = capped.get(f"{API}/sessions", params={"device_id": DEVICE}).json()
    assert len(listed["sessions"]) == 3


def test_errors_use_one_format_and_carry_the_disclaimer(client) -> None:
    for response in (
        client.get(f"{API}/nope"),
        client.post(f"{API}/predict", json={}),
    ):
        body = response.json()
        assert set(body) >= {"code", "detail", "disclaimer"}
        assert body["disclaimer"] == DISCLAIMER


def test_oversized_request_is_rejected_before_parsing() -> None:
    with TestClient(create_app(settings_with(max_request_bytes=1000))) as small:
        response = small.post(f"{API}/tremor", content=b"x" * 5000,
                              headers={"content-type": "application/json"})
    assert response.status_code == 413
    assert response.json()["code"] == "request_too_large"


def test_rate_limit_returns_429_after_the_limit() -> None:
    with TestClient(create_app(settings_with(rate_limit_per_minute=2))) as limited:
        codes = [limited.get(f"{API}/sessions", params={"device_id": DEVICE}).status_code
                 for _ in range(3)]
        assert limited.get(f"{API}/health").status_code == 200
    assert codes == [200, 200, 429]


def test_rate_limiter_window_expires() -> None:
    limiter = RateLimiter(1)
    assert limiter.allow("a", now=0.0)
    assert not limiter.allow("a", now=1.0)
    assert limiter.allow("a", now=61.0)
    assert limiter.allow("b", now=1.0)


def test_cors_allows_only_configured_origins() -> None:
    with TestClient(create_app(settings_with(allowed_origins="https://app.example"))) as cors:
        allowed = cors.get(f"{API}/health", headers={"Origin": "https://app.example"})
        denied = cors.get(f"{API}/health", headers={"Origin": "https://evil.example"})
    assert allowed.headers.get("access-control-allow-origin") == "https://app.example"
    assert "access-control-allow-origin" not in denied.headers


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.1, "lower_concern"), (0.9, "higher_concern"), (0.52, "uncertain"), (0.48, "uncertain")],
)
def test_screening_label_uses_the_uncertain_margin(probability: float, expected: str) -> None:
    assert screening_label(probability, 0.5, 0.05) == expected


def test_effective_grid_respects_the_pass_cap() -> None:
    assert effective_grid(8, 64) == 7
    assert effective_grid(8, 65) == 8
    assert effective_grid(32, 10) == 3
    assert effective_grid(1, 1) == 1


def test_occluded_batch_covers_each_cell_once() -> None:
    image = np.zeros((1, 3, 224, 224), dtype=np.float32)
    batch = occluded_batch(image, 4)
    assert batch.shape == (16, 3, 224, 224)
    assert all(int((b[0] == 1.0).sum()) == 56 * 56 for b in batch)
    assert batch.sum() == 16 * 3 * 56 * 56


def test_normalize_drop_scales_positive_drops_and_zeroes_flat_maps() -> None:
    scaled = normalize_drop(np.array([[0.2, -0.1], [0.1, 0.0]]))
    assert scaled.max() == 1.0 and scaled.min() == 0.0
    assert not normalize_drop(np.zeros((2, 2))).any()


def test_decode_rejects_bad_base64_and_oversize() -> None:
    with pytest.raises(ImageDecodeError):
        decode_image_base64("@@@@", 1000)
    with pytest.raises(ImageDecodeError):
        decode_image_base64(b64(b"x" * 50), 10)


def test_serving_preprocessing_matches_training_preprocessing(tmp_path: Path) -> None:
    png = make_png(257)
    path = tmp_path / "drawing.png"
    path.write_bytes(png)
    training_tensor = np.transpose(
        preprocess_image(path, PreprocessConfig()), (2, 0, 1)
    )[None].astype(np.float32)
    serving_tensor = to_model_input(png)
    assert serving_tensor.shape == (1, 3, 224, 224)
    assert np.array_equal(training_tensor, serving_tensor)


def test_tremor_contributions_sum_to_the_raw_score() -> None:
    card = json.loads((MODEL_DIR / "tremor_model_card.json").read_text(encoding="utf-8"))
    model = TremorModel(MODEL_DIR / card["booster_file"], card)
    from tremor.features import compute_features

    spiral = generate_synthetic_spiral(tremor_freq_hz=6.0, tremor_amplitude=4.0)
    features = compute_features(spiral).features
    row = model.vector(features)
    contributions = np.asarray(model.booster.predict(row, pred_contrib=True))[0]
    raw = float(model.booster.predict(row, raw_score=True)[0])
    assert abs(contributions.sum() - raw) < 1e-9
