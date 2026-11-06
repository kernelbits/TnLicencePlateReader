"""API tests for backend/backend.py.

Everything external is mocked: the Roboflow detection model, the PaddleOCR HTTP
endpoint, the Ollama HTTP endpoint (chat and vision validation) and Supabase
(database and storage). No network, GPU or model weights are needed.
"""
import io
import re
from unittest import mock

import requests
from PIL import Image

OCR_URL = "http://ocr.test/layout-parsing"
OLLAMA_URL = "http://ollama.test/api/generate"

BOX = {"x": 320, "y": 320, "width": 200, "height": 80, "confidence": 0.9, "class": "plate"}


def _response(status=200, payload=None, text=""):
    resp = mock.MagicMock(name="Response")
    resp.status_code = status
    resp.text = text
    resp.json.return_value = payload if payload is not None else {}
    return resp


def _jpeg_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (1000, 800), color=(200, 200, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def _detection(backend_module, predictions):
    """Make the mocked Roboflow model return the given boxes."""
    prediction = mock.MagicMock(name="Prediction")
    prediction.json.return_value = {"predictions": predictions}
    backend_module.model.predict.return_value = prediction


def _route_http(ocr_payload=None, vision_text="1258365"):
    """Answer PaddleOCR and Ollama calls by URL; anything else fails the test."""

    def fake_post(url, *args, **kwargs):
        if url == OCR_URL:
            return _response(200, ocr_payload)
        if url == OLLAMA_URL:
            return _response(200, {"response": vision_text})
        raise AssertionError(f"unexpected HTTP call to {url}")

    return fake_post


def _ocr_text(text):
    return {"result": {"layoutParsingResults": [{"markdown": {"text": text}}]}}


def _post_image(client):
    return client.post("/detect", files={"image": ("plate.jpg", _jpeg_bytes(), "image/jpeg")})


# --- /detect ---------------------------------------------------------------

def test_detect_builds_plate_number_and_validates_with_vision(client, backend_module):
    _detection(backend_module, [BOX])
    with mock.patch.object(backend_module.requests, "post",
                           side_effect=_route_http(_ocr_text("125 8365"), vision_text="1258365")):
        res = _post_image(client)

    assert res.status_code == 200
    body = res.json()
    assert body["plate_number"] == "125تونس8365"
    assert body["driver_info"] is None  # no Supabase client configured
    assert body["image_url"] is None
    assert body["vision_validation"]["match"] is True
    assert body["vision_validation"]["message"] == "AI confirms OCR result."
    assert len(body["predictions"]) == 1


def test_detect_takes_first_three_and_last_four_digits(client, backend_module):
    _detection(backend_module, [BOX])
    with mock.patch.object(backend_module.requests, "post",
                           side_effect=_route_http(_ocr_text("ab 1 2 3 4 5 6 7 8 9 0 x"))):
        res = _post_image(client)

    # Digits are 1234567890: first three "123", last four "7890".
    assert res.json()["plate_number"] == "123تونس7890"


def test_detect_pads_short_ocr_digits_with_zeros(client, backend_module):
    # Documents current behaviour: a short OCR read is zero-padded silently.
    _detection(backend_module, [BOX])
    with mock.patch.object(backend_module.requests, "post",
                           side_effect=_route_http(_ocr_text("42"))):
        res = _post_image(client)

    assert res.json()["plate_number"] == "042تونس0042"


def test_detect_reports_disagreement_between_ocr_and_vision(client, backend_module):
    _detection(backend_module, [BOX])
    with mock.patch.object(backend_module.requests, "post",
                           side_effect=_route_http(_ocr_text("125 8365"), vision_text="1258356")):
        res = _post_image(client)

    validation = res.json()["vision_validation"]
    assert validation["match"] is False
    assert validation["message"] == "AI suggests different result: 1258356 vs OCR 1258365"


def test_detect_returns_404_when_no_plate_found(client, backend_module):
    _detection(backend_module, [])
    with mock.patch.object(backend_module.requests, "post") as post:
        res = _post_image(client)

    assert res.status_code == 404
    assert res.json() == {"error": "No plates detected"}
    post.assert_not_called()


def test_detect_returns_422_when_ocr_returns_no_layout_results(client, backend_module):
    _detection(backend_module, [BOX])
    with mock.patch.object(backend_module.requests, "post",
                           side_effect=_route_http({"result": {"layoutParsingResults": []}})):
        res = _post_image(client)

    assert res.status_code == 422
    assert res.json() == {"error": "No OCR text found"}


def test_detect_returns_422_when_ocr_text_is_empty(client, backend_module):
    _detection(backend_module, [BOX])
    with mock.patch.object(backend_module.requests, "post",
                           side_effect=_route_http(_ocr_text(""))):
        res = _post_image(client)

    assert res.status_code == 422
    assert res.json() == {"error": "No text found in OCR result"}


def test_detect_sends_ocr_request_with_token_header(client, backend_module):
    _detection(backend_module, [BOX])
    with mock.patch.object(backend_module.requests, "post",
                           side_effect=_route_http(_ocr_text("125 8365"))) as post:
        _post_image(client)

    ocr_calls = [c for c in post.call_args_list if c.args and c.args[0] == OCR_URL]
    assert len(ocr_calls) == 1
    assert ocr_calls[0].kwargs["headers"]["Authorization"] == "token test-token"
    assert ocr_calls[0].kwargs["json"]["fileType"] == 1


# --- /chat -----------------------------------------------------------------

def test_chat_returns_503_when_ollama_is_unreachable(client):
    with mock.patch.object(requests, "post",
                           side_effect=requests.exceptions.ConnectionError("refused")):
        res = client.post("/chat", json={"message": "hello"})

    assert res.status_code == 503
    assert "Ollama service is not running" in res.json()["error"]


def test_chat_returns_plain_answer_without_database(client):
    with mock.patch.object(requests, "post",
                           return_value=_response(200, {"response": "ACTION: ANSWER\nDATA: Hello there"})):
        res = client.post("/chat", json={"message": "hi"})

    assert res.status_code == 200
    assert res.json() == {"answer": "Hello there"}


def test_chat_runs_generated_query_and_summarises_rows(client, backend_module):
    rows = [{"driver_name": "Ali", "vehicle_make": "Ford"}]
    query = mock.MagicMock(name="query")
    query.ilike.return_value = query
    query.limit.return_value = query
    query.execute.return_value.data = rows
    supabase = mock.MagicMock(name="supabase")
    supabase.table.return_value.select.return_value = query
    backend_module.supabase = supabase

    query_spec = (
        'ACTION: QUERY\nDATA: {"table": "license_plates", "select": "*", '
        '"filters": [{"col": "vehicle_make", "op": "ilike", "val": "Ford"}], "limit": 5}'
    )
    responses = [_response(200, {"response": query_spec}),
                 _response(200, {"response": "Ali drives a Ford."})]

    with mock.patch.object(requests, "post", side_effect=responses):
        res = client.post("/chat", json={"message": "who drives a ford?"})

    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == "Ali drives a Ford."
    assert body["data"] == rows
    supabase.table.assert_called_once_with("license_plates")
    query.ilike.assert_called_once_with("vehicle_make", "%Ford%")
    query.limit.assert_called_once_with(5)


# --- helpers used by /detect -----------------------------------------------

def test_log_detection_strips_non_ascii_from_storage_filename(backend_module):
    storage = mock.MagicMock(name="storage")
    storage.from_.return_value.get_public_url.return_value = "https://cdn.test/plates/x.jpg"
    supabase = mock.MagicMock(name="supabase")
    supabase.storage = storage
    backend_module.supabase = supabase

    url = backend_module.log_detection("125تونس8365", b"jpeg-bytes")

    assert url == "https://cdn.test/plates/x.jpg"
    upload_kwargs = storage.from_.return_value.upload.call_args.kwargs
    assert re.fullmatch(r"detection_\d+_1258365\.jpg", upload_kwargs["path"])
    assert upload_kwargs["file_options"] == {"content-type": "image/jpeg"}
    supabase.table.assert_called_with("detection_logs")


def test_validate_with_vision_reports_ollama_error_status(backend_module):
    with mock.patch.object(requests, "post", return_value=_response(500, text="boom")):
        result = backend_module.validate_with_vision("aGVsbG8=", "125تونس8365")

    assert result == {"error": "Vision API error: 500"}
