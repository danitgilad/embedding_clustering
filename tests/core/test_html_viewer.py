from PIL import Image
from src.core.html_viewer import image_to_data_uri

def test_image_to_data_uri_downscales_and_encodes(tmp_path):
    p = tmp_path / "x.png"
    Image.new("RGB", (512, 256), (200, 10, 10)).save(p)
    uri = image_to_data_uri(p, max_px=96)
    assert uri.startswith("data:image/png;base64,")
    import base64, io
    raw = base64.b64decode(uri.split(",", 1)[1])
    assert max(Image.open(io.BytesIO(raw)).size) == 96

def test_image_to_data_uri_missing_returns_empty(tmp_path):
    assert image_to_data_uri(tmp_path / "nope.png") == ""
