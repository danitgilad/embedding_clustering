from src.part_b.generate import generate_faces


def test_generate_dedups_and_stops_at_n(tmp_path):
    blobs = [b"AAAA", b"AAAA", b"BBBB", b"CCCC"]
    calls = {"i": 0}
    def fake_fetch(url):
        b = blobs[min(calls["i"], len(blobs) - 1)]; calls["i"] += 1
        return b
    saved = generate_faces(n=3, url="http://x", out_dir=tmp_path, delay_s=0.0,
                           max_retries=2, fetch=fake_fetch)
    assert len(saved) == 3
    assert len(set(p.read_bytes() for p in saved)) == 3

def test_generate_retries_on_error(tmp_path):
    seq = [RuntimeError("net"), b"AAAA"]
    def flaky_fetch(url):
        x = seq.pop(0)
        if isinstance(x, Exception):
            raise x
        return x
    saved = generate_faces(n=1, url="http://x", out_dir=tmp_path, delay_s=0.0,
                           max_retries=3, fetch=flaky_fetch)
    assert len(saved) == 1
