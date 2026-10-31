"""PaddlePaddle 3.x's oneDNN CPU acceleration raises NotImplementedError on some machines (often Windows).
The engine must default it off on Windows and recover by retrying without it anywhere else."""

import numpy as np
from PIL import Image

from p3docintel.config import Settings
from p3docintel.ocr import PaddleEngine

OK = {"rec_texts": ["Due date: 25/10/2026"], "rec_scores": [0.95], "rec_boxes": [[10, 20, 110, 30]],
      "doc_preprocessor_res": {"angle": 0, "output_img": np.zeros((100, 200, 3), dtype=np.uint8)}}


class Paddle:
    def __init__(self, mkldnn, fails_with_mkldnn):
        self.mkldnn, self.fails = mkldnn, fails_with_mkldnn

    def predict(self, path):
        if self.mkldnn and self.fails:
            raise NotImplementedError("(Unimplemented) ConvertPirAttribute2RuntimeAttribute not support")
        return [OK]


def engine(mkldnn: bool, fails_with_mkldnn: bool) -> tuple[PaddleEngine, list]:
    built = []
    e = object.__new__(PaddleEngine)  # skip __init__: no PaddleOCR needed

    def make(m):
        built.append(m)
        return Paddle(m, fails_with_mkldnn)

    e._make, e.mkldnn = make, mkldnn
    e.ocr = make(mkldnn)
    return e, built


def page(tmp_path):
    p = tmp_path / "page-1.png"
    Image.new("RGB", (200, 100), "white").save(p)
    return p


def test_retries_without_onednn_when_it_fails(tmp_path):
    e, built = engine(mkldnn=True, fails_with_mkldnn=True)
    [line] = e.read(page(tmp_path), 1)
    assert line.text == "Due date: 25/10/2026"
    assert built == [True, False] and e.mkldnn is False  # rebuilt once, and stays off for later pages
    e.read(page(tmp_path), 2)
    assert built == [True, False]


def test_no_retry_when_already_off(tmp_path):
    e, built = engine(mkldnn=False, fails_with_mkldnn=True)
    assert len(e.read(page(tmp_path), 1)) == 1 and built == [False]


def test_default_is_off_on_windows_only(monkeypatch):
    s = Settings(_env_file=None)
    monkeypatch.setattr("sys.platform", "win32")
    assert s.resolved_ocr_mkldnn() is False
    monkeypatch.setattr("sys.platform", "darwin")
    assert s.resolved_ocr_mkldnn() is True
    assert Settings(ocr_mkldnn=False, _env_file=None).resolved_ocr_mkldnn() is False
