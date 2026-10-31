"""Rotated photos: PaddleOCR turns the page upright before reading it, and its boxes refer to that upright
image. The engine must save the upright image as the page, or every highlight lands in the wrong place.
Runs without PaddleOCR installed: the OCR object is replaced by a stand-in that returns PaddleOCR's shapes.
"""

import numpy as np
from PIL import Image

from p3docintel.ocr import PaddleEngine


class FakePaddle:
    def __init__(self, upright: np.ndarray, angle: int):
        self.upright, self.angle = upright, angle

    def predict(self, path):
        h, w = self.upright.shape[:2]
        return [{"rec_texts": ["DL No SS-1"], "rec_scores": [0.97], "rec_boxes": [[w * 0.1, h * 0.2, w * 0.5, h * 0.3]],
                 "doc_preprocessor_res": {"angle": self.angle, "output_img": self.upright[:, :, ::-1]}}]  # BGR


def engine_with(fake) -> PaddleEngine:
    engine = object.__new__(PaddleEngine)  # skip __init__: no PaddleOCR needed
    engine.ocr = fake
    return engine


def test_rotated_page_is_saved_upright_and_boxes_match_it(tmp_path):
    upright = np.zeros((200, 400, 3), dtype=np.uint8)
    upright[:, :200] = (255, 0, 0)                               # left half red, so orientation is checkable
    sideways = tmp_path / "page-1.png"
    Image.fromarray(np.rot90(upright)).save(sideways)            # what the user uploaded: 400 tall, 200 wide

    [line] = engine_with(FakePaddle(upright, angle=270)).read(sideways, 1)

    saved = np.asarray(Image.open(sideways).convert("RGB"))
    assert saved.shape == (200, 400, 3)                          # the page on disk is now upright
    assert tuple(saved[100, 50]) == (255, 0, 0) and tuple(saved[100, 350]) == (0, 0, 0)  # and not mirrored
    assert line.bbox == (0.1, 0.2, 0.5, 0.3)                     # normalized against the upright size


def test_upright_page_is_left_alone(tmp_path):
    page = tmp_path / "page-1.png"
    Image.new("RGB", (300, 100), "white").save(page)
    before = page.read_bytes()
    engine_with(FakePaddle(np.zeros((100, 300, 3), dtype=np.uint8), angle=0)).read(page, 1)
    assert page.read_bytes() == before
