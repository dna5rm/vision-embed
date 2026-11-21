"""Regression test for the WD14 tagger preprocessing (v12 fix).

The previous RealTagger.tag() divided the pixel array by 255 and fed the
model a stretched, non-padded, RGB-ordered blob. The reference WD14
preprocessing (SmilingWolf/wd-tagger Predictor.prepare_image) requires:
pad to square with WHITE fill, BICUBIC resize, RAW 0-255 float32 values,
RGB->BGR channel reversal, NHWC layout. This test builds a fake ONNX
session that records the blob it receives and asserts the corrected
behavior — no 1.26 GB model file needed.
"""

import numpy as np
import pytest

from vision_embed.adapters.real_describe import RealTagger


class FakeInput:
    def __init__(self) -> None:
        self.name = "input"


class FakeSession:
    """Stand-in for onnxruntime.InferenceSession that records the blob."""

    def __init__(self, scores: np.ndarray) -> None:
        self._scores = scores
        self.blobs: list[np.ndarray] = []
        self._input = FakeInput()

    def get_inputs(self) -> list[FakeInput]:
        return [self._input]

    def run(self, _out, feed) -> list[np.ndarray]:
        self.blobs.append(feed[self._input.name])
        # scores shaped (1, num_tags) to mirror the real session output
        return [self._scores[None, ...]]


def make_tagger() -> tuple[RealTagger, FakeSession]:
    """RealTagger with a fake session + fake names + fixed input size.

    Skips load() (which requires a real ONNX file); the CUDA gate and
    names-length check stay untouched in production code and exercise the
    real .load() elsewhere.
    """
    session = FakeSession(np.array([0.99, 0.40, 0.36, 0.01], dtype=np.float32))
    tagger = RealTagger.__new__(RealTagger)
    tagger._provider = "CPUExecutionProvider"
    tagger._names_path = None  # not used once .load() is bypassed
    tagger._session = session
    tagger._names = ["general", "one", "two", "three"]
    tagger._input_size = (448, 448)
    return tagger, session


def run_tag(tagger, session, rgb):
    tagger.tag(rgb, threshold=0.35)
    assert len(session.blobs) == 1
    return session.blobs[0]


def test_blob_shape_dtype_and_no_255_division():
    """Shape (1,448,448,3) float32; max value ~255, not ~1.0 (/255 bug)."""
    tagger, session = make_tagger()
    rgb = np.zeros((32, 64, 3), dtype=np.uint8)
    b = run_tag(tagger, session, rgb)
    assert b.dtype == np.float32
    assert b.shape == (1, 448, 448, 3)
    assert b.max() > 250.0, f"max={b.max()} — looks like a /255 regression"
    assert b.max() <= 255.0


def test_channels_are_bgr_ordered():
    """RGB left half red -> BGR left half blue in channel 0; right half
    green -> BGR right half green in channel 0. Reversal is relative to
    the input RGB."""
    tagger, session = make_tagger()
    rgb = np.zeros((32, 64, 3), dtype=np.uint8)
    rgb[:, :32] = (255, 0, 0)  # red left half
    rgb[:, 32:] = (0, 0, 255)  # blue right half (green is RGB/BGR-symmetric)
    b = run_tag(tagger, session, rgb)[0]
    # inside the content region (vertically centered band)
    row = 224  # center row of the padded square
    # left content half: RGB red -> BGR = (0,0,255): channel 2 high
    left = b[row, 100]
    assert left[2] > 250 and left[0] < 5 and left[1] < 5
    # right content half: RGB blue -> BGR = (255,0,0): channel 0 high
    right = b[row, 400]
    assert right[0] > 250 and right[1] < 5 and right[2] < 5


def test_padded_to_square_white_fill_not_stretched():
    """A 64x32 image is padded to 448x448 with white, centered vertically:
    pad band rows are pure 255 and the content is centered, not stretched
    to full height."""
    tagger, session = make_tagger()
    rgb = np.zeros((32, 64, 3), dtype=np.uint8)
    rgb[:, :32] = (255, 0, 0)
    rgb[:, 32:] = (0, 0, 255)  # blue right half
    b = run_tag(tagger, session, rgb)[0]
    # aspect-preserved content band: 448x(32/64*448)=448x224, centered
    # vertically -> rows 112..335 are content, rows 0..111 and 336..447 pad
    for probe_row in (0, 60, 100, 350, 380, 447):  # >1px off the BICUBIC edge ring
        assert np.all(b[probe_row] == 255.0), (
            f"row {probe_row} not white pad: {b[probe_row]}"
        )
    # content actually starts at the top pad edge (some antialiasing on
    # edge rows is fine — probe the interior of each half)
    content_top, content_bottom = b[120], b[325]
    for band in (content_top, content_bottom):
        left = band[100]
        right = band[400]
        assert left[2] > 250 and right[0] > 250  # red->BGR blue-ch, blue->BGR red-ch
    # centered: pad bands above and below have equal height
    assert np.all(b[5] == 255.0) and np.all(b[442] == 255.0)


def test_tag_contract_unchanged():
    """Return type/contract: list[(tag, float)] cut at the caller threshold,
    sorted score-descending."""
    tagger, session = make_tagger()
    rgb = np.zeros((32, 64, 3), dtype=np.uint8)
    rgb[0, 0] = (255, 255, 255)
    out = tagger.tag(rgb, threshold=0.35)
    assert out == [
        ("general", pytest.approx(0.99)),
        ("one", pytest.approx(0.40)),
        ("two", pytest.approx(0.36)),
    ]
    assert all(isinstance(t, str) and isinstance(s, float) for t, s in out)
