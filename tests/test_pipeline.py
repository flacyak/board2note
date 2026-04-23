"""Integration tests: synthetic board image → position encoding."""

import re
import pytest
from board2note.pipeline import process_image
from .conftest import make_board_image


class TestPipeline:
    def test_empty_9x9_returns_result(self):
        img = make_board_image(board_size=9)
        result = process_image(img, hint_size=9)
        assert result.board_size == 9
        assert len(result.labels) == 81
        assert all(l in ("black", "white", "empty") for l in result.labels)

    def test_empty_board_has_no_stones_in_sgf(self):
        img = make_board_image(board_size=9)
        result = process_image(img, hint_size=9)
        assert "AB" not in result.sgf
        assert "AW" not in result.sgf
        assert "SZ[9]" in result.sgf

    def test_stones_detected(self):
        img = make_board_image(
            board_size=9,
            black_stones=[(2, 2), (4, 4)],
            white_stones=[(6, 6)],
        )
        result = process_image(img, hint_size=9)
        # We expect some stones — exact positions depend on localisation accuracy.
        stone_count = sum(1 for l in result.labels if l != "empty")
        assert stone_count > 0

    def test_sgf_valid_format(self):
        img = make_board_image(board_size=9)
        result = process_image(img, hint_size=9)
        assert result.sgf.startswith("(;FF[4]")
        assert result.sgf.endswith(")")

    def test_gtp_has_boardsize(self):
        img = make_board_image(board_size=9)
        result = process_image(img, hint_size=9)
        assert "boardsize 9" in result.gtp

    def test_ascii_row_count(self):
        img = make_board_image(board_size=9)
        result = process_image(img, hint_size=9)
        lines = result.ascii.splitlines()
        assert len(lines) == 10  # header + 9 data rows

    def test_low_confidence_list_type(self):
        img = make_board_image(board_size=9)
        result = process_image(img, hint_size=9)
        assert isinstance(result.low_confidence, list)
        for row, col, conf in result.low_confidence:
            assert 0 <= row < 9
            assert 0 <= col < 9
            assert 0.0 <= conf < 0.6
