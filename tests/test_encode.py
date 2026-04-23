"""Tests for Layer 4 — Notation Encoder (no CV required)."""

import re
import pytest
from board2note.encode import (
    label_grid_to_sgf,
    label_grid_to_gtp,
    label_grid_to_ascii,
)


def _flat(board_size, **stones):
    """Build a flat label list. stones: e.g. black=[(0,0)], white=[(1,1)]."""
    grid = ["empty"] * (board_size * board_size)
    for row, col in stones.get("black", []):
        grid[row * board_size + col] = "black"
    for row, col in stones.get("white", []):
        grid[row * board_size + col] = "white"
    return grid


class TestSGF:
    def test_empty_board(self):
        sgf = label_grid_to_sgf(["empty"] * 81, 9)
        assert "SZ[9]" in sgf
        assert "AB" not in sgf
        assert "AW" not in sgf

    def test_single_black_stone(self):
        labels = _flat(9, black=[(0, 0)])
        sgf = label_grid_to_sgf(labels, 9)
        assert "AB[aa]" in sgf

    def test_single_white_stone(self):
        labels = _flat(9, white=[(0, 0)])
        sgf = label_grid_to_sgf(labels, 9)
        assert "AW[aa]" in sgf

    def test_bottom_right_corner_19x19(self):
        labels = _flat(19, black=[(18, 18)])
        sgf = label_grid_to_sgf(labels, 19)
        assert "AB[ss]" in sgf

    def test_stone_count(self):
        labels = _flat(9, black=[(0, 0), (1, 1)], white=[(2, 2)])
        sgf = label_grid_to_sgf(labels, 9)
        # two-char lowercase coords match stone positions
        coords = re.findall(r"\[[a-s]{2}\]", sgf)
        assert len(coords) == 3  # 2 black + 1 white

    def test_valid_parens(self):
        sgf = label_grid_to_sgf(["empty"] * 81, 9)
        assert sgf.startswith("(") and sgf.endswith(")")


class TestGTP:
    def test_empty_board(self):
        gtp = label_grid_to_gtp(["empty"] * 81, 9)
        assert "boardsize 9" in gtp
        assert "clear_board" in gtp
        assert "play" not in gtp

    def test_black_top_left(self):
        labels = _flat(9, black=[(0, 0)])
        gtp = label_grid_to_gtp(labels, 9)
        assert "play black A9" in gtp

    def test_white_bottom_right(self):
        labels = _flat(9, white=[(8, 8)])
        gtp = label_grid_to_gtp(labels, 9)
        assert "play white J1" in gtp

    def test_no_i_column(self):
        labels = _flat(9, black=[(0, c) for c in range(9)])
        gtp = label_grid_to_gtp(labels, 9)
        assert " I" not in gtp


class TestASCII:
    def test_dimensions(self):
        ascii_board = label_grid_to_ascii(["empty"] * 81, 9)
        lines = ascii_board.splitlines()
        assert len(lines) == 10  # 1 header + 9 rows

    def test_stone_symbols(self):
        labels = _flat(9, black=[(0, 0)], white=[(8, 8)])
        ascii_board = label_grid_to_ascii(labels, 9)
        assert "X" in ascii_board
        assert "O" in ascii_board

    def test_row_numbers_descending(self):
        ascii_board = label_grid_to_ascii(["empty"] * 81, 9)
        lines = ascii_board.splitlines()
        assert lines[1].startswith(" 9")
        assert lines[-1].startswith(" 1")
