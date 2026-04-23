"""Layer 4 — Notation Encoder.

Input:  N×N flat label list (row-major, top-to-bottom) + board size
Output: SGF string, GTP setup commands, or ASCII diagram
"""

from typing import Literal
from .classify import Stone

# SGF uses 'a'–'s' for both axes (col left→right, row top→bottom).
_SGF_CHARS = "abcdefghijklmnopqrs"

# GTP uses A–T (no I) for columns and 1–19 for rows (1 at bottom).
_GTP_COLS = "ABCDEFGHJKLMNOPQRST"


def _sgf_coord(col: int, row: int) -> str:
    return _SGF_CHARS[col] + _SGF_CHARS[row]


def _gtp_coord(col: int, row: int, board_size: int) -> str:
    return f"{_GTP_COLS[col]}{board_size - row}"


def label_grid_to_sgf(labels: list[Stone], board_size: int) -> str:
    """Full SGF string with AB/AW setup properties."""
    black: list[str] = []
    white: list[str] = []
    for i, label in enumerate(labels):
        row, col = divmod(i, board_size)
        if label == "black":
            black.append(_sgf_coord(col, row))
        elif label == "white":
            white.append(_sgf_coord(col, row))

    parts = [f"(;FF[4]GM[1]SZ[{board_size}]"]
    if black:
        parts.append("AB" + "".join(f"[{c}]" for c in black))
    if white:
        parts.append("AW" + "".join(f"[{c}]" for c in white))
    parts.append(")")
    return "".join(parts)


def label_grid_to_gtp(labels: list[Stone], board_size: int) -> str:
    """GTP commands to set up the position from scratch."""
    lines = [f"boardsize {board_size}", "clear_board"]
    for i, label in enumerate(labels):
        if label == "empty":
            continue
        row, col = divmod(i, board_size)
        color = "black" if label == "black" else "white"
        lines.append(f"play {color} {_gtp_coord(col, row, board_size)}")
    return "\n".join(lines)


def label_grid_to_ascii(labels: list[Stone], board_size: int) -> str:
    """Human-readable ASCII board diagram."""
    header = "   " + " ".join(_GTP_COLS[c] for c in range(board_size))
    rows = [header]
    for r in range(board_size):
        row_label = f"{board_size - r:2d} "
        cells = []
        for c in range(board_size):
            stone = labels[r * board_size + c]
            cells.append("X" if stone == "black" else "O" if stone == "white" else ".")
        rows.append(row_label + " ".join(cells))
    return "\n".join(rows)


def encode(
    labels: list[Stone],
    board_size: int,
    fmt: Literal["sgf", "gtp", "ascii"] = "sgf",
) -> str:
    if fmt == "gtp":
        return label_grid_to_gtp(labels, board_size)
    if fmt == "ascii":
        return label_grid_to_ascii(labels, board_size)
    return label_grid_to_sgf(labels, board_size)
