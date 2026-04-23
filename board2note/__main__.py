import argparse
import sys
import time

from .pipeline import process_path
from .encode import _GTP_COLS


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="board2note",
        description="Detect a Go board position from an image and output SGF/GTP/ASCII.",
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--size", type=int, choices=[9, 13, 19], default=None,
        help="Board size hint (auto-detected if omitted)",
    )
    parser.add_argument(
        "--out", choices=["sgf", "gtp", "ascii"], default="sgf",
        help="Output format (default: sgf)",
    )
    parser.add_argument("--model", default=None, help="Path to ONNX classifier model")
    parser.add_argument(
        "--conf-threshold", type=float, default=0.6,
        help="Confidence threshold for warnings (default: 0.6)",
    )
    args = parser.parse_args()

    model = None
    if args.model:
        try:
            import onnxruntime as ort
            model = ort.InferenceSession(args.model)
        except ImportError:
            print("Warning: onnxruntime not installed; using HSV classifier.", file=sys.stderr)

    t0 = time.perf_counter()
    try:
        result = process_path(
            args.image,
            hint_size=args.size,
            model=model,
            conf_threshold=args.conf_threshold,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    elapsed = time.perf_counter() - t0

    if args.out == "ascii":
        print(result.ascii)
    elif args.out == "gtp":
        print(result.gtp)
    else:
        print(result.sgf)

    if result.low_confidence:
        print("\nWarnings — low-confidence intersections:", file=sys.stderr)
        for row, col, conf in result.low_confidence:
            coord = f"{_GTP_COLS[col]}{result.board_size - row}"
            print(f"  {coord}: {conf:.2f}", file=sys.stderr)

    print(f"[{elapsed:.2f}s, {result.board_size}×{result.board_size}]", file=sys.stderr)


if __name__ == "__main__":
    main()
