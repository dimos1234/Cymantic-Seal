"""Cymatic Seal CLI — seal and verify audio from the command line."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cymatic-seal",
        description="Cymatic Seal — adversarial audio protection for creators.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── seal ──────────────────────────────────────────────────────
    seal_p = sub.add_parser("seal", help="Apply adversarial seal to an audio file.")
    seal_p.add_argument("input", help="Path to the input audio file.")
    seal_p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path for the sealed audio (default: <input>_sealed.<ext>).",
    )
    seal_p.add_argument(
        "-c",
        "--certificate",
        default=None,
        help="Output path for the JSON certificate (default: <output>.cert.json).",
    )
    seal_p.add_argument("--artist", default="", help="Artist name for the certificate.")
    seal_p.add_argument("--title", default="", help="Track title for the certificate.")
    seal_p.add_argument(
        "--method",
        choices=["fgsm", "ifgsm"],
        default="ifgsm",
        help="Attack method (default: ifgsm).",
    )
    seal_p.add_argument(
        "--steps", type=int, default=5, help="Number of I-FGSM steps (default: 5)."
    )
    seal_p.add_argument(
        "--epsilon",
        type=float,
        default=0.008,
        help="Max perturbation amplitude (default: 0.008).",
    )
    seal_p.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto | cpu | cuda (default: auto).",
    )
    seal_p.add_argument(
        "--model",
        default="htdemucs",
        help="Demucs model name (default: htdemucs).",
    )
    seal_p.add_argument(
        "--targets",
        nargs="+",
        default=["vocals"],
        help="Source stems to target (default: vocals).",
    )
    seal_p.add_argument(
        "--margin-db",
        type=float,
        default=-4.0,
        help="Safety margin below masking threshold in dB (default: -4.0).",
    )

    # ── verify ────────────────────────────────────────────────────
    ver_p = sub.add_parser("verify", help="Verify a sealed audio file against its certificate.")
    ver_p.add_argument("audio", help="Path to the sealed audio file.")
    ver_p.add_argument("certificate", help="Path to the .cert.json certificate file.")

    return parser


def _cmd_seal(args: argparse.Namespace) -> int:
    from cymatic_seal.seal import seal_audio

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(
            f"{input_path.stem}_sealed{input_path.suffix}"
        )

    if args.certificate:
        cert_path = Path(args.certificate)
    else:
        cert_path = output_path.with_suffix(output_path.suffix + ".cert.json")

    print(f"Sealing: {input_path}")
    print(f"  Method : {args.method} ({args.steps} steps)")
    print(f"  Device : {args.device}")
    print(f"  Model  : {args.model}")
    print(f"  Targets: {args.targets}")

    _, cert = seal_audio(
        input_path,
        output_path=output_path,
        certificate_path=cert_path,
        artist=args.artist,
        title=args.title,
        method=args.method,
        steps=args.steps,
        epsilon=args.epsilon,
        device=args.device,
        model_name=args.model,
        target_sources=args.targets,
        margin_db=args.margin_db,
    )

    print(f"\nSealed audio : {output_path}")
    print(f"Certificate  : {cert_path}")
    print(f"Duration     : {cert.duration_seconds:.2f}s")
    print(f"Perturbation : mean={cert.perturbation_stats['mean_abs']:.6f}, "
          f"max={cert.perturbation_stats['max_abs']:.6f}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from cymatic_seal.verify import verify_seal

    audio_path = Path(args.audio)
    cert_path = Path(args.certificate)

    if not audio_path.exists():
        print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
        return 1
    if not cert_path.exists():
        print(f"Error: certificate not found: {cert_path}", file=sys.stderr)
        return 1

    result = verify_seal(audio_path, cert_path)

    if result.verified:
        print("VERIFIED — this audio file matches its Cymatic Seal certificate.")
    else:
        print(f"NOT VERIFIED — {result.reason}")

    if result.certificate:
        c = result.certificate
        print(f"\n  Algorithm : {c.algorithm}")
        print(f"  Sealed at : {c.timestamp}")
        print(f"  Artist    : {c.artist or '(not set)'}")
        print(f"  Title     : {c.title or '(not set)'}")
        print(f"  Duration  : {c.duration_seconds:.2f}s")

    return 0 if result.verified else 2


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "seal":
        return _cmd_seal(args)
    if args.command == "verify":
        return _cmd_verify(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
