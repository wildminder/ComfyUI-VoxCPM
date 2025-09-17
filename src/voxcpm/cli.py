#!/usr/bin/env python3
"""
VoxCPM Command Line Interface

Unified CLI for voice cloning, direct TTS synthesis, and batch processing.

Usage examples:
    # Direct synthesis (single sample)
    voxcpm --text "Hello world" --output output.wav

    # Voice cloning (with reference audio and text)
    voxcpm --text "Hello world" --prompt-audio voice.wav --prompt-text "reference text" --output output.wav --denoise

    # Batch processing (each line in the file is one sample)
    voxcpm --input texts.txt --output-dir ./outputs/
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List
import soundfile as sf

from voxcpm.core import VoxCPM


def validate_file_exists(file_path: str, file_type: str = "file") -> Path:
    """Validate that a file exists."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{file_type} '{file_path}' does not exist")
    return path


def validate_output_path(output_path: str) -> Path:
    """Validate the output path and create parent directories if needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_model(args) -> VoxCPM:
    """Load VoxCPM model.

    Prefer --model-path if provided; otherwise use from_pretrained (Hub).
    """
    print("Loading VoxCPM model...")

    # 兼容旧参数：ZIPENHANCER_MODEL_PATH 环境变量作为默认
    zipenhancer_path = getattr(args, "zipenhancer_path", None) or os.environ.get(
        "ZIPENHANCER_MODEL_PATH", None
    )

    # Load from local path if provided
    if getattr(args, "model_path", None):
        try:
            model = VoxCPM(
                voxcpm_model_path=args.model_path,
                zipenhancer_model_path=zipenhancer_path,
                enable_denoiser=not getattr(args, "no_denoiser", False),
            )
            print("Model loaded (local).")
            return model
        except Exception as e:
            print(f"Failed to load model (local): {e}")
            sys.exit(1)

    # Otherwise, try from_pretrained (Hub); exit on failure
    try:
        model = VoxCPM.from_pretrained(
            hf_model_id=getattr(args, "hf_model_id", "openbmb/VoxCPM-0.5B"),
            load_denoiser=not getattr(args, "no_denoiser", False),
            zipenhancer_model_id=zipenhancer_path,
            cache_dir=getattr(args, "cache_dir", None),
            local_files_only=getattr(args, "local_files_only", False),
        )
        print("Model loaded (from_pretrained).")
        return model
    except Exception as e:
        print(f"Failed to load model (from_pretrained): {e}")
        sys.exit(1)


def cmd_clone(args):
    """Voice cloning command."""
    # Validate inputs
    if not args.text:
        print("Error: Please provide text to synthesize (--text)")
        sys.exit(1)
    
    if not args.prompt_audio:
        print("Error: Voice cloning requires a reference audio (--prompt-audio)")
        sys.exit(1)
        
    if not args.prompt_text:
        print("Error: Voice cloning requires a reference text (--prompt-text)")
        sys.exit(1)
    
    # Validate files
    prompt_audio_path = validate_file_exists(args.prompt_audio, "reference audio file")
    output_path = validate_output_path(args.output)
    
    # Load model
    model = load_model(args)
    
    # Generate audio
    print(f"Synthesizing text: {args.text}")
    print(f"Reference audio: {prompt_audio_path}")
    print(f"Reference text: {args.prompt_text}")
    
    audio_array = model.generate(
        text=args.text,
        prompt_wav_path=str(prompt_audio_path),
        prompt_text=args.prompt_text,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        normalize=args.normalize,
        denoise=args.denoise
    )
    
    # Save audio
    sf.write(str(output_path), audio_array, 16000)
    print(f"Saved audio to: {output_path}")
    
    # Stats
    duration = len(audio_array) / 16000
    print(f"Duration: {duration:.2f}s")


def cmd_synthesize(args):
    """Direct TTS synthesis command."""
    # Validate inputs
    if not args.text:
        print("Error: Please provide text to synthesize (--text)")
        sys.exit(1)
    # Validate output path
    output_path = validate_output_path(args.output)
    # Load model
    model = load_model(args)
    # Generate audio
    print(f"Synthesizing text: {args.text}")
    
    audio_array = model.generate(
        text=args.text,
        prompt_wav_path=None,
        prompt_text=None,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        normalize=args.normalize,
        denoise=False  # 无参考音频时不需要降噪
    )
    
    # Save audio
    sf.write(str(output_path), audio_array, 16000)
    print(f"Saved audio to: {output_path}")
    
    # Stats
    duration = len(audio_array) / 16000
    print(f"Duration: {duration:.2f}s")


def cmd_batch(args):
    """Batch synthesis command."""
    # Validate input file
    input_file = validate_file_exists(args.input, "input file")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Failed to read input file: {e}")
        sys.exit(1)
    if not texts:
        print("Error: Input file is empty or contains no valid lines")
        sys.exit(1)
    print(f"Found {len(texts)} lines to process")
    
    model = load_model(args)
    prompt_audio_path = None
    if args.prompt_audio:
        prompt_audio_path = str(validate_file_exists(args.prompt_audio, "reference audio file"))
    
    success_count = 0
    for i, text in enumerate(texts, 1):
        print(f"\nProcessing {i}/{len(texts)}: {text[:50]}...")
        
        try:
            audio_array = model.generate(
                text=text,
                prompt_wav_path=prompt_audio_path,
                prompt_text=args.prompt_text,
                cfg_value=args.cfg_value,
                inference_timesteps=args.inference_timesteps,
                normalize=args.normalize,
                denoise=args.denoise and prompt_audio_path is not None
            )
            output_file = output_dir / f"output_{i:03d}.wav"
            sf.write(str(output_file), audio_array, 16000)
            
            duration = len(audio_array) / 16000
            print(f"  Saved: {output_file} ({duration:.2f}s)")
            success_count += 1
            
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    print(f"\nBatch finished: {success_count}/{len(texts)} succeeded")

def _build_unified_parser():
    """Build unified argument parser (no subcommands, route by args)."""
    parser = argparse.ArgumentParser(
        description="VoxCPM CLI (single parser) - voice cloning, direct TTS, and batch processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Direct synthesis (single sample)
  voxcpm --text "Hello world" --output out.wav

  # Voice cloning (reference audio + text)
  voxcpm --text "Hello world" --prompt-audio voice.wav --prompt-text "reference text" --output out.wav --denoise

  # Batch processing
  voxcpm --input texts.txt --output-dir ./outs

  # Select model (from Hub)
  voxcpm --text "Hello" --output out.wav --hf-model-id openbmb/VoxCPM-0.5B
        """
    )

    # Task selection (automatic routing by presence of args)
    parser.add_argument("--input", "-i", help="Input text file (one line per sample)")
    parser.add_argument("--output-dir", "-od", help="Output directory (for batch mode)")
    parser.add_argument("--text", "-t", help="Text to synthesize (single-sample mode)")
    parser.add_argument("--output", "-o", help="Output audio file path (single-sample mode)")

    # Prompt audio (for voice cloning)
    parser.add_argument("--prompt-audio", "-pa", help="Reference audio file path")
    parser.add_argument("--prompt-text", "-pt", help="Reference text corresponding to the audio")
    parser.add_argument("--denoise", action="store_true", help="Enable prompt speech enhancement (denoising)")

    # Generation parameters
    parser.add_argument("--cfg-value", type=float, default=2.0, help="CFG guidance scale (default: 2.0)")
    parser.add_argument("--inference-timesteps", type=int, default=10, help="Inference steps (default: 10)")
    parser.add_argument("--normalize", action="store_true", help="Enable text normalization")

    # Model loading parameters
    parser.add_argument("--model-path", type=str, help="Local VoxCPM model path (overrides Hub download)")
    parser.add_argument("--hf-model-id", type=str, default="openbmb/VoxCPM-0.5B", help="Hugging Face repo id (e.g., openbmb/VoxCPM-0.5B)")
    parser.add_argument("--cache-dir", type=str, help="Cache directory for Hub downloads")
    parser.add_argument("--local-files-only", action="store_true", help="Use only local files (no network)")
    parser.add_argument("--no-denoiser", action="store_true", help="Disable denoiser model loading")
    parser.add_argument("--zipenhancer-path", type=str, default="iic/speech_zipenhancer_ans_multiloss_16k_base", help="ZipEnhancer model id or local path (default reads from env)")

    return parser


def main():
    """Unified CLI entrypoint: route by provided arguments."""
    parser = _build_unified_parser()
    args = parser.parse_args()

    # Routing: prefer batch → single (clone/direct)
    if args.input:
        if not args.output_dir:
            print("Error: Batch mode requires --output-dir")
            parser.print_help()
            sys.exit(1)
        return cmd_batch(args)

    # Single-sample mode
    if not args.text or not args.output:
        print("Error: Single-sample mode requires --text and --output")
        parser.print_help()
        sys.exit(1)

    # If prompt audio+text provided → voice cloning
    if args.prompt_audio or args.prompt_text:
        if not args.prompt_audio or not args.prompt_text:
            print("Error: Voice cloning requires both --prompt-audio and --prompt-text")
            sys.exit(1)
        return cmd_clone(args)

    # Otherwise → direct synthesis
    return cmd_synthesize(args)


if __name__ == "__main__":
    main()
