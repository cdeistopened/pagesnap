#!/usr/bin/env python3
"""
OCR using Gemini 2.0 Flash for PageSnap sessions.
Processes all images in a session and concatenates OCR results.
"""

import os
import sys
import base64
import glob
import argparse
from pathlib import Path

try:
    from google import genai
except ImportError:
    print("Please install google-genai: pip install google-genai")
    sys.exit(1)


def load_env():
    """Load environment variables from .env file."""
    env_paths = [
        Path(__file__).parent / ".env",
        Path.cwd() / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ.setdefault(key.strip(), value.strip())
            break


def encode_image(image_path: str) -> bytes:
    """Read image as bytes."""
    with open(image_path, "rb") as f:
        return f.read()


def ocr_image(client, image_path: str) -> str:
    """OCR a single image using Gemini 2.0 Flash."""
    image_data = encode_image(image_path)

    prompt = """Extract all text from this scanned book page.
Output the text exactly as it appears, preserving:
- Paragraph breaks
- Headers and subheaders
- Any formatting like italics or bold (use markdown)
- Footnotes (place at bottom of page text)

Do not add any commentary or explanations. Just output the text."""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            genai.types.Part.from_bytes(data=image_data, mime_type="image/jpeg"),
            prompt
        ]
    )

    return response.text


def _handle_error(message: str, exit_on_error: bool):
    """Handle errors differently for CLI vs. library usage."""
    if exit_on_error:
        print(f"Error: {message}")
        sys.exit(1)
    raise RuntimeError(message)


def process_session(session_path: str, output_path: str = None, progress_callback=None, exit_on_error: bool = True) -> str:
    """Process all images in a session directory."""
    load_env()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        _handle_error("GEMINI_API_KEY environment variable not set", exit_on_error)

    client = genai.Client(api_key=api_key)

    session_dir = Path(session_path)
    if not session_dir.exists():
        _handle_error(f"Session directory not found: {session_path}", exit_on_error)

    images = sorted(glob.glob(str(session_dir / "*.jpg")))
    if not images:
        _handle_error(f"No JPG images found in {session_path}", exit_on_error)

    print(f"Found {len(images)} images in session")

    total_pages = len(images)
    if progress_callback:
        progress_callback(0, total_pages, None)

    all_text = []
    for i, img_path in enumerate(images, 1):
        print(f"Processing {i}/{len(images)}: {Path(img_path).name}...")
        try:
            text = ocr_image(client, img_path)
            all_text.append(f"<!-- Page {i} -->\n\n{text}")
        except Exception as e:
            print(f"  Error: {e}")
            all_text.append(f"<!-- Page {i} - OCR FAILED: {e} -->")
        if progress_callback:
            progress_callback(i, total_pages, Path(img_path).name)

    combined = "\n\n---\n\n".join(all_text)

    if output_path is None:
        output_path = session_dir / f"{session_dir.name}_ocr.md"

    with open(output_path, "w") as f:
        f.write(combined)

    print(f"\nOCR complete! Output saved to: {output_path}")
    if progress_callback:
        progress_callback(total_pages, total_pages, None)
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="OCR PageSnap sessions using Gemini 2.0 Flash")
    parser.add_argument("session", help="Path to session directory containing JPG images")
    parser.add_argument("-o", "--output", help="Output markdown file path (default: <session>_ocr.md)")

    args = parser.parse_args()
    process_session(args.session, args.output)


if __name__ == "__main__":
    main()
