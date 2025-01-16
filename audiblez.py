#!/usr/bin/env python3
# audiblez - A program to convert e-books into audiobooks using
# Kokoro-82M model for high-quality text-to-speech synthesis.
# by Claudio Santini 2025 - https://claudio.uk
# Updated: 2025-01 for improved user experience and GPU usage

import argparse
import sys
import time
import shutil
import subprocess
import soundfile as sf
import ebooklib
import warnings
import re
from pathlib import Path
from bs4 import BeautifulSoup
# NEW: import onnxruntime and pass GPU providers to Kokoro
import onnxruntime as ort
from kokoro_onnx import Kokoro
from ebooklib import epub
from pick import pick
from tqdm import tqdm


def main(kokoro, file_path, lang, voice, pick_manually):
    """
    Processes the EPUB, identifies chapters, converts them sequentially to WAV
    files via Kokoro TTS, and optionally combines into an M4B.
    """
    filename = Path(file_path).name
    output_folder = Path(filename.replace('.epub', ''))
    output_folder.mkdir(exist_ok=True)

    with warnings.catch_warnings():
        book = epub.read_epub(file_path)

    title = _try_get_metadata(book, 'title', default="Untitled")
    creator = _try_get_metadata(book, 'creator', default="Unknown Author")
    intro = f"{title} by {creator}"

    print(intro)
    all_docs = [c for c in book.get_items() if c.get_type() ==
                ebooklib.ITEM_DOCUMENT]
    print("Found Chapters/Docs:", [c.get_name() for c in all_docs])

    if pick_manually:
        chapters = pick_chapters(book)
    else:
        chapters = find_chapters(book)

    print("Selected chapters:", [c.get_name() for c in chapters])
    texts = extract_texts(chapters)

    has_ffmpeg = shutil.which('ffmpeg') is not None
    if not has_ffmpeg:
        print('\033[91mffmpeg not found. No m4b creation.\033[0m')

    total_chars = sum(len(t) for t in texts)
    print("Started at:", time.strftime('%H:%M:%S'))
    print(f"Total characters: {total_chars:,}")
    print("Total words:", len(' '.join(texts).split(' ')))

    start_time_all = time.time()
    n_chapters = len(texts)
    with tqdm(total=n_chapters, desc="Chapters Processed", unit="chapter") as pbar:
        for i, text in enumerate(texts, start=1):
            if not text.strip():
                pbar.update(1)
                continue

            chapter_filename = output_folder / \
                filename.replace('.epub', f"_chapter_{i}.wav")

            if chapter_filename.exists():
                print(f"[Skipping] Chapter {i} => Already exists.")
                pbar.update(1)
                continue

            actual_text = intro + "\n\n" + text if i == 1 else text

            start_chapter_time = time.time()
            samples, sample_rate = kokoro.create(
                actual_text, voice=voice, speed=1.0, lang=lang
            )
            sf.write(chapter_filename, samples, sample_rate)
            chapter_duration = time.time() - start_chapter_time

            c_len = len(actual_text)
            c_rate = c_len / chapter_duration if chapter_duration else 0
            print(
                f"Chapter {i} done: {chapter_filename}\n"
                f"    + {c_len:,} chars, took {chapter_duration:.2f} s, ~{c_rate:.0f} chars/s"
            )

            pbar.update(1)

    total_time = time.time() - start_time_all
    print(f"All chapters processed in {total_time:.2f} seconds.")

    if has_ffmpeg:
        create_m4b_ffmpeg_concat(n_chapters, output_folder, filename)
    else:
        print("Skipping M4B creation (no ffmpeg).")


def _try_get_metadata(book, field, default="Unknown"):
    data = book.get_metadata('DC', field)
    if not data or not data[0] or not data[0][0]:
        return default
    return data[0][0]


def extract_texts(chapters):
    texts = []
    from lxml import etree
    for chapter in chapters:
        xml = chapter.get_body_content()
        soup = BeautifulSoup(xml, "lxml")
        chapter_text = ""
        for child in soup.find_all(["title", "p", "h1", "h2", "h3", "h4"]):
            if child.text:
                chapter_text += child.text.strip() + "\n"
        texts.append(chapter_text)
    return texts


def is_chapter(c):
    name = c.get_name().lower()
    if re.search(r"part\d{1,3}", name):
        return True
    if re.search(r"ch\d{1,3}", name):
        return True
    if "chapter" in name:
        return True
    return False


def find_chapters(book, verbose=False):
    chapters = [c for c in book.get_items()
                if c.get_type() == ebooklib.ITEM_DOCUMENT and is_chapter(c)]
    if verbose:
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                size = len(item.get_body_content())
                label = "X" if item in chapters else "-"
                print(f"{item.get_name()} (size={size}) [{label}]")

    if not chapters:
        print("No obvious chapters found. Using all document items.")
        chapters = [c for c in book.get_items() if c.get_type() ==
                    ebooklib.ITEM_DOCUMENT]
    return chapters


def pick_chapters(book):
    all_chapters = [c for c in book.get_items() if c.get_type()
                    == ebooklib.ITEM_DOCUMENT]
    names = [c.get_name() for c in all_chapters]
    title = "Select which chapters to convert:"
    selection = pick(names, title, multiselect=True, min_selection_count=1)
    chosen_names = [sel[0] for sel in selection]
    return [c for c in all_chapters if c.get_name() in chosen_names]


def create_m4b_ffmpeg_concat(chapter_count, output_folder, filename):
    print("Creating M4B via ffmpeg concat...")
    concat_file = output_folder / "wav_list.txt"
    with open(concat_file, "w") as f:
        for i in range(1, chapter_count + 1):
            wav_file = output_folder / \
                filename.replace(".epub", f"_chapter_{i}.wav")
            if wav_file.exists():
                f.write(f"file '{wav_file}'\n")

    tmp_filename = output_folder / filename.replace(".epub", ".tmp.m4a")
    final_filename = output_folder / filename.replace(".epub", ".m4b")

    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c:a", "aac", "-b:a", "64k",
        str(tmp_filename)
    ], check=True)

    subprocess.run([
        "ffmpeg", "-y", "-i", str(tmp_filename),
        "-c", "copy", "-f", "mp4", str(final_filename)
    ], check=True)

    tmp_filename.unlink(missing_ok=True)
    concat_file.unlink(missing_ok=True)
    print(f"Finished combining into M4B: {final_filename}\n"
          f"You can delete the .wav files if you like.")


def cli_main():
    import onnxruntime as ort

    # Define the providers to use
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session_options = ort.SessionOptions()
    try:
        # Attempt to create a GPU-based session
        session = ort.InferenceSession(
            "kokoro-v0_19.onnx",
            providers=providers,
            sess_options=session_options
        )
        print("ONNX Runtime GPU session created successfully.")
    except Exception as e:
        print("GPU session initialization failed, falling back to CPU.\n", e)
        session = ort.InferenceSession(
            "kokoro-v0_19.onnx",
            providers=["CPUExecutionProvider"],
            sess_options=session_options
        )

    # Initialize Kokoro with its model path and voice file
    kokoro = Kokoro("kokoro-v0_19.onnx", "voices.json")
    # Override the session with the GPU (or fallback CPU) session
    kokoro.session = session

    # Continue with the rest of cli_main as before.
    voices = list(kokoro.get_voices())
    voices_str = ", ".join(voices)
    default_voice = "af_sky" if "af_sky" in voices else voices[0]

    epilog = "Example:\n  audiblez my_book.epub -l en-us -v af_sky"
    parser = argparse.ArgumentParser(
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("epub_file_path", help="Path to the .epub file")
    parser.add_argument("-l", "--lang", default="en-gb", help="Language code")
    parser.add_argument("-v", "--voice", default=default_voice,
                        help=f"Choose voice: {voices_str}")
    parser.add_argument("-p", "--pick", default=False, action="store_true",
                        help="Manually pick chapters (interactive)")

    if not Path("kokoro-v0_19.onnx").exists() or not Path("voices.json").exists():
        print(
            "Error: kokoro-v0_19.onnx and voices.json must exist in the current directory.")
        sys.exit(1)

    args = parser.parse_args()

    main(kokoro, args.epub_file_path, args.lang, args.voice, args.pick)


if __name__ == '__main__':
    cli_main()
