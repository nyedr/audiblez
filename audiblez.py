#!/usr/bin/env python3
# audiblez - A program to convert e-books into audiobooks using
# Kokoro-82M model for high-quality text-to-speech synthesis.
# by Claudio Santini 2025 - https://claudio.uk

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
from string import Formatter
from bs4 import BeautifulSoup
from kokoro_onnx import Kokoro
from ebooklib import epub
from pick import pick
from pydub import AudioSegment
import concurrent.futures

def main(kokoro, file_path, lang, voice, pick_manually):
    """
    Main function that processes the EPUB, identifies chapters, and orchestrates
    parallel TTS generation followed by final audiobook creation.
    """
    filename = Path(file_path).name
    # Create output folder named after the EPUB file (without extension)
    output_folder = Path(filename.replace('.epub', ''))
    output_folder.mkdir(exist_ok=True)

    with warnings.catch_warnings():
        book = epub.read_epub(file_path)

    title = book.get_metadata('DC', 'title')[0][0] if book.get_metadata('DC', 'title') else "Untitled"
    creator = book.get_metadata('DC', 'creator')[0][0] if book.get_metadata('DC', 'creator') else "Unknown Author"
    intro = f"{title} by {creator}"

    print(intro)
    all_documents = [c for c in book.get_items() if c.get_type() == ebooklib.ITEM_DOCUMENT]
    print("Found Chapters/Docs:", [c.get_name() for c in all_documents])

    if pick_manually:
        chapters = pick_chapters(book)
    else:
        chapters = find_chapters(book)

    print("Selected chapters:", [c.get_name() for c in chapters])
    texts = extract_texts(chapters)

    has_ffmpeg = shutil.which('ffmpeg') is not None
    if not has_ffmpeg:
        print('\033[91m' + 'ffmpeg not found. Please install ffmpeg to create mp3 and m4b audiobook files.' + '\033[0m')

    total_chars = sum(len(t) for t in texts)
    print("Started at:", time.strftime('%H:%M:%S'))
    print(f"Total characters: {total_chars:,}")
    print("Total words:", len(' '.join(texts).split(' ')))

    # Generate WAV chapters in parallel
    chapter_args = []
    for i, text in enumerate(texts, start=1):
        if not text.strip():
            continue
        # Build the path for the WAV file inside the output folder
        chapter_filename = output_folder / filename.replace('.epub', f"_chapter_{i}.wav")
        # If the file exists, skip re-generation to save time
        if chapter_filename.exists():
            print(f"File for chapter {i} already exists. Skipping TTS.")
        else:
            # Prepend the intro only for the first chapter
            actual_text = intro + ".\n\n" + text if i == 1 else text
            chapter_args.append((kokoro, actual_text, voice, lang, chapter_filename, i))

    # Use multiprocessing to parallelize TTS creation
    if chapter_args:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for result in executor.map(process_chapter, chapter_args):
                print(result)  # print status messages from each chapter

    # If ffmpeg is available, create the final M4B
    if has_ffmpeg:
        # We rely on ffmpeg's concat mode for a faster combine process
        create_m4b_ffmpeg_concat(chapter_count=len(texts), output_folder=output_folder, filename=filename)
    else:
        print("Skipping M4B creation as ffmpeg is not found.")

def process_chapter(args):
    """
    Worker function to process each chapter text with Kokoro TTS in parallel.
    """
    kokoro, text, voice, lang, chapter_filename, chapter_index = args
    start_time = time.time()

    samples, sample_rate = kokoro.create(
        text, voice=voice, speed=1.0, lang=lang
    )
    sf.write(chapter_filename, samples, sample_rate)

    end_time = time.time()
    delta_seconds = end_time - start_time
    chars_per_sec = len(text) / delta_seconds if delta_seconds else 0
    msg = (
        f"Chapter {chapter_index} written to {chapter_filename}\n"
        f"Took {delta_seconds:.2f}s ({chars_per_sec:.0f} chars/s)"
    )
    return msg

def extract_texts(chapters):
    texts = []
    for chapter in chapters:
        xml = chapter.get_body_content()
        soup = BeautifulSoup(xml, features='lxml')
        chapter_text = ''
        html_content_tags = ['title', 'p', 'h1', 'h2', 'h3', 'h4']
        for child in soup.find_all(html_content_tags):
            inner_text = child.text.strip() if child.text else ""
            if inner_text:
                chapter_text += inner_text + '\n'
        texts.append(chapter_text)
    return texts

def is_chapter(c):
    name = c.get_name().lower()
    part = r"part\d{1,3}"
    if re.search(part, name):
        return True
    ch = r"ch\d{1,3}"
    if re.search(ch, name):
        return True
    if "chapter" in name:
        return True

def find_chapters(book, verbose=False):
    chapters = [c for c in book.get_items() if c.get_type() == ebooklib.ITEM_DOCUMENT and is_chapter(c)]
    if verbose:
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                print(f"'{item.get_name()}' -> length: {len(item.get_body_content())}")
    if len(chapters) == 0:
        print("No obvious chapters found. Defaulting to all available documents.")
        chapters = [c for c in book.get_items() if c.get_type() == ebooklib.ITEM_DOCUMENT]
    return chapters

def pick_chapters(book):
    all_chapters_names = [c.get_name() for c in book.get_items() if c.get_type() == ebooklib.ITEM_DOCUMENT]
    title = 'Select which chapters to read in the audiobook'
    selected_chapters_names = pick(all_chapters_names, title, multiselect=True, min_selection_count=1)
    selected_chapters_names = [c[0] for c in selected_chapters_names]
    selected_chapters = [c for c in book.get_items() if c.get_name() in selected_chapters_names]
    return selected_chapters

def strfdelta(tdelta, fmt='{D:02}d {H:02}h {M:02}m {S:02}s'):
    remainder = int(tdelta)
    f = Formatter()
    desired_fields = [field_tuple[1] for field_tuple in f.parse(fmt)]
    possible_fields = ('W', 'D', 'H', 'M', 'S')
    constants = {'W': 604800, 'D': 86400, 'H': 3600, 'M': 60, 'S': 1}
    values = {}
    for field in possible_fields:
        if field in desired_fields and field in constants:
            values[field], remainder = divmod(remainder, constants[field])
    return f.format(fmt, **values)

def create_m4b_ffmpeg_concat(chapter_count, output_folder, filename):
    """
    Use ffmpeg's concat demuxer for faster combining of .wav files into a single M4B.
    """
    print("Creating M4B via ffmpeg concat...")
    # Build a text file with the list of WAV files in correct order
    concat_file = output_folder / "wav_list.txt"
    with open(concat_file, "w") as f:
        for i in range(1, chapter_count+1):
            wav_file = output_folder / filename.replace('.epub', f"_chapter_{i}.wav")
            if wav_file.exists():
                f.write(f"file '{wav_file}'\n")

    tmp_filename = output_folder / filename.replace('.epub', '.tmp.m4a')
    final_filename = output_folder / filename.replace('.epub', '.m4b')

    # 1) Combine all .wav into .m4a with the concat demuxer
    subprocess.run([
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', str(concat_file),
        '-c:a', 'aac', '-b:a', '64k',
        str(tmp_filename)
    ], check=True)

    # 2) Convert .m4a to .m4b container
    subprocess.run([
        'ffmpeg', '-y', '-i', str(tmp_filename),
        '-c', 'copy', '-f', 'mp4', str(final_filename)
    ], check=True)

    # Cleanup
    tmp_filename.unlink(missing_ok=True)
    concat_file.unlink(missing_ok=True)

    print(f"M4B created: {final_filename}\nYou can delete the .wav files if you wish.")

def cli_main():
    if not Path('kokoro-v0_19.onnx').exists() or not Path('voices.json').exists():
        print('Error: kokoro-v0_19.onnx and voices.json must be in the current directory. Please download them with:')
        print('wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx')
        print('wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json')
        sys.exit(1)

    kokoro = Kokoro('kokoro-v0_19.onnx', 'voices.json')
    voices = list(kokoro.get_voices())
    voices_str = ', '.join(voices)
    epilog = (
        "example:\n"
        "  audiblez book.epub -l en-us -v af_sky"
    )
    default_voice = 'af_sky' if 'af_sky' in voices else voices[0]
    parser = argparse.ArgumentParser(
        epilog=epilog, 
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('epub_file_path', help='Path to the epub file')
    parser.add_argument('-l', '--lang', default='en-gb', help='Language code: en-gb, en-us, fr-fr, ja, ko, cmn')
    parser.add_argument('-v', '--voice', default=default_voice, help=f'Choose narrating voice: {voices_str}')
    parser.add_argument(
        '-p',
        '--pick',
        default=False,
        help='Manually select which chapters to read in the audiobook',
        action='store_true'
    )
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    main(kokoro, args.epub_file_path, args.lang, args.voice, args.pick)

if __name__ == '__main__':
    cli_main()
