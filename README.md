# Audiblez: Generate  audiobooks from e-books

Audiblez generates `.m4b` audiobooks from regular `.epub` e-books, 
using Kokoro's high-quality speech synthesis.

[Kokoro v0.19](https://huggingface.co/hexgrad/Kokoro-82M) is a recently published text-to-speech model with just 82M params and very natural sounding output.
It's released under Apache licence and it was trained on < 100 hours of audio.
It currently supports American, British English, French, Korean, Japanese and Mandarin, and a bunch of very good voices.

An example of the quality:

<audio controls=""><source type="audio/wav" src="https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/demo/HEARME.wav"></audio>

On my M2 MacBook Pro, **it takes about 2 hours to convert to mp3 the Selfish Gene by Richard Dawkins**, which is about 100,000 words (or 600,000 characters),
at a rate of about 80 characters per second.

## How to install and run

If you have python3 on your computer, you can install it with pip. 
Then you also need to download the onnx and voices files in the same folder, which are about ~360MB:

```bash
pip install audiblez
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json
```

Then to convert an epub file into an audiobook, just run:

```bash
audiblez book.epub -l en-gb -v af_sky
```

It will first create a bunch of `book_chapter_1.wav`, `book_chapter_2.wav`, etc. files in the same directory,
and at the end it will produce a `book.m4b` file with the whole book you can listen with VLC or any
 audiobook player.
It will only produce the `.m4b` file if you have `ffmpeg` installed on your machine.

## Supported Languages

- 🇺🇸 en-US
- 🇬🇧 en-GB
- 🇫🇷 fr-FR
- 🇯🇵 ja-JP
- 🇰🇷 ko-KR
- 🇨🇳 zh-CN

## Supported Voices

You can try them here: [https://huggingface.co/spaces/hexgrad/Kokoro-TTS](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)

- af
- af_bella
- af_nicole
- af_sarah 
- af_sky
- am_adam
- am_michael
- bf_emma
- bf_isabella
- bm_george
- bm_lewis

