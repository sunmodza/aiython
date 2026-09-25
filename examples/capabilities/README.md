# Bundled capability examples

These scripts use ordinary file paths. No `aiython` import is needed in the user
script; the AI chooses among the capabilities configured for the project.

| File | Used by | Contents |
| --- | --- | --- |
| `policy.md` | `documents.py` | Fictional store refund policy with a 30-day request window and required order details. |
| `meeting.mp3` | `media.py` | 15-second synthetic product-launch conversation in English. |
| `meeting-transcript.md` | Verification | Ground-truth script for the sample audio. |
| `red-shoe.jpg` | `products.py` | Catalog photo of an unbranded red running shoe. |
| `blue-boot.jpg` | `products.py` | Catalog photo of an unbranded navy boot. |
| `shoe.jpg` | `products.py`, `media.py` | A second camera angle of the red running shoe; the expected visual-search match is `red-shoe.jpg`. |
| `clip.mp4` | `media.py` | A short product-preview montage showing the red shoe, then the blue boot. |

The three product images were created with the built-in image generation tool for
this repository: an unbranded red mesh running shoe with a white sole, an
unbranded navy ankle boot, and another view of the same red shoe. The images have
no brand marks. The video was assembled from the generated images with FFmpeg;
the audio was synthesized from `meeting-transcript.md` using FFmpeg's Flite voices.
All names, products, and policy details are fictional.

Run `aiython setup` from the repository root or this directory and choose
**Configure capabilities**. Select the routes needed by a script; the menu lets
you add several routes in one session and prompts for route-specific credentials.
For example, `aiython setup --capability speech_to_text --model openai/whisper-1`
adds a transcription route. `media.py` also needs text to speech, vision, image
generation and editing, and video understanding and generation. Use the model IDs
available to your provider account; the IDs in `aiython.toml.example` are examples.
`video_generation.py` uses only the reasoning and video generation routes, so it
is a smaller way to try OpenRouter video without running the full media pipeline.
The sample scripts inherit the root configuration. A local `aiython.toml` is
optional and would need its own credential setup. Each script anchors bundled
asset paths to its own file, so it works from either directory. If a required
route is missing, Aiython prints the matching `aiython setup --capability ...`
command, adds an inert commented example to the project config, and stops.
Static inspection needs no credentials:

```bash
uv run aiython --explain examples/capabilities/documents.py
uv run aiython --explain examples/capabilities/products.py
uv run aiython --explain examples/capabilities/media.py
uv run aiython --explain examples/capabilities/video_generation.py
```

Running the scripts with AI can call paid providers. `media.py` requests several
generation capabilities and can take longer than the other examples.
