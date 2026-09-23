# Bundled capability examples

These scripts use ordinary file paths. No `aithon` import is needed in the user
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

To use a configured provider, run `aithon setup` at the repository root and add
the routes from `aithon.toml.example` to the root configuration, replacing its
placeholder model IDs. The sample scripts inherit that configuration. A local
`aithon.toml` is optional and would need its own credential setup. Each script
anchors bundled asset paths to its own file, so it works from either directory.
The media example requires the version 3 speech, vision, image and video routes
shown in that template.
If a required route is missing, Aithon adds an inert, commented setup example to
this project's `aithon.toml` and stops with the missing capability named.
Static inspection needs no credentials:

```bash
uv run aithon --explain examples/capabilities/documents.py
uv run aithon --explain examples/capabilities/products.py
uv run aithon --explain examples/capabilities/media.py
```

Running the scripts with AI can call paid providers. `media.py` requests several
generation capabilities and can take longer than the other examples.
