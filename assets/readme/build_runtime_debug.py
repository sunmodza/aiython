"""Build the README's source-to-model runtime GIF and static SVG poster.

FFmpeg is needed only to regenerate these README assets, not to run Aiython.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape


HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parent.parent / "examples/recipes/03_loop.py"
WIDTH = 1560
HEIGHT = 620
BLUE = "#6acdf5"
VIOLET = "#b9a7ff"
AMBER = "#ffc56c"
PINK = "#efa2d8"
GREEN = "#77dfb8"
CODE_FONT_SIZE = 19
CODE_ADVANCE = CODE_FONT_SIZE * 0.602  # DejaVu Sans Mono glyph width.
CODE_LINES = (
    (8, "for message in inbox:"),
    (9, '    kind: Literal["bug", "billing"] = classify this message'),
    (10, "    queues[kind].append(message)"),
    (12, "print(queues)"),
)
OWNER = {1: "frontend", 2: "python", 3: "runtime", 4: "runtime",
         5: "model", 6: "model", 7: "runtime", 8: "python"}
CURRENT_LINE = {1: 9, 2: 8, 3: 9, 4: 9, 5: 9, 6: 9, 7: 9, 8: 10}
ACCENT = {1: VIOLET, 2: BLUE, 3: AMBER, 4: AMBER,
          5: PINK, 6: PINK, 7: AMBER, 8: GREEN}
SCENES = {
    1: ("01 / BEFORE EXECUTION", "Mark the AI expression",
        ("source  : classify this message",
         "compiled: __aiython_runtime__.execute(...)",
         "scope   : only the red-underlined span"),
        "Aiython transforms an in-memory copy of the source."),
    2: ("02 / PYTHON", "Run the loop normally",
        ('message = "Upload crashes"', 'queues["bug"] = []'),
        "Python picks this item; no model is needed yet."),
    3: ("03 / ENTER AI MODE", "Pause at the generated call",
        ("line 9: __aiython_runtime__.execute(...)",),
        "Python waits here for one value for kind."),
    4: ("04 / AI RUNTIME", "Build context for this invocation",
        ("statement: classify this message",
         "source   : nearby Python lines",
         "binding  : message (str handle)",
         'expected : Literal["bug", "billing"]'),
        "The live value can be read through a tool."),
    5: ("05 / MODEL TOOL (EXAMPLE)", "Read the live binding",
        ('get_binding(name="message")',
         'tool result.value: "Upload crashes"'),
        "This tool step is optional; the frame remains live."),
    6: ("06 / MODEL RESULT (EXAMPLE)", "Return a candidate value",
        ('finish(outcome={"kind":"literal",',
         '                "value":"bug"})'),
        "The model answers for this invocation only."),
    7: ("07 / AI RUNTIME", "Validate the declared result type",
        ('returned: "bug"',
         'contract: Literal["bug", "billing"]',
         "check   : passed"),
        "An invalid value cannot be assigned to kind."),
    8: ("08 / RETURN TO PYTHON MODE", "Resume and run the next line",
        ('kind = "bug"',
         'queues["bug"] = ["Upload crashes"]'),
        "Python performs the append, then continues the loop."),
}


def label(x: int, y: int, value: str, *, size: int = 20,
          color: str = "#eef6ff", weight: int = 400,
          mono: bool = False, anchor: str = "start") -> str:
    family = "DejaVu Sans Mono, monospace" if mono else "DejaVu Sans, Arial, sans-serif"
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{color}" '
        f'font-family="{family}" font-size="{size}" font-weight="{weight}">'
        f"{escape(value)}</text>"
    )


def box(x: int, y: int, width: int, height: int, *, fill: str,
        stroke: str = "none", radius: int = 12, stroke_width: int = 1) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        f'rx="{radius}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{stroke_width}"/>'
    )


def ai_squiggle(baseline: int, code_x: int) -> str:
    """Underline exactly the natural-language expression in the rendered code."""
    prefix = 'kind: Literal["bug", "billing"] = '
    expression = "classify this message"
    start = round(code_x + len(prefix) * CODE_ADVANCE)
    end = round(start + len(expression) * CODE_ADVANCE)
    y = baseline + 10
    path = f"M{start} {y}"
    for x in range(start, end, 16):
        step = min(16, end - x)
        half = step / 2
        path += f" q{step / 4:g} -5 {half:g} 0 q{step / 4:g} 5 {half:g} 0"
    return f'<path d="{path}" fill="none" stroke="#ff6e78" stroke-width="2.8" stroke-linecap="round"/>'


def editor(stage: int) -> list[str]:
    current = CURRENT_LINE[stage]
    accent = ACCENT[stage]
    fill = {
        "frontend": "#302c48", "python": "#1e3549",
        "runtime": "#3a3020", "model": "#3b293b",
    }[OWNER[stage]]
    if stage == 8:
        fill = "#193c34"
    parts = [
        box(40, 116, 830, 452, fill="#142334", stroke="#405a70", radius=19, stroke_width=2),
        box(40, 116, 830, 49, fill="#1b3045", radius=18),
        box(40, 150, 830, 15, fill="#1b3045", radius=0),
    ]
    for x, color in ((66, "#fa817a"), (85, "#f7c663"), (104, "#7bd4a9")):
        parts.append(f'<circle cx="{x}" cy="141" r="5" fill="{color}"/>')
    parts += [
        label(130, 148, "03_loop.py", size=18, weight=700),
        label(838, 147, "SOURCE VIEW", size=16, color="#a6c0d2", anchor="end"),
        box(57, 172, 796, 334, fill="#0f1d2d", radius=11),
    ]
    for index, (line_number, code) in enumerate(CODE_LINES):
        top = 188 + index * 62
        baseline = top + 35
        indent = len(code) - len(code.lstrip(" "))
        code_x = 145 + round(indent * CODE_ADVANCE)
        if line_number == current:
            parts.append(box(65, top, 779, 51, fill=fill, stroke=accent,
                             radius=8, stroke_width=2))
            if stage != 1:  # The frontend scans before Python starts executing.
                parts.append(f'<path d="M76 {top + 16}l13 9-13 9z" fill="{accent}"/>')
        parts += [
            label(119, baseline, str(line_number), size=18,
                  color=accent if line_number == current else "#7894a8",
                  mono=True, anchor="end"),
            label(code_x, baseline, code.lstrip(" "), size=CODE_FONT_SIZE,
                  color="#ffffff" if line_number == current else "#bdccda",
                  mono=True, weight=700 if line_number == current else 400),
        ]
        if line_number == 9:
            parts.append(ai_squiggle(baseline, code_x))
    location = "SOURCE SCAN" if stage == 1 else f"LINE {current}"
    activity = {
        1: "Aiython prepares the code in memory",
        2: "Python running",
        3: "Python paused at the AI boundary",
        4: "Runtime builds model context",
        5: "Model can read the live frame",
        6: "Model returns a candidate",
        7: "Runtime validates the candidate",
        8: "Python running again",
    }[stage]
    parts += [
        label(65, 543, location, size=17, color=accent, weight=700, mono=True),
        label(228, 543, activity, size=17, color="#c4d7e6"),
    ]
    return parts


def execution_path(stage: int) -> list[str]:
    active = 1 if stage == 1 else 2 if stage == 2 else 4 if stage == 8 else 3
    accent = ACCENT[stage]
    mode = "PREPARING" if stage == 1 else "PYTHON MODE" if stage in (2, 8) else "AI MODE"
    nodes = (
        (925, 125, "PREPARE CODE", "mark AI span", VIOLET),
        (1068, 125, "PYTHON", "run loop", BLUE),
        (1211, 125, "AI RUNTIME", "context + model", accent if active == 3 else AMBER),
        (1354, 142, "PYTHON", "resume loop", GREEN),
    )
    parts = [
        box(900, 116, 620, 452, fill="#142334", stroke="#405a70",
            radius=19, stroke_width=2),
        label(925, 151, "Execution path", size=22, weight=700),
        box(1352, 128, 143, 34, fill="#1d3042", stroke=accent,
            radius=9, stroke_width=2),
        label(1423, 151, mode, size=15, color=accent, weight=700,
              anchor="middle"),
    ]
    for number, (x, width, title, subtitle, color) in enumerate(nodes, 1):
        lit = number == active
        parts += [
            box(x, 177, width, 57,
                fill={1: "#302c48", 2: "#1e3549", 3: "#3a3020", 4: "#193c34"}[number]
                if lit else "#1d2a39",
                stroke=color if lit else "#486075", radius=11, stroke_width=2),
            label(x + width // 2, 202, title,
                  size=13 if title == "PREPARE CODE" else 14,
                  color=color if lit else "#b4c8d7", weight=700,
                  anchor="middle"),
            label(x + width // 2, 221, subtitle, size=12,
                  color="#e1ebf3" if lit else "#8da4b7",
                  anchor="middle"),
        ]
    # A single highlighted arrow shows each handoff between execution phases.
    connector = {2: 1059, 3: 1202, 8: 1345}
    for x in (1059, 1202, 1345):
        parts.append(label(x, 211, "→", size=18,
                           color=accent if connector.get(stage) == x else "#617e91", weight=700,
                           anchor="middle"))
    return parts


def detail(stage: int) -> list[str]:
    kicker, heading, lines, footer = SCENES[stage]
    accent = ACCENT[stage]
    parts = [
        box(924, 255, 572, 293, fill="#0d1b29", stroke=accent,
            radius=13, stroke_width=2),
        label(946, 289, kicker, size=16, color=accent, weight=700),
        label(946, 325, heading, size=22, weight=700),
        box(945, 344, 529, 151, fill="#172a3a", radius=8),
    ]
    first_y = 374 if len(lines) >= 3 else 390
    for index, line in enumerate(lines):
        parts.append(label(961, first_y + index * 32, line,
                           size=17 if stage in (1, 4) else 18,
                           color="#ffffff", mono=True))
    parts.append(label(946, 527, footer, size=17, color="#c4d8e7"))
    return parts


def frame(stage: int) -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title description">',
        "<title id=\"title\">How Aiython hands a Python expression to an AI model</title>",
        "<desc id=\"description\">The frontend marks and transforms the red-underlined expression "
        "in memory. Python executes to that line, where the AI runtime builds a request from "
        "the statement, nearby source, object metadata and output type. In this example the "
        "model reads the live message with a tool, returns bug, the runtime checks the type, "
        "and Python resumes to update its queue.</desc>",
        box(0, 0, WIDTH, HEIGHT, fill="#0b1726", radius=24),
        label(40, 57, "One expression: source → context → model → Python",
              size=31, weight=700),
        label(40, 91, "Arrow = current line. Red squiggle = invalid in plain Python. Lit box = current phase.",
              size=20, color="#b9cfdf"),
        *editor(stage),
        *execution_path(stage),
        *detail(stage),
        label(40, 602, "Illustrative path for one loop iteration. Model tool use may vary.",
              size=17, color="#b9cfdf"),
        "</svg>",
    ]
    return "\n".join(parts) + "\n"


def run(*args: str) -> None:
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL)


def main() -> None:
    source_lines = EXAMPLE.read_text(encoding="utf-8").splitlines()
    for line_number, code in CODE_LINES:
        if source_lines[line_number - 1] != code:
            raise ValueError(f"The README animation no longer matches {EXAMPLE}:{line_number}")
    (HERE / "runtime-debug.svg").write_text(frame(8), encoding="utf-8")
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        frames = []
        for stage in SCENES:
            svg = temporary / f"stage-{stage}.svg"
            png = temporary / f"stage-{stage}.png"
            svg.write_text(frame(stage), encoding="utf-8")
            run("ffmpeg", "-v", "error", "-y", "-i", str(svg),
                "-frames:v", "1", str(png))
            frames.append(png)
        sequence = temporary / "sequence.txt"
        durations = (1.8, 1.6, 1.6, 2.7, 2.2, 1.8, 1.7, 2.4)
        sequence.write_text("".join(
            f"file '{png}'\nduration {duration}\n"
            for png, duration in zip(frames, durations)
        ) + f"file '{frames[-1]}'\n", encoding="utf-8")
        palette = temporary / "palette.png"
        inputs = ("-v", "error", "-y", "-safe", "0", "-f", "concat", "-i", str(sequence))
        run("ffmpeg", *inputs, "-vf", "palettegen=max_colors=96:reserve_transparent=0",
            str(palette))
        run("ffmpeg", *inputs, "-i", str(palette), "-lavfi",
            "[0:v][1:v]paletteuse=dither=none", "-fps_mode", "vfr",
            "-loop", "0", str(HERE / "runtime-debug.gif"))


if __name__ == "__main__":
    main()
