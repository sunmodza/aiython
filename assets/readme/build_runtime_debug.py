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
    (11, "for ticket in tickets:"),
    (12, '    kind: Literal["bug", "billing"] = classify this ticket'),
    (13, "    queues[kind].append(ticket)"),
    (15, "summary = summarize the routed tickets in one sentence"),
    (16, "print(queues, summary)"),
)
AI_EXPRESSIONS = {
    12: ('kind: Literal["bug", "billing"] = ', 'classify this ticket'),
    15: ('summary = ', 'summarize the routed tickets in one sentence'),
}
OWNER = {1: "frontend", 2: "python", 3: "runtime", 4: "model",
         5: "python", 6: "python", 7: "runtime", 8: "model",
         9: "python", 10: "python", 11: "runtime", 12: "model", 13: "python"}
CURRENT_LINE = {1: 12, 2: 11, 3: 12, 4: 12, 5: 13, 6: 11,
                7: 12, 8: 12, 9: 13, 10: 15, 11: 15, 12: 15, 13: 16}
ACCENT = {stage: (VIOLET if owner == "frontend" else AMBER if owner == "runtime"
                  else PINK if owner == "model" else GREEN if stage in (5, 9, 13) else BLUE)
          for stage, owner in OWNER.items()}
SCENES = {
    1: ("01 / BEFORE EXECUTION", "Mark both AI expressions",
        ("inside loop: classify this ticket",
         "after loop : summarize routed tickets",
         "scope      : two red-underlined spans"),
        "Aiython transforms an in-memory copy of the source."),
    2: ("02 / PYTHON", "Pick the first ticket",
        ('ticket = tickets[0]', 'preview: "PDF upload freezes..."'),
        "The loop selects the ticket before AI is called."),
    3: ("03 / AI RUNTIME", "Classify the first ticket",
        ('statement: classify this ticket',
         'binding  : ticket (str handle)',
         'expected : Literal["bug", "billing"]'),
        "Only the current ticket is in this invocation."),
    4: ("04 / MODEL RESULT", "Return bug for ticket 1",
        ('get_binding(name="ticket")',
         'finish(... "bug")',
         'Literal contract: passed'),
        "The model answers once for the current ticket."),
    5: ("05 / PYTHON", "Append the first ticket",
        ('kind = "bug"',
         'queues["bug"] = [tickets[0]]'),
        "Python performs the append, then loops again."),
    6: ("06 / PYTHON", "Pick the second ticket",
        ('ticket = tickets[1]', 'preview: "Invoice and billing contact..."'),
        "The loop reaches the same AI expression again."),
    7: ("07 / AI RUNTIME", "Classify the second ticket",
        ('statement: classify this ticket',
         'binding  : ticket (new live value)',
         'expected : Literal["bug", "billing"]'),
        "This is a new invocation for this iteration."),
    8: ("08 / MODEL RESULT", "Return billing for ticket 2",
        ('get_binding(name="ticket")',
         'finish(... "billing")',
         'Literal contract: passed'),
        "The second result selects the billing queue."),
    9: ("09 / PYTHON", "Append the second ticket",
        ('kind = "billing"',
         'queues["billing"] = [tickets[1]]'),
        "Both tickets are now routed; the loop ends."),
    10: ("10 / PYTHON", "Move past the loop",
         ('bug queue    : 1 ticket',
          'billing queue: 1 ticket',
          'next line    : summary = ...'),
         "The summary expression runs once after the loop."),
    11: ("11 / AI RUNTIME", "Build the summary context",
         ('statement: summarize the routed tickets',
          '           in one sentence',
          'binding  : queues (two populated lists)',
          'output   : no declared type'),
         "AI can inspect the completed queues in the live frame."),
    12: ("12 / MODEL RESULT", "Return one overall summary",
         ('get_binding(name="queues")',
          'finish(outcome={"kind":"literal",',
          '  "value":"One bug and one billing request."})'),
         "This call runs once, after both classifications."),
    13: ("13 / PYTHON", "Print the queues and summary",
         ('queues: bug=1, billing=1',
          'summary: "One bug and one billing request."'),
         "Python finishes the script with both results."),
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


def red_squiggle(start: int, end: int, y: int) -> str:
    path = f"M{start} {y}"
    for x in range(start, end, 16):
        step = min(16, end - x)
        half = step / 2
        path += f" q{step / 4:g} -5 {half:g} 0 q{step / 4:g} 5 {half:g} 0"
    return f'<path d="{path}" fill="none" stroke="#ff6e78" stroke-width="2.8" stroke-linecap="round"/>'


def ai_squiggle(baseline: int, code_x: int, prefix: str, expression: str) -> str:
    """Underline exactly one expression plain Python cannot parse."""
    start = round(code_x + len(prefix) * CODE_ADVANCE)
    end = round(start + len(expression) * CODE_ADVANCE)
    return red_squiggle(start, end, baseline + 10)


def editor(stage: int) -> list[str]:
    current = CURRENT_LINE[stage]
    accent = ACCENT[stage]
    fill = {
        "frontend": "#302c48", "python": "#1e3549",
        "runtime": "#3a3020", "model": "#3b293b",
    }[OWNER[stage]]
    if stage in (5, 9, 13):
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
        top = 181 + index * 56
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
        if line_number in AI_EXPRESSIONS:
            parts.append(ai_squiggle(baseline, code_x, *AI_EXPRESSIONS[line_number]))
    location = "SOURCE SCAN" if stage == 1 else f"LINE {current}"
    activity = {
        1: "Aiython prepares the code in memory",
        2: "Python selects ticket 1",
        3: "Runtime builds classification context",
        4: "AI classifies ticket 1 as bug",
        5: "Python appends ticket 1",
        6: "Python selects ticket 2",
        7: "Runtime builds classification context",
        8: "AI classifies ticket 2 as billing",
        9: "Python appends ticket 2",
        10: "Python exits the loop",
        11: "Runtime reads the completed queues",
        12: "AI summarizes both queues once",
        13: "Python prints the final result",
    }[stage]
    parts += [
        label(65, 543, location, size=17, color=accent, weight=700, mono=True),
        label(228, 543, activity, size=17, color="#c4d7e6"),
    ]
    return parts


def execution_path(stage: int) -> list[str]:
    active = 1 if stage == 1 else 4 if stage in (5, 9, 13) else 2 if OWNER[stage] == "python" else 3
    accent = ACCENT[stage]
    mode = "PREPARING" if stage == 1 else "PYTHON MODE" if OWNER[stage] == "python" else "AI MODE"
    nodes = (
        (925, 125, "PREPARE CODE", "mark AI span", VIOLET),
        (1068, 125, "PYTHON", "run script", BLUE),
        (1211, 125, "AI RUNTIME", "context + model", accent if active == 3 else AMBER),
        (1354, 142, "PYTHON", "resume script", GREEN),
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
    connector = {2: 1059, 3: 1202, 4: 1345}.get(active)
    for x in (1059, 1202, 1345):
        parts.append(label(x, 211, "→", size=18,
                           color=accent if connector == x else "#617e91", weight=700,
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
        "<title id=\"title\">How Aiython routes two tickets and summarizes the queues</title>",
        "<desc id=\"description\">The frontend marks two red-underlined expressions "
        "in memory. Python routes the first ticket to bug and the second to billing, making "
        "one classification call per loop iteration. After the loop, one AI call summarizes "
        "the completed queues. Python prints both queues and the summary.</desc>",
        box(0, 0, WIDTH, HEIGHT, fill="#0b1726", radius=24),
        label(40, 57, "Two tickets → two classifications → one summary",
              size=31, weight=700),
        label(40, 91, "Arrow = current line. Red squiggles = invalid in plain Python. Lit box = current phase.",
              size=20, color="#b9cfdf"),
        *editor(stage),
        *execution_path(stage),
        *detail(stage),
        label(40, 602, "Illustrative model results. Python owns the loop and prints the final output.",
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
    (HERE / "runtime-debug.svg").write_text(frame(13), encoding="utf-8")
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
        durations = (1.8, 1.5, 1.8, 2.0, 1.6, 1.5, 1.8, 2.0, 1.6, 1.6, 2.0, 2.2, 2.5)
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
