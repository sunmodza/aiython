# The sample audio, image, and video are bundled beside this script.
# Running the full example requires version 3 routes for speech_to_text,
# text_to_speech, vision, image_generation, image_editing, and video.
# See aithon.toml.example for the required profile structure.
from pathlib import Path

ASSETS = Path(__file__).resolve().parent
meeting = str(ASSETS / "meeting.mp3")
shoe = str(ASSETS / "shoe.jpg")
clip = str(ASSETS / "clip.mp4")

transcript: str = transcribe meeting in its original language
summary: str = summarize transcript in English
speech = speak summary aloud
analysis: str = describe what is visible in shoe
banner = create a shoe banner from analysis
edited = edit shoe to have a white background
video_summary: str = summarize the events visible in clip
video = create a short promotional shoe video from analysis

print(transcript, summary, video_summary)
print(speech.path, banner.path, edited.path, video.path)
