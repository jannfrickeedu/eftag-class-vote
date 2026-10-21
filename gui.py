import cv2
import gradio as gr
from pathlib import Path
from PIL import Image, ImageDraw
from main import Vote, Option

EXAMPLES_DIR = Path(__file__).parent / "resources" / "images" / "class-data-set"

COLORS = {"Thumb_Up": "lime", "Thumb_Down": "red"}


def draw_boxes(image, details):
    draw = ImageDraw.Draw(image)
    width = max(2, image.width // 300)
    for box, gesture in details:
        color = COLORS.get(gesture, "gray")
        x0, y0 = box.origin_x, box.origin_y
        draw.rectangle((x0, y0, x0 + box.width, y0 + box.height), outline=color, width=width)
        draw.text((x0 + 4, y0 + 4), gesture or "no hand", fill=color)
    return image


def take_snapshot(camera_index=0):
    # Capture on the Python side so it does not depend on browser camera permissions.
    cap = cv2.VideoCapture(camera_index)
    try:
        if not cap.isOpened():
            raise gr.Error(f"Could not open webcam {camera_index}")
        # skip the first frames so auto exposure can settle
        for _ in range(10):
            ok, frame = cap.read()
        if not ok:
            raise gr.Error("Could not read a frame from the webcam")
    finally:
        cap.release()
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def count_votes(image, name_a, name_b):
    if image is None:
        return None, "No image"

    image = image.convert("RGB")
    a = Option(name_a or "positive")
    b = Option(name_b or "negative")
    vote = Vote(image, a, b)

    if a.votes > b.votes:
        winner = f"Winner: {a.name}"
    elif b.votes > a.votes:
        winner = f"Winner: {b.name}"
    else:
        winner = "Tie"

    annotated = draw_boxes(image.copy(), vote.details)
    return annotated, f"{a}\n{b}\nPeople detected: {len(vote.details)}\n{winner}"


with gr.Blocks(title="Class Vote") as demo:
    gr.Markdown("# Class Vote\nUpload a class photo or take a webcam snapshot, then count 👍 / 👎.")
    with gr.Row():
        inp = gr.Image(type="pil", sources=["upload"], label="Class photo")
        out = gr.Image(type="pil", label="Detected votes")
    with gr.Row():
        name_a = gr.Textbox("positive", label="👍 option")
        name_b = gr.Textbox("negative", label="👎 option")
    with gr.Row():
        snap_btn = gr.Button("📷 Take webcam snapshot")
        btn = gr.Button("Count votes", variant="primary")
    result = gr.Textbox(label="Result", lines=4)
    snap_btn.click(take_snapshot, None, inp)
    btn.click(count_votes, [inp, name_a, name_b], [out, result])
    gr.Examples([str(p) for p in sorted(EXAMPLES_DIR.iterdir())], inputs=inp)

if __name__ == "__main__":
    demo.launch()
