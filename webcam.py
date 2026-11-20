import argparse
import base64
import os
import threading
import time
from pathlib import Path

import cv2

MODEL = os.environ.get("")
PROMPT = "Describe what you see in this webcam frame in two or three sentences."
SNAP_DIR = Path("snapshots")


class Assistant:
    def __init__(self):
        self.reply = ""
        self.busy = False
        self.client = None
        if os.environ.get("OPENAI_KEY"):
            try:
                import openai
                self.client = openai.OpenAI()
            except ImportError:
                self.reply = ""
            self.reply = ""

    def ask(self, frame, question=PROMPT):
        if self.client is None or self.busy:
            return 
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return
        data = base64.standard_b64encode(buf.tobytes()).decode()
        self.busy, self.reply = True, "Thinking..."
        threading.Thread(target=self._run, args=(data, question), daemon=True).start()

    def _run(self, data, question):
        try:
            msg = self.client.messages.create(
                model=MODEL,
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image",
                         "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
                        {"type": "text", "text": question},
                    ],
                }],
            )
            self.reply = "".join(b.text for b in msg.content if b.type == "text")
            print(f"\nClaude: {self.reply}\n")
        except Exception as e:  # network, auth, rate limit...
            self.reply = f"Error: {e}"
            print(self.reply)
        finally:
            self.busy = False 
        


def draw_text(frame, text, y0=28, max_chars=70):
    """Overlay wrapped text at the bottom of the frame."""
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 > max_chars:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        lines.append(line)
    lines = lines[-4:]
    h = frame.shape[0]
    top = h - 10 - 24 * len(lines)
    cv2.rectangle(frame, (0, top - 8), (frame.shape[1], h), (0, 0, 0), -1)
    for i, ln in enumerate(lines):
        cv2.putText(frame, ln, (10, top + 18 + 24 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Is another app using it?")

    SNAP_DIR.mkdir(exist_ok=True)
    assistant = Assistant()
    mirror = True
    status, status_until = "", 0.0
    print("Camera on. s = snapshot, m = mirror, q = quit")

    while True:
        ok, frame = cap.read() 
        if not ok:
            print("Lost the camera feed.")
            break
        if mirror:
            frame = cv2.flip(frame, 1)

        raw = frame.copy()  # clean copy for snapshots and the AI
        view = frame
        cv2.putText(view, "s: snapshot  a: ask Claude  m: mirror  q: quit", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        if time.time() < status_until:
            cv2.putText(view, status, (10, 46), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 
                        0.55, (0, 220, 120), 1, cv2.LINE_AA)
        if assistant.reply:
            draw_text(view, assistant.reply)

        cv2.imshow("Webcam for my AI assistant", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or cv2.getWindowProperty("Webcam for my AI assistant", cv2.WND_PROP_VISIBLE) < 1:
            break
        elif key == ord("s"):
            path = SNAP_DIR / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(str(path), raw)
            status, status_until = f"Saved {path}", time.time() + 2.5
            print(status)
        elif key == ord("a"):
            assistant.ask(raw)
        elif key == ord("m"):
            mirror = not mirror

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()