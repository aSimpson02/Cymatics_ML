import os
import uuid
import subprocess
import shutil
from pathlib import Path

import numpy as np           
import librosa
from flask import Flask, request, url_for, render_template_string, jsonify

from cymatics_viz import save_cymatic_image




# Config

BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
CYMATIC_FOLDER = BASE_DIR / "static" / "cymatics"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
CYMATIC_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".webm"}

app = Flask(
    __name__,
    static_folder=str(BASE_DIR / "static"),
)



# Helpers

def allowed_file(filename: str) -> bool:
    return "." in filename and Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def tempo_to_freq(tempo_bpm: float) -> float:
    """Convert BPM -> frequency(Hz) for cymatics pattern."""
    if tempo_bpm is None or tempo_bpm <= 0:
        return 1.0
    return max(tempo_bpm / 60.0, 0.5)


def analyze_audio_file(path: Path):
    """Extract tempo + duration from WAV/MP3/OGG etc."""
    print(f"[librosa] Loading audio: {path}")
    y, sr = librosa.load(str(path), sr=None, mono=True)

    duration_sec = float(librosa.get_duration(y=y, sr=sr))

    try:
        tempo_arr = librosa.beat.tempo(y=y, sr=sr, aggregate=None)
        tempo_bpm = float(np.median(tempo_arr)) if tempo_arr.size > 0 else 0.0
    except Exception as e:
        print(f"[librosa] Tempo detection failed: {e}")
        tempo_bpm = 0.0

    freq_hz = tempo_to_freq(tempo_bpm)

    print(f"[analysis] duration={duration_sec:.2f}s tempo={tempo_bpm:.2f} freq={freq_hz:.2f}Hz")
    return duration_sec, tempo_bpm, freq_hz



def ffmpeg_convert_to_wav(src: Path, dst: Path):
    """
    Convert ANY audio file (webm/ogg/m4a/mp3/etc) to WAV.
    Always attempts the actual installed Homebrew ffmpeg path first.

    This avoids PATH issues inside Python/Flask.
    """


    ffmpeg_bin = Path("/opt/homebrew/bin/ffmpeg")


    if not ffmpeg_bin.exists():
        print("[ffmpeg] Homebrew ffmpeg not found. Trying PATH...")
        found = shutil.which("ffmpeg")
        if found is None:
            raise RuntimeError(
                "ffmpeg not found on system. Install it or ensure it's in PATH."
            )
        ffmpeg_bin = Path(found)

    print(f"[ffmpeg] Using binary: {ffmpeg_bin}")
    print(f"[ffmpeg] Converting {src} -> {dst}")

    cmd = [
        str(ffmpeg_bin),
        "-y",
        "-i", str(src),
        "-ac", "1",
        "-ar", "44100",
        str(dst),
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("[ffmpeg] Conversion OK")
    except subprocess.CalledProcessError as e:
        print("[ffmpeg] Conversion FAILED")
        raise RuntimeError(f"ffmpeg conversion failed: {e}") from e



# HTML template

PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Cymatics Live</title>
    <style>
      body {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: #050816;
        color: #f5f5f5;
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 40px 16px;
      }
      .wrapper {
        display: grid;
        grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr);
        gap: 24px;
        max-width: 1080px;
        width: 100%;
      }
      .card {
        background: radial-gradient(circle at top left, #1e293b 0, #020617 55%);
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 18px 30px rgba(0,0,0,0.4);
      }
      h1 {
        margin-top: 0;
        margin-bottom: 8px;
        font-size: 28px;
      }
      h2 {
        margin-top: 0;
        margin-bottom: 8px;
        font-size: 20px;
      }
      p {
        margin-top: 4px;
        margin-bottom: 12px;
        color: #c3c7e6;
      }
      form {
        margin-top: 16px;
        margin-bottom: 16px;
      }
      input[type="file"] {
        margin-bottom: 12px;
      }
      button {
        background: #4f46e5;
        border: none;
        color: white;
        padding: 9px 16px;
        border-radius: 999px;
        cursor: pointer;
        font-size: 14px;
      }
      button:hover {
        background: #4338ca;
      }
      button.secondary {
        background: #0f172a;
        border: 1px solid #1d283a;
      }
      button.secondary:hover {
        background: #111827;
      }
      .stats {
        margin-top: 12px;
        font-size: 14px;
      }
      .image-wrapper {
        margin-top: 18px;
        text-align: center;
      }
      img {
        max-width: 100%;
        border-radius: 16px;
        border: 1px solid #272b3f;
      }
      .footer {
        margin-top: 24px;
        font-size: 12px;
        color: #7c81a8;
        text-align: center;
      }
      .pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 11px;
        background: rgba(79,70,229,0.12);
        border-radius: 999px;
        padding: 4px 10px;
        margin-bottom: 6px;
        color: #a5b4fc;
      }
      .status {
        font-size: 13px;
        margin-top: 8px;
        color: #e5e7eb;
      }
      .status.error { color: #f97373; }
      .status.ok { color: #4ade80; }
      code {
        background: #020617;
        padding: 2px 5px;
        border-radius: 4px;
        font-size: 11px;
      }
      @media (max-width: 900px) {
        .wrapper { grid-template-columns: minmax(0,1fr); }
      }
    </style>
  </head>
  <body>

    <div class="wrapper">

      <!-- Upload mode -->
      <div class="card">
        <div class="pill">Mode 1 · Upload track</div>
        <h1>Cymatics Live</h1>
        <p>Upload an audio file and generate a cymatic pattern.</p>

        <form method="post" enctype="multipart/form-data">
          <input type="file" name="audio" accept=".mp3,.wav,.flac,.ogg,.m4a,.webm" required>
          <br>
          <button type="submit">Generate from file</button>
        </form>

        {% if error %}
          <p style="color:#f97373;">{{ error }}</p>
        {% endif %}

        {% if image_url %}
          <div class="stats">
            <div>Duration: {{ duration_sec | round(1) }} s</div>
            <div>Tempo: {{ tempo_bpm | round(1) }} BPM</div>
            <div>Cymatic freq: {{ freq_hz | round(2) }} Hz</div>
          </div>

          <div class="image-wrapper">
            <img src="{{ image_url }}" alt="Cymatic pattern">
          </div>
        {% endif %}
      </div>

      <!-- Live recording -->
      <div class="card">
        <div class="pill">Mode 2 · Live mic</div>
        <h2>Live recording → Cymatics</h2>
        <p>Record audio from your microphone and generate a cymatic pattern.</p>

        <button id="recordBtn" class="secondary">🎙 Start recording</button>
        <button id="stopBtn" disabled>⏹ Stop</button>

        <div id="recordStatus" class="status">Idle.</div>

        <div id="liveStats" class="stats" style="display:none;">
          <div>Duration: <span id="liveDuration"></span> s</div>
          <div>Tempo: <span id="liveTempo"></span> BPM</div>
          <div>Cymatic freq: <span id="liveFreq"></span> Hz</div>
        </div>

        <div class="image-wrapper" id="liveImageWrap" style="display:none;">
          <img id="liveImage" src="" alt="Live cymatic pattern">
        </div>

        <p style="margin-top:16px;font-size:12px;color:#9ca3af;">
          Mic access works at <code>http://localhost</code>.  
        </p>
      </div>

    </div>

    <div class="footer">Local analysis only — no APIs.</div>

    <script>
      let mediaRecorder = null;
      let audioChunks = [];

      const recordBtn = document.getElementById("recordBtn");
      const stopBtn = document.getElementById("stopBtn");
      const statusEl = document.getElementById("recordStatus");
      const liveStats = document.getElementById("liveStats");
      const liveDuration = document.getElementById("liveDuration");
      const liveTempo = document.getElementById("liveTempo");
      const liveFreq = document.getElementById("liveFreq");
      const liveImageWrap = document.getElementById("liveImageWrap");
      const liveImage = document.getElementById("liveImage");

      async function startRecording() {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

          let options = {};
          if (MediaRecorder.isTypeSupported("audio/ogg")) {
            options = { mimeType: "audio/ogg" };
          } else if (MediaRecorder.isTypeSupported("audio/webm")) {
            options = { mimeType: "audio/webm" };
          }

          mediaRecorder = new MediaRecorder(stream, options);
          audioChunks = [];

          mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) audioChunks.push(e.data);
          };

          mediaRecorder.onstop = async () => {
            statusEl.textContent = "Processing...";
            statusEl.className = "status";

            const type = mediaRecorder.mimeType || "audio/webm";
            const blob = new Blob(audioChunks, { type });

            const formData = new FormData();
            formData.append("audio_data", blob, type.includes("ogg") ? "rec.ogg" : "rec.webm");

            try {
              const resp = await fetch("/upload_recording", { method: "POST", body: formData });
              if (!resp.ok) throw new Error("Server error " + resp.status);

              const data = await resp.json();
              if (data.error) {
                statusEl.textContent = "Error: " + data.error;
                statusEl.className = "status error";
                return;
              }

              statusEl.textContent = "Done.";
              statusEl.className = "status ok";

              liveStats.style.display = "block";
              liveImageWrap.style.display = "block";
              liveDuration.textContent = data.duration_sec.toFixed(1);
              liveTempo.textContent = data.tempo_bpm.toFixed(1);
              liveFreq.textContent = data.freq_hz.toFixed(2);
              liveImage.src = data.image_url + "?t=" + Date.now();

            } catch (err) {
              statusEl.textContent = "Error: " + err.message;
              statusEl.className = "status error";
            }
          };

          mediaRecorder.start();
          statusEl.textContent = "Recording...";
          statusEl.className = "status ok";
          recordBtn.disabled = true;
          stopBtn.disabled = false;

        } catch (err) {
          statusEl.textContent = "Mic error: " + err.message;
          statusEl.className = "status error";
        }
      }

      function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== "inactive") {
          mediaRecorder.stop();
        }
        recordBtn.disabled = false;
        stopBtn.disabled = true;
      }

      recordBtn.addEventListener("click", startRecording);
      stopBtn.addEventListener("click", stopRecording);
    </script>

  </body>
</html>
"""



# Routes

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    image_url = None
    duration_sec = None
    tempo_bpm = None
    freq_hz = None

    # Upload track mode
    if request.method == "POST" and "audio" in request.files:
        file = request.files.get("audio")
        if not file or file.filename == "":
            error = "No file selected."
        elif not allowed_file(file.filename):
            error = "Unsupported file type."
        else:
            ext = Path(file.filename).suffix.lower()
            uid = uuid.uuid4().hex

            upload_path = UPLOAD_FOLDER / f"upload_{uid}{ext}"
            file.save(str(upload_path))

            # Convert to WAV if needed
            if ext != ".wav":
                wav_path = UPLOAD_FOLDER / f"upload_{uid}.wav"
                try:
                    ffmpeg_convert_to_wav(upload_path, wav_path)
                    audio_path = wav_path
                except Exception as e:
                    error = f"Conversion failed: {e}"
                    return render_template_string(PAGE_TEMPLATE, error=error)

            else:
                audio_path = upload_path

            try:
                duration_sec, tempo_bpm, freq_hz = analyze_audio_file(audio_path)
            except Exception as e:
                error = f"Failed to analyze audio: {e}"
            else:
                img_filename = f"cymatic_{uid}.png"
                img_path = CYMATIC_FOLDER / img_filename
                save_cymatic_image(freq_hz, str(img_path))
                image_url = url_for("static", filename=f"cymatics/{img_filename}")

    return render_template_string(
        PAGE_TEMPLATE,
        error=error,
        duration_sec=duration_sec,
        tempo_bpm=tempo_bpm,
        freq_hz=freq_hz,
        image_url=image_url,
    )


@app.route("/upload_recording", methods=["POST"])
def upload_recording():
    file = request.files.get("audio_data")
    if not file:
        return jsonify({"error": "Missing audio_data"}), 400

    original_name = file.filename or "rec.webm"
    ext = Path(original_name).suffix.lower() or ".webm"

    uid = uuid.uuid4().hex
    raw_path = UPLOAD_FOLDER / f"recording_{uid}{ext}"
    file.save(str(raw_path))

    wav_path = UPLOAD_FOLDER / f"recording_{uid}.wav"

    try:
        ffmpeg_convert_to_wav(raw_path, wav_path)
        duration_sec, tempo_bpm, freq_hz = analyze_audio_file(wav_path)
    except Exception as e:
        return jsonify({"error": f"Failed to process recording: {e}"}), 500

    img_filename = f"cymatic_recording_{uid}.png"
    img_path = CYMATIC_FOLDER / img_filename
    save_cymatic_image(freq_hz, str(img_path))

    image_url = url_for("static", filename=f"cymatics/{img_filename}")

    return jsonify(
        {
            "duration_sec": duration_sec,
            "tempo_bpm": tempo_bpm,
            "freq_hz": freq_hz,
            "image_url": image_url,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
