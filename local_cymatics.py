import sys
from pathlib import Path

import numpy as np
import librosa

from cymatics_viz import save_cymatic_image


def tempo_to_freq(tempo_bpm: float) -> float:
    """
    Map tempo (beats per minute) to a pseudo 'frequency' in Hz for cymatic pattern.
    """
    if tempo_bpm is None or tempo_bpm <= 0:
        return 1.0
    return max(tempo_bpm / 60.0, 0.5)


def extract_features_from_file(audio_path: str):
    """
    Load a local audio file and compute simple features:
      - duration
      - tempo (BPM)
      - a few spectral stats
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    duration_sec = float(librosa.get_duration(y=y, sr=sr))

    # Tempo
    tempo_arr = librosa.beat.tempo(y=y, sr=sr, aggregate=np.median)
    tempo_bpm = float(tempo_arr[0]) if tempo_arr.size > 0 else 0.0

    # Spectral features (just for fun / future ML)
    S = np.abs(librosa.stft(y))
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr).mean()
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr).mean()
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr).mean()
    rms = librosa.feature.rms(S=S).mean()

    feature_vec = np.array(
        [tempo_bpm, centroid, bandwidth, rolloff, rms],
        dtype=float,
    )

    return feature_vec, duration_sec, tempo_bpm


def main():
    if len(sys.argv) < 2:
        print("Usage: python local_cymatics.py path/to/audio_file.(mp3|wav|flac)")
        sys.exit(1)

    audio_path = sys.argv[1]
    audio_path = str(Path(audio_path).expanduser().resolve())

    print(f"Analyzing local audio file: {audio_path}")

    features, duration_sec, tempo_bpm = extract_features_from_file(audio_path)

    print(f"  Duration: {duration_sec:.1f} s")
    print(f"  Tempo:    {tempo_bpm:.1f} BPM")

    dom_freq = tempo_to_freq(tempo_bpm)
    print(f"  Cymatic pseudo-frequency: {dom_freq:.2f} Hz")

    out_path = save_cymatic_image(dom_freq, "cymatics/local_cymatic.png")
    print(f"\nCymatic pattern saved to: {out_path}")


if __name__ == "__main__":
    main()
