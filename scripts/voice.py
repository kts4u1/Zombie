"""
The victim's voice, built from ONE actor's continuous performance so the screams,
sobs and groans all come from the same throat:

  530471  "male scream pain torture" (martian, CC0)       - startle, terror, agony, sobbing
  675807  "Wheezing of dying man who's been shot" (craigsmith, CC0) - shock / dying breaths

Pieces are cut at the natural pauses of the recording (silence between breaths) and
placed on the story beats from timing.json.  The same plan produces a per-frame
envelope (24 fps) and a per-frame 'kind' used by the animation for lip sync, facial
expression and struggle effort, so the face and body move with the voice.

usage: python voice.py timing.json sounds_wav_dir out_voice.json [out_voice.wav]

Sound choice was checked with an AudioSet tagger and CLAP text matching (windows that
read as crow / bleat / baby cry / laughter / speech were avoided).
"""
import json
import os
import sys
import wave

import numpy as np

SR = 48000
DUR = 30.0
N = int(SR * DUR)
FPS = 24
NFRAMES = 720

VOICE = "530471"
DYING = "675807"


def plan(T):
    """list of (t, src, a, b, gain_db, fade_in, fade_out, lowpass_hz, kind, cut_at)."""
    TN, TI, TL = T["T_V_NOTICE"], T["T_IMPACT"], T["T_LAND"]
    l0, cl, t0, t1, rip, ce = T["BITES"][0]
    l0b, clb, t0b, t1b, ripb, ceb = T["BITES"][1]
    TS = T["T_SHOCK"]
    P = [
        # looks up from the phone: startled, voiced gasp ("ah-!")
        (TN + 0.04, VOICE, 0.46, 0.86, -7, 0.004, 0.06, 5000, "gasp", None),
        # frozen: two quick panicked inhales
        (TN + 0.47, VOICE, 5.28, 5.56, +4, 0.02, 0.05, 6000, "breath", None),
        (TN + 0.70, VOICE, 10.70, 11.06, +6, 0.02, 0.05, 6000, "breath", None),
        # backing away as it closes in: terror scream - cut dead by the impact
        (TI - 0.70, VOICE, 4.38, 5.27, -1, 0.02, 0.03, None, "scream", TI + 0.02),
        # back hits the asphalt: the wind knocked out of him
        (TL + 0.02, VOICE, 13.68, 13.82, -6, 0.004, 0.04, 3500, "grunt", None),
        # the teeth sink in: agony, fighting it off
        (cl + 0.06, VOICE, 1.12, 1.88, 0, 0.01, 0.05, None, "scream", None),
        (cl + 0.84, VOICE, 1.87, 2.13, +2, 0.01, 0.03, 6000, "breath", None),
        (cl + 1.10, VOICE, 11.05, 12.40, 0, 0.01, 0.05, None, "scream", rip - 0.12),
        # the flesh tears away: the peak, choked off into shock
        (rip - 0.10, VOICE, 12.48, 13.10, +1, 0.01, 0.10, None, "scream", None),
        # shock: wet, wheezing gasps of a man bleeding out
        (TS + 0.42, DYING, 3.20, 6.00, -6, 0.04, 0.25, 6500, "wheeze", None),
        (TS + 3.30, DYING, 12.00, 13.35, -9, 0.05, 0.30, 6000, "wheeze", clb - 0.02),
        # second bite: barely a cry left
        (clb + 0.06, VOICE, 26.33, 27.27, -9, 0.04, 0.30, 2600, "moan", None),
        (clb + 1.55, VOICE, 27.82, 28.38, -13, 0.04, 0.30, 2200, "moan", None),
        # last breaths
        (ripb + 0.35, DYING, 14.45, 15.70, -14, 0.05, 0.6, 4500, "wheeze", None),
    ]
    return P


# ------------------------------------------------------------------ dsp
def load(d, sid):
    with wave.open(os.path.join(d, f"{sid}.wav")) as w:
        ch, sr, n = w.getnchannels(), w.getframerate(), w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float64) / 32768
    assert sr == SR
    return x.reshape(-1, ch).mean(axis=1)


def lowpass(x, hz, slope=4.0):
    if not hz or len(x) < 8:
        return x
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return np.fft.irfft(X / (1 + (f / hz) ** slope), len(x))


def fade(x, a, b):
    x = x.copy()
    na, nb = min(len(x), int(a * SR)), min(len(x), int(b * SR))
    if na:
        x[:na] *= np.linspace(0, 1, na) ** 2
    if nb:
        x[-nb:] *= np.linspace(1, 0, nb) ** 2
    return x


def ref_level(x):
    """RMS of the loud part (dB) - used to put both sources on one scale."""
    n = int(0.05 * SR)
    e = np.sqrt(np.convolve(x ** 2, np.ones(n) / n, mode="same"))
    loud = x[e > np.percentile(e, 70)]
    return 20 * np.log10(np.sqrt(np.mean(loud ** 2)) + 1e-12)


def render(T, wavdir):
    """mono voice stem (N samples), list of (t0, t1, kind) spans."""
    src = {s: load(wavdir, s) for s in (VOICE, DYING)}
    # put both recordings on a common loudness reference (-20 dB loud-part RMS)
    for s in src:
        src[s] = src[s] * 10 ** ((-20 - ref_level(src[s])) / 20)
    out = np.zeros(N)
    spans = []
    for (t, s, a, b, g, fi, fo, lp, kind, cut) in plan(T):
        x = src[s][int(a * SR): int(b * SR)]
        if cut is not None:
            x = x[: max(1, int((cut - t) * SR))]
        x = fade(lowpass(x, lp), fi, fo) * 10 ** (g / 20)
        i = int(round(t * SR))
        if i >= N:
            continue
        m = min(len(x), N - i)
        out[i:i + m] += x[:m]
        spans.append((t, t + m / SR, kind))
    return out, spans


def envelope(x, spans):
    """per-frame mouth/effort envelope 0..1 and kind (mouth leads the sound by ~1 frame)."""
    hop = SR // 200                       # 5 ms
    n = len(x) // hop
    e = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1))
    db = 20 * np.log10(e + 1e-9)
    raw = np.clip((db + 46.0) / 30.0, 0, 1)          # -46 dB closed .. -16 dB wide open
    # jaw dynamics: fast to open, slower to close
    sm = np.zeros_like(raw)
    a_up, a_dn = 1 - np.exp(-1 / (0.025 * 200)), 1 - np.exp(-1 / (0.09 * 200))
    for i in range(1, n):
        k = a_up if raw[i] > sm[i - 1] else a_dn
        sm[i] = sm[i - 1] + k * (raw[i] - sm[i - 1])
    env, kinds = [], []
    for f in range(1, NFRAMES + 1):
        t = (f - 1) / FPS + 1.0 / FPS                 # lead: show the mouth one frame early
        i0, i1 = int((t - 0.02) * 200), int((t + 0.02) * 200)
        env.append(float(sm[max(0, i0):max(i0 + 1, i1)].max()) if i0 < n else 0.0)
        kd = "none"
        for (a, b, k) in spans:
            if a - 0.06 <= t <= b + 0.04:
                kd = k
        kinds.append(kd)
    return env, kinds


def bursts(env, T, fps=FPS):
    """struggle bursts: every rise of the voice during the fight (+ a few fixed ones).
    Shared by the animation (body effort) and the mix (foley), so they line up.
    returns [(t, side +-1, strength)]."""
    TL, cl, rip = T["T_LAND"], T["BITES"][0][1], T["BITES"][0][4]
    out, armed = [], True
    for i in range(1, len(env)):
        t = i / fps
        if not (TL - 0.05 < t < rip + 0.05):
            continue
        if armed and env[i] > 0.42:
            out.append(t)
            armed = False
        elif env[i] < 0.25:
            armed = True
    for t in (cl + 0.42, cl + 0.95, 15.75, 16.15, 16.45):
        if all(abs(t - b) > 0.28 for b in out):
            out.append(t)
    out.sort()
    return [(t, 1 if k % 2 == 0 else -1, 1.0 if t < rip - 0.6 else 1.15) for k, t in enumerate(out)]


def write_wav(path, x):
    y = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(y.tobytes())


if __name__ == "__main__":
    T = json.load(open(sys.argv[1]))
    x, spans = render(T, sys.argv[2])
    env, kinds = envelope(x, spans)
    json.dump({"fps": FPS, "env": env, "kind": kinds, "spans": spans}, open(sys.argv[3], "w"))
    if len(sys.argv) > 4:
        write_wav(sys.argv[4], x * 0.5)
    print("WROTE", sys.argv[3], "voiced frames:", sum(1 for v in env if v > 0.15))
