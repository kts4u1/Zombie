"""
Procedural sound design for the 30 s shot (numpy only, 48 kHz stereo WAV).

usage: python audio.py timing.json out.wav
"""
import json
import random
import sys
import wave

import numpy as np

SR = 48000
DUR = 30.0
N = int(SR * DUR)
rng = np.random.default_rng(1234)


# ---------------------------------------------------------------- helpers
def t_axis(n):
    return np.arange(n) / SR


def fft_filter(x, lo=None, hi=None, slope=4.0):
    """smooth band-pass in the frequency domain."""
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    g = np.ones_like(f)
    if lo:
        g *= 1 / (1 + (lo / np.maximum(f, 1e-3)) ** slope)
    if hi:
        g *= 1 / (1 + (f / hi) ** slope)
    return np.fft.irfft(X * g, n)


def env_adsr(n, a, d, s, r, sustain_len):
    a_n, d_n, s_n, r_n = int(a * SR), int(d * SR), int(sustain_len * SR), int(r * SR)
    e = np.concatenate([np.linspace(0, 1, max(a_n, 1)), np.linspace(1, s, max(d_n, 1)),
                        np.full(max(s_n, 0), s), np.linspace(s, 0, max(r_n, 1))])
    out = np.zeros(n)
    out[:min(n, len(e))] = e[:n]
    return out


def place(buf, sig, t0, gain=1.0, pan=0.0):
    i = int(t0 * SR)
    if i >= N or i + len(sig) <= 0:
        return
    j0 = max(0, -i)
    i = max(0, i)
    seg = sig[j0:j0 + N - i] * gain
    l = np.cos((pan + 1) * np.pi / 4)
    r = np.sin((pan + 1) * np.pi / 4)
    buf[0, i:i + len(seg)] += seg * l
    buf[1, i:i + len(seg)] += seg * r


def noise(dur):
    return rng.standard_normal(int(dur * SR))


def formant(x, freqs, bw=1.6):
    out = np.zeros_like(x)
    for f, g in freqs:
        out += g * fft_filter(x, f / bw, f * bw, 6)
    return out


# ---------------------------------------------------------------- sounds
def footstep(heavy=1.0, dur=0.18):
    n = int(dur * SR)
    t = t_axis(n)
    thump = np.sin(2 * np.pi * (60 + 30 * heavy) * t) * np.exp(-t * 35)
    click = fft_filter(noise(dur), 300, 3500) * np.exp(-t * 90)
    grit = fft_filter(noise(dur), 1500, 7000) * np.exp(-t * 40) * 0.25
    return (thump * 0.9 * heavy + click * 0.6 + grit) * 0.7


def growl(dur, f0=55.0, bright=1.0, rough=1.0):
    n = int(dur * SR)
    t = t_axis(n)
    # vocal fry: noise amplitude-modulated by an irregular low pulse train
    jitter = np.cumsum(1 + 0.25 * rng.standard_normal(n) / 30) / SR
    pulses = (np.sin(2 * np.pi * f0 * jitter * (1 + 0.15 * np.sin(2 * np.pi * 3.1 * t))) > 0.6).astype(float)
    pulses = fft_filter(pulses, None, 900)
    src = noise(dur) * (0.35 + pulses * 1.2 * rough)
    v = formant(src, [(420, 1.0), (1100 * bright, 0.7), (2500 * bright, 0.25)])
    return np.tanh(v * 2.0) * 0.5


def shriek(dur=1.1):
    n = int(dur * SR)
    t = t_axis(n)
    f = 520 + 380 * np.minimum(1, t / 0.12) - 160 * np.maximum(0, t - 0.5) + 25 * np.sin(2 * np.pi * 9 * t)
    ph = 2 * np.pi * np.cumsum(f) / SR
    saw = (ph / np.pi) % 2 - 1
    src = saw + 0.9 * noise(dur)
    v = formant(src, [(900, 1.0), (1800, 0.8), (3200, 0.5)])
    e = env_adsr(n, 0.03, 0.15, 0.7, 0.35, dur - 0.53)
    return np.tanh(v * 3.0) * e * 0.55


def scream(dur, base=430.0):
    n = int(dur * SR)
    t = t_axis(n)
    f = base * (1 + 0.18 * np.sin(2 * np.pi * 0.7 * t + 1) + 0.04 * np.sin(2 * np.pi * 6.5 * t))
    ph = 2 * np.pi * np.cumsum(f) / SR
    src = np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.3 * np.sin(3 * ph) + 0.6 * noise(dur)
    v = formant(src, [(800, 1.0), (1250, 0.9), (2800, 0.4)], bw=1.4)
    # breaths: amplitude envelope with gaps
    am = 0.55 + 0.45 * np.sin(2 * np.pi * 0.55 * t) ** 2
    e = env_adsr(n, 0.08, 0.2, 0.8, 0.3, dur - 0.58) * am
    return np.tanh(v * 2.2) * e * 0.35


def boom(dur=2.5, f0=70, f1=28):
    n = int(dur * SR)
    t = t_axis(n)
    f = f1 + (f0 - f1) * np.exp(-t * 3)
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * np.exp(-t * 1.6)
    s += fft_filter(noise(dur), 30, 400) * np.exp(-t * 6) * 0.6
    return np.tanh(s * 1.5) * 0.9


def crunch(dur=0.35):
    n = int(dur * SR)
    t = t_axis(n)
    imp = np.zeros(n)
    idx = rng.integers(0, int(n * 0.6), 40)
    imp[idx] = rng.uniform(-1, 1, len(idx))
    c = fft_filter(imp, 900, 7000) * 6
    wet = fft_filter(noise(dur), 250, 1800) * np.exp(-t * 9)
    return (c + wet * 0.8) * np.exp(-t * 5) * 0.8


def squelch(dur=0.16, pitch=1.0):
    n = int(dur * SR)
    t = t_axis(n)
    src = noise(dur)
    f = (600 + 900 * np.exp(-t * 25)) * pitch
    v = fft_filter(src, 300 * pitch, 1800 * pitch)
    bub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 30) * 0.4
    return (v * np.exp(-t * 20) + bub) * 0.7


def tear(dur, rate=4.2):
    n = int(dur * SR)
    t = t_axis(n)
    am = np.abs(np.sin(np.pi * rate * t)) ** 3
    v = fft_filter(noise(dur), 200, 2600) * am
    rip = fft_filter(noise(dur), 1500, 6000) * (am > 0.85) * 0.5
    return (v + rip) * env_adsr(n, 0.05, 0.1, 0.9, 0.2, dur - 0.35) * 0.9


def gulp():
    d = 0.22
    t = t_axis(int(d * SR))
    f = 180 - 90 * t / d
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / d) * 0.5


def plip():
    d = 0.05
    t = t_axis(int(d * SR))
    return np.sin(2 * np.pi * (1800 - 9000 * t) * t) * np.exp(-t * 120) * 0.15


# ---------------------------------------------------------------- score
def build(timing):
    out = np.zeros((2, N))
    T_N, T_L, T_I, T_LD = timing["T_NOTICE"], timing["T_LAUNCH"], timing["T_IMPACT"], timing["T_LAND"]
    t = t_axis(N)

    # ambience: low city rumble + wind
    brown = np.cumsum(rng.standard_normal(N))
    brown = fft_filter(brown - brown.mean(), 25, 260)
    brown /= np.abs(brown).max()
    wind = fft_filter(rng.standard_normal(N), 200, 900)
    wind *= 0.4 + 0.6 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.07 * t + 1.3)) ** 2
    wind /= np.abs(wind).max()
    out += 0.10 * brown + 0.05 * wind

    # sodium lamp buzz (flickering lamp near the zombie) - same schedule as environment.py
    r = random.Random(7)
    f, on, sched = 1, True, []
    while f < 760:
        dur = r.choice([2, 3, 4, 40, 60, 90]) if on else r.choice([1, 2, 3, 5])
        sched.append((f, 1.0 if on else 0.0))
        f += dur
        on = not on
    gate = np.zeros(N)
    for (f0, v), (f1, _) in zip(sched[:-1], sched[1:]):
        gate[int((f0 - 1) / 24 * SR):int((f1 - 1) / 24 * SR)] = v
    gate = fft_filter(gate, None, 60)
    buzz = sum(np.sin(2 * np.pi * 120 * k * t) / k for k in range(1, 9))
    near = np.clip(1 - (t - 6.5) / 2.0, 0, 1)  # only near the zombie's spot
    out += 0.025 * buzz * gate * near
    crackle_t = [i / SR for i in np.nonzero(np.abs(np.diff(gate > 0.5)))[0]]
    for ct in crackle_t:
        if ct < 8:
            place(out, crunch(0.08) * 0.3, ct, 0.5)

    # drone / score: low cluster swelling into the chase
    drone = (np.sin(2 * np.pi * 36.7 * t) + 0.7 * np.sin(2 * np.pi * 55.2 * t) + 0.4 * np.sin(2 * np.pi * 73.6 * t + 1)
             + 0.25 * np.sin(2 * np.pi * 110.6 * t))
    denv = np.interp(t, [0, 4, T_N, T_N + 0.5, T_L, T_I - 0.1, T_I, 15, 26, 29, 30],
                     [0.15, 0.3, 0.35, 0.6, 0.55, 1.0, 0.2, 0.45, 0.6, 0.4, 0.0])
    out += 0.09 * drone * denv
    # tension riser during the sprint
    rise = fft_filter(rng.standard_normal(N), 400, 5000)
    rise_env = np.clip((t - T_L) / (T_I - T_L), 0, 1) ** 2 * (t < T_I)
    sweep = np.sin(2 * np.pi * np.cumsum(200 + 900 * np.clip((t - T_L) / (T_I - T_L), 0, 1) ** 2) / SR)
    out += (0.05 * rise + 0.04 * sweep) * rise_env

    # zombie idle: raspy breathing and wet growls
    for i, tb in enumerate(np.arange(0.2, T_N, 1.0 / 0.28)):
        place(out, growl(1.6, 48, 0.8, 0.5) * env_adsr(int(1.6 * SR), 0.4, 0.2, 0.7, 0.8, 0.2), tb, 0.35, -0.2)
    for tg in (1.32, 2.71, 4.55):
        place(out, growl(0.5, 70, 1.1, 1.3) * env_adsr(int(0.5 * SR), 0.01, 0.1, 0.8, 0.2, 0.2), tg, 0.5, -0.2)
    # notice: shriek + hit
    place(out, shriek(1.15), T_N + 0.02, 0.9, 0.0)
    place(out, boom(3.0, 90, 30), T_N, 0.7)
    # sprint: footsteps synced to contacts, snarls
    for ts, kind in timing["zombie_steps"]:
        if ts >= T_N:
            place(out, footstep(1.2 if kind in ("sprint", "push") else 0.7), ts, 0.9, rng.uniform(-0.2, 0.2))
    for tg in np.arange(T_L + 0.2, T_I - 0.2, 0.62):
        place(out, growl(0.45, 80, 1.2, 1.4) * env_adsr(int(0.45 * SR), 0.02, 0.1, 0.8, 0.2, 0.15), tg, 0.55)
    # victim: walking steps, gasp, scream
    for ts, kind in timing["victim_steps"]:
        if ts < T_I:
            dist = 1.0 if ts > 11.5 else 0.35
            place(out, footstep(0.4, 0.12) * dist, ts, 0.5, 0.3)
    place(out, fft_filter(noise(0.45), 500, 3000) * env_adsr(int(0.45 * SR), 0.02, 0.1, 0.6, 0.25, 0.1), timing["T_V_NOTICE"], 0.4, 0.3)
    place(out, scream(0.9, 470), 12.9, 0.8, 0.3)
    # impact & landing
    place(out, boom(1.8, 110, 40), T_I, 1.0)
    place(out, crunch(0.25) * 0.8, T_I + 0.01, 0.7)
    place(out, boom(1.2, 80, 35) * 0.8, T_LD, 0.9)
    place(out, footstep(1.6, 0.3), T_LD + 0.02, 0.8)
    # feeding: screams, bites, tears, chewing, swallows, drips
    place(out, scream(3.4, 450), 14.35, 0.75, 0.15)
    place(out, scream(2.8, 500), 17.9, 0.6, 0.15)
    place(out, scream(2.2, 410), 20.4, 0.4, 0.15)
    for tg, d in ((22.4, 1.2), (24.2, 1.3), (25.8, 1.0)):     # weakening gurgles
        g = fft_filter(noise(d), 150, 900) * (0.5 + 0.5 * np.sin(2 * np.pi * 7 * t_axis(int(d * SR))))
        place(out, g * env_adsr(int(d * SR), 0.1, 0.2, 0.6, 0.4, d - 0.7) * 0.4, tg, 0.6, 0.15)
    for tb in (15.25, 21.95):
        place(out, crunch(0.4), tb, 1.0, -0.05)
        place(out, growl(0.9, 60, 0.9, 1.5) * env_adsr(int(0.9 * SR), 0.01, 0.2, 0.7, 0.3, 0.3), tb + 0.05, 0.6)
    for t0, t1, rate in ((15.3, 16.6, 6.0), (16.6, 17.45, 4.2), (22.0, 23.0, 6.5), (24.0, 25.2, 4.0)):
        place(out, tear(t1 - t0, rate), t0, 0.75, -0.05)
    place(out, crunch(0.3) * 1.2, 17.42, 1.0)
    place(out, crunch(0.3) * 1.2, 25.18, 1.0)
    chew_f = 1.55
    for k in range(int(30 * chew_f)):
        tc = (k + 0.62) / chew_f
        if 17.6 < tc < 21.1 or tc > 25.45:
            place(out, squelch(0.17, rng.uniform(0.8, 1.2)), tc, 0.85, -0.05)
            if rng.random() < 0.5:
                place(out, crunch(0.12) * 0.5, tc + 0.02, 0.6)
            place(out, growl(0.4, 50, 0.8, 0.6) * env_adsr(int(0.4 * SR), 0.05, 0.1, 0.5, 0.2, 0.1) * 0.4, tc - 0.2, 0.5)
    for tg in (21.05, 28.6):
        place(out, gulp(), tg, 0.8)
    for k in range(40):
        tp = rng.uniform(16.0, 30.0)
        place(out, plip(), tp, rng.uniform(0.3, 0.8), rng.uniform(-0.4, 0.4))

    # master: gentle compression, fade-out, normalise
    out = np.tanh(out * 1.3) / np.tanh(1.3)
    fade = np.clip((DUR - t) / 1.0, 0, 1)
    out *= fade
    fadein = np.clip(t / 0.4, 0, 1)
    out *= fadein
    out /= np.abs(out).max() / 0.89
    return out


def write_wav(path, x):
    pcm = (np.clip(x.T, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    timing = json.load(open(sys.argv[1]))
    write_wav(sys.argv[2], build(timing))
    print("WROTE", sys.argv[2])
