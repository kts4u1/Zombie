"""
Sound design for the 30 s shot, mixed from CC0 field/foley recordings (Freesound,
see scripts/sounds.json) and synced to the animation timeline.

usage: python audio.py timing.json sounds_wav_dir out.wav

Every event is placed from the story beats in timing.json (foot contacts, notice,
leap, impact, landing, bite/tear/rip/chew/swallow times), so the sound follows the
picture.  Levels follow the camera: louder in close-ups, distant in wide shots.
"""
import json
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import voice as V

SR = 48000
DUR = 30.0
N = int(SR * DUR)
FPS = 24
rng = np.random.default_rng(7)

# shot cuts (seconds) and how close the camera is to the zombie / the victim (dB)
SHOTS = [  # start, zombie_db, victim_db
    (0.00, 0, -30), (4.17, -3, -9), (5.50, 1, -30), (6.79, -2, -30), (8.17, -3, -26),
    (10.58, -3, -22), (12.00, None, 0), (13.17, -3, -3), (14.58, -2, -2), (17.58, 0, -1),
    (21.58, -3, -3), (25.42, 1, 0), (28.42, -11, -11)]


# ---------------------------------------------------------------- dsp helpers
def fft_filter(x, lo=None, hi=None, slope=4.0):
    n = len(x)
    if n < 8:
        return x
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    g = np.ones_like(f)
    if lo:
        g *= 1 / (1 + (lo / np.maximum(f, 1e-3)) ** slope)
    if hi:
        g *= 1 / (1 + (f / hi) ** slope)
    return np.fft.irfft(X * g, n)


def resample(x, ratio):
    """ratio > 1: higher pitch / shorter."""
    n = int(len(x) / ratio)
    return np.interp(np.arange(n) * ratio, np.arange(len(x)), x)


def fade(x, a=0.005, b=0.005):
    x = x.copy()
    na, nb = min(len(x), int(a * SR)), min(len(x), int(b * SR))
    if na:
        x[:na] *= np.linspace(0, 1, na)
    if nb:
        x[-nb:] *= np.linspace(1, 0, nb)
    return x


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)


def norm(x, target=-20.0):
    """normalise the loud part of a sound to a reference RMS (dBFS)."""
    if len(x) == 0:
        return x
    n = max(1, int(0.05 * SR))
    e = np.sqrt(np.convolve(x ** 2, np.ones(n) / n, mode="same"))
    loud = x[e > np.percentile(e, 60)] if len(x) > n else x
    return x * 10 ** ((target - rms_db(loud)) / 20)


def env_db(x, win=0.005):
    n = max(1, int(win * SR))
    return 20 * np.log10(np.sqrt(np.convolve(x ** 2, np.ones(n) / n, mode="same")) + 1e-9)


def onsets(x, rise=12.0, min_gap=0.12, floor=-50):
    hop = int(0.005 * SR)
    ee = env_db(x)[::hop]
    out, last = [], -1e9
    for i in range(10, len(ee)):
        t = i * hop / SR
        if ee[i] > floor and ee[i] - ee[i - 8:i].min() > rise and t - last > min_gap:
            out.append(t)
            last = t
    return out


def shot_at(t):
    s = SHOTS[0]
    for sh in SHOTS:
        if t >= sh[0]:
            s = sh
    return s


def prox(t, who="z"):
    s = shot_at(t)
    v = s[1] if who == "z" else s[2]
    if v is None:          # S07: the zombie rushes in from far away toward the victim cam
        u = np.clip((t - 12.0) / 1.17, 0, 1)
        v = -15 + 12 * u ** 1.6
    return v


# ---------------------------------------------------------------- library
class Lib:
    def __init__(self, d):
        self.d = d
        self.cache = {}

    def __call__(self, sid, a=0.0, b=None, mono=True):
        if sid not in self.cache:
            with wave.open(os.path.join(self.d, f"{sid}.wav")) as w:
                ch, sr, n = w.getnchannels(), w.getframerate(), w.getnframes()
                x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float64) / 32768
                x = x.reshape(-1, ch).T
                assert sr == SR, (sid, sr)
            self.cache[sid] = x
        x = self.cache[sid]
        i0 = int(a * SR)
        i1 = x.shape[1] if b is None else min(x.shape[1], int(b * SR))
        y = x[:, i0:i1]
        return y.mean(axis=0) if mono else y


# ---------------------------------------------------------------- mixer
class Mix:
    def __init__(self):
        self.buf = np.zeros((2, N))
        self.rev_send = np.zeros(N)

    def place(self, x, t0, db=0.0, pan=0.0, rev=0.15, who=None):
        """mono sound at time t0 (s); who='z'/'v' applies the camera-distance gain."""
        if who:
            db += prox(t0, who)
        i = int(round(t0 * SR))
        if i >= N or len(x) == 0:
            return
        j0 = max(0, -i)
        i = max(0, i)
        seg = x[j0:j0 + N - i] * 10 ** (db / 20)
        l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        self.buf[0, i:i + len(seg)] += seg * l * 1.41
        self.buf[1, i:i + len(seg)] += seg * r * 1.41
        self.rev_send[i:i + len(seg)] += seg * rev

    def place_track(self, x, t0, db=0.0, pan=0.0, rev=0.15, who="v"):
        """long mono track whose level follows the camera distance shot by shot (10 ms crossfades)."""
        tt = t0 + np.arange(len(x)) / SR
        g = np.array([prox(t, who) for t in tt[::240]]).repeat(240)[: len(x)]
        k = int(0.01 * SR)
        g = np.convolve(np.pad(g, (k, k), mode="edge"), np.ones(k) / k, mode="same")[k:-k]
        self.place(x * 10 ** (g / 20), t0, db, pan, rev)

    def bed(self, xs, t0, t1, db, fin=0.5, fout=0.5, gain_env=None):
        """stereo (2,n) bed looped/cut to [t0, t1] with fades; optional gain envelope fn(t)."""
        n = int((t1 - t0) * SR)
        if xs.ndim == 1:
            xs = np.vstack([xs, xs])
        reps = int(np.ceil(n / xs.shape[1])) + 1
        y = np.tile(xs, reps)[:, :n].copy()
        e = np.ones(n)
        a, b = int(fin * SR), int(fout * SR)
        if a:
            e[:a] *= np.linspace(0, 1, a)
        if b:
            e[-b:] *= np.linspace(1, 0, b)
        if gain_env is not None:
            tt = t0 + np.arange(n) / SR
            e *= 10 ** (np.vectorize(gain_env)(tt[::480]).repeat(480)[:n] / 20)
        i = int(t0 * SR)
        m = min(n, N - i)
        self.buf[:, i:i + m] += y[:, :m] * e[:m] * 10 ** (db / 20)

    def master(self):
        # short street reflections (synthetic IR) to glue the dry foley together
        ir_n = int(0.6 * SR)
        t = np.arange(ir_n) / SR
        ir = rng.standard_normal(ir_n) * np.exp(-t / 0.13)
        ir[: int(0.012 * SR)] = 0
        ir = fft_filter(ir, 250, 5000)
        ir /= np.sqrt(np.sum(ir ** 2))
        wet = np.fft.irfft(np.fft.rfft(self.rev_send, N + ir_n) * np.fft.rfft(ir, N + ir_n), N + ir_n)[:N]
        out = self.buf + np.vstack([wet, np.roll(wet, int(0.011 * SR))]) * 0.5
        out = fft_filter(out[0], 28, None, 6), fft_filter(out[1], 28, None, 6)
        out = np.vstack(out)
        # final fade with the picture
        tt = np.arange(N) / SR
        out *= np.clip(tt / 0.4, 0, 1) * np.clip((30.0 - tt) / 0.9, 0, 1) ** 1.5
        # loudness: bring the active programme to about -19 dBFS RMS, then limit the peaks
        act = np.max(np.abs(out), axis=0) > 1e-3
        out *= 10 ** ((-19.0 - rms_db(out[:, act])) / 20)
        peak = np.max(np.abs(out), axis=0)
        thr, ceil = 0.5, 0.95
        # soft-knee transfer: above thr, compress smoothly toward the ceiling
        over = np.maximum(peak - thr, 0)
        target = np.where(peak > thr, thr + (ceil - thr) * (1 - np.exp(-over / (ceil - thr))), peak)
        g_raw = target / np.maximum(peak, 1e-9)
        # look-ahead: the gain must already be down when a peak arrives (min filter),
        # then smooth it so it never 'clicks'
        k = int(0.003 * SR)
        g = g_raw.copy()
        for sft in range(1, k + 1):
            g[:-sft] = np.minimum(g[:-sft], g_raw[sft:])
            g[sft:] = np.minimum(g[sft:], g_raw[:-sft])
        g = np.convolve(g, np.ones(k) / k, mode="same")
        out *= g
        return np.clip(out, -0.97, 0.97)


# ---------------------------------------------------------------- the soundtrack
def build(T, L):
    M = Mix()
    T_N, T_L, T_LEAP, T_I, T_LD = T["T_NOTICE"], T["T_LAUNCH"], T["T_LEAP"], T["T_IMPACT"], T["T_LAND"]
    BITES, SWALLOWS, T_SHOCK = T["BITES"], T["SWALLOWS"], T["T_SHOCK"]
    end_scream = T_SHOCK

    # ---- city at night: distant traffic + air tone, steady through the piece
    M.bed(L("479561", 10, 50, mono=False), 0, 30, -24, 1.0, 1.0)
    M.bed(L("237329", 5, 40, mono=False), 0, 30, -27, 1.0, 1.0)
    # the failing sodium lamp above the zombie: buzz + flicker crackle (close in S01-S03)
    lamp_db = lambda t: 0.0 if t < 6.8 else -9.0 if t < 13.2 else -14.0
    M.bed(L("500693", 0, 21.5, mono=False), 0, 30, -27, 0.5, 1.5, gain_env=lamp_db)
    M.bed(L("125064", 2.5, 22, mono=False), 0, 30, -33, 0.5, 1.5, gain_env=lamp_db)

    # ---- music: low suspense drone, building to the attack, darker under the feeding
    drone = L("130982", 0.5, 22.5, mono=False)
    M.bed(drone, 0, 14.0, -22, 2.0, 0.8,
          gain_env=lambda t: -8 + 8 * np.clip((t - 4.5) / 9.0, 0, 1) ** 2)
    M.bed(L("155142", 3, 60, mono=False), 14.0, 30, -31, 3.0, 1.0)
    # heartbeat while the victim realises what is coming
    hb = norm(L("22416"), -20)
    for k, t0 in enumerate(np.arange(11.95, 13.55, 0.44)):
        M.place(fft_filter(hb[: int(0.42 * SR)], None, 400), t0, -9 + 2 * k, rev=0.0)

    # ---- zombie idle: wet, rattling breathing; twitches crack and grunt
    br1 = norm(L("426229"), -20)
    br2 = norm(L("426409"), -20)
    # (stops dead when he notices the prey)
    M.place(fade(resample(br1, 0.88)[: int((T_N - 0.2) * SR)], 0.05, 0.12), 0.15, -8, 0.05, who="z")
    M.place(fade(resample(br2, 0.9)[: int((T_N - 0.7) * SR)], 0.05, 0.12), 0.6, -11, -0.05, who="z")
    snaps = norm(L("574752"), -20)
    so = onsets(snaps, rise=14)
    grunt_src = norm(L("125405"), -20)
    for k, tw in enumerate([1.32, 2.71, 3.38, 4.55, 5.02]):
        a = so[(3 * k) % len(so)]
        M.place(fade(fft_filter(snaps[int(a * SR) - 200: int((a + 0.12) * SR)], 300, 6000), 0.002, 0.04),
                tw, -21, 0.1, rev=0.1, who="z")
        g0 = [2.0, 7.4, 12.1, 18.6, 25.3][k]
        M.place(fade(resample(grunt_src[int(g0 * SR): int((g0 + 0.5) * SR)], 0.92), 0.02, 0.15),
                tw + 0.02, -16, 0, who="z")

    # ---- notice: sharp intake of breath, then the shriek as the jaw drops
    gasp = norm(L("344407", 0.12, 0.6), -20)
    M.place(fade(resample(gasp, 0.7), 0.01, 0.06), T_N - 0.04, -12, who="z")
    shriek = norm(L("435651", 0.03, 2.2), -18)
    shriek = fade(shriek[: int(1.25 * SR)], 0.01, 0.35)
    M.place(shriek, T_N + 0.10, -2, rev=0.35, who="z")
    roar = norm(L("245223", 0.25, 1.6), -20)
    M.place(fade(fft_filter(roar, None, 1500), 0.03, 0.4), T_N + 0.12, -9, rev=0.3, who="z")

    # ---- sprint: running steps on pavement at every foot contact (one step sliced per contact) over
    # the rustle of clothes in motion - isolated hard clicks in a fast rhythm read as hooves, this does not
    run = norm(L("547276"), -20)
    steps = []
    for t in onsets(run, rise=12, min_gap=0.18):
        s_ = run[int((t - 0.01) * SR): int((t + 0.22) * SR)]
        if len(s_) > 0.18 * SR and rms_db(s_) > -40:
            steps.append(norm(fade(s_, 0.002, 0.09), -20))
    zsteps = [t for t, k in T["zombie_steps"] if t >= T_L - 0.6]
    for k, ts in enumerate(zsteps):
        s_ = steps[(k * 7) % len(steps)]
        spd = np.clip((ts - T_L) / 1.6, 0.3, 1.0)
        body = fft_filter(resample(s_, 0.95 * rng.uniform(0.97, 1.03)), 60, 6000)
        thump = np.zeros_like(body)
        lp = fft_filter(s_, None, 150) * 1.2
        thump[: len(lp)] = lp[: len(body)]
        M.place(body + thump, ts + rng.uniform(-0.004, 0.004), -6 + 4 * spd + rng.uniform(-1.5, 1.5), rev=0.25, who="z")
    cloth_run = norm(L("611276"), -20)
    n_run = int((T_LEAP - T_L + 0.15) * SR)
    M.place(fade(cloth_run[int(2 * SR): int(2 * SR) + n_run], 0.25, 0.1), T_L, -12, rev=0.1, who="z")
    # ragged, snarling breath while running
    pant = norm(L("394941", 1.0, 30.0), -20)
    pant = resample(pant, 1.25)
    i0 = int(0.3 * SR)
    M.place(fade(pant[i0: i0 + int((T_LEAP - T_L + 0.2) * SR)], 0.2, 0.15), T_L, -15, rev=0.2, who="z")
    snarl = norm(L("133974", 0.27, 2.5), -20)
    for ts, a, b in ((8.9, 0.0, 1.1), (11.2, 0.5, 1.6)):
        M.place(fade(snarl[int(a * SR): int(b * SR)], 0.04, 0.3), ts, -12, rev=0.25, who="z")

    # ---- victim: wet-sneaker steps while walking, gasp, panic
    walk = L("138265")
    wst = []
    for t in onsets(walk, rise=8, min_gap=0.3):
        s = walk[int((t - 0.01) * SR): int((t + 0.25) * SR)]
        if len(s) > 0.2 * SR:
            wst.append(norm(fade(s, 0.003, 0.1), -20))
    for k, (ts, kind) in enumerate(T["victim_steps"]):
        if ts < T_I:
            M.place(wst[(k * 5) % len(wst)], ts, -16, pan=0.0, rev=0.2, who="v")

    # ---- leap & tackle
    roar2 = norm(L("179345", 0.15, 0.86), -18)
    M.place(fade(roar2, 0.01, 0.15), T_LEAP - 0.12, -4, rev=0.35, who="z")
    hit = norm(L("675930", 0.0, 0.45), -16)
    M.place(fade(hit, 0.001, 0.15), T_I - 0.01, -1, rev=0.25)
    boom = norm(L("415278", 0.0, 1.55), -16)
    M.place(fade(fft_filter(boom, None, 2500), 0.002, 0.6), T_I - 0.01, -8, rev=0.0)
    rustle = norm(L("235243"), -20)
    M.place(fade(rustle[int(0.4 * SR): int(1.3 * SR)], 0.01, 0.3), T_I - 0.02, -10, rev=0.15, who="v")
    fall = norm(L("504626", 0.3, 1.5), -16)
    M.place(fade(fall, 0.003, 0.3), T_LD - 0.12, 0, rev=0.25)
    fall2 = norm(L("675930", 0.78, 1.7), -16)
    M.place(fade(fall2, 0.003, 0.3), T_LD - 0.02, -3, rev=0.25)
    phone = norm(L("254158", 0.98, 1.6), -18)
    M.place(fade(phone, 0.002, 0.1), T_I + 0.47, -9, pan=0.3, rev=0.2)

    # ---- feeding
    eat_bed = norm(L("675434", 2.0, 80.0), -20)      # low growl + wet eating (the animal)
    juicy = norm(L("234237", 0.2, 17.0), -20)
    bites_src = norm(L("155973"), -18)
    bite_ev = [(0.0, 1.6), (1.85, 3.2), (3.25, 4.7), (4.75, 7.0), (7.15, 9.3), (9.35, 10.7), (10.75, 12.7)]
    tear_src = norm(L("211621"), -20)
    rip_src = norm(L("234236"), -20)
    slime = norm(L("211605"), -20)
    riptear = norm(L("30931"), -16)
    gush = norm(L("345985", 0.0, 2.2), -18)
    crunch = norm(L("249811"), -18)
    cr_on = onsets(crunch, rise=12, min_gap=0.15)
    chews = L("482823")
    ch_on = [t for t in onsets(chews, rise=10, min_gap=0.2) if t < 13.0]
    chew_sl = [norm(fade(chews[int((t - 0.01) * SR): int((t + 0.26) * SR)], 0.002, 0.08), -20) for t in ch_on]
    wet = L("655707")
    wet_on = onsets(wet, rise=10, min_gap=0.25)
    wet_sl = [norm(fade(wet[int((t - 0.01) * SR): int((t + 0.3) * SR)], 0.002, 0.1), -20) for t in wet_on]
    gr_eat = norm(L("448089"), -20)
    gr_eat2 = norm(L("236331"), -20)
    swallow = norm(L("368711", 0.15, 1.1), -18)

    # growling, wet bed under all of the eating (follows the camera distance)
    feed_db = lambda t: prox(t, "z") + (2 if t < 17.0 else 0)
    M.bed(np.vstack([eat_bed, eat_bed]), T_LD + 0.05, 30.0, -19, 0.4, 1.0, gain_env=feed_db)
    for bi, (l0, cl, t0, t1, rip, ce) in enumerate(BITES):
        # clamp: teeth sink in (bite + bone crunch + snarl)
        a, b = bite_ev[bi * 2 % len(bite_ev)]
        M.place(fade(bites_src[int(a * SR): int(b * SR)], 0.002, 0.3), cl - 0.03, -2, rev=0.15, who="z")
        c0 = cr_on[bi * 4 % len(cr_on)]
        M.place(fade(crunch[int((c0 - 0.01) * SR): int((c0 + 0.35) * SR)], 0.002, 0.1), cl, -6, rev=0.1, who="z")
        g0 = [3.2, 21.4][bi]
        M.place(fade(resample(grunt_src[int(g0 * SR): int((g0 + 1.2) * SR)], 0.9), 0.03, 0.3), cl + 0.05, -9, who="z")
        # worrying the wound: wet squelching
        n = int((t0 - cl) * SR)
        M.place(fade(slime[int(1.0 * SR): int(1.0 * SR) + n], 0.1, 0.2), cl + 0.1, -10, rev=0.1, who="z")
        # violent tearing while the head shakes (4.2 Hz), sinew and gristle giving way
        n = int((rip - t0 + 0.1) * SR)
        tt = np.arange(n) / SR
        shake_am = 0.45 + 0.55 * np.abs(np.sin(2 * np.pi * 4.2 * tt))
        src = rip_src[int((3.0 + 17 * bi) * SR): int((3.0 + 17 * bi) * SR) + n]
        src = src * shake_am[: len(src)]
        M.place(fade(src, 0.08, 0.05), t0, -3, rev=0.15, who="z")
        src2 = tear_src[int((2.0 + 8 * bi) * SR): int((2.0 + 8 * bi) * SR) + n]
        M.place(fade(src2, 0.08, 0.05), t0, -6, rev=0.1, who="z")
        for k in range(3):
            c = cr_on[(bi * 4 + 1 + k) % len(cr_on)]
            M.place(fade(crunch[int((c - 0.01) * SR): int((c + 0.25) * SR)], 0.002, 0.08),
                    t0 + 0.25 + k * (rip - t0 - 0.3) / 3, -8, rev=0.1, who="z")
        # the rip: flesh tears free, blood sprays
        M.place(fade(riptear, 0.002, 0.25), rip - 0.06, 0, rev=0.2, who="z")
        a, b = bite_ev[(bi * 2 + 1) % len(bite_ev)]
        M.place(fade(bites_src[int(a * SR): int((a + 0.7) * SR)], 0.002, 0.3), rip - 0.02, -5, who="z")
        M.place(fade(gush, 0.01, 0.6), rip, -8, rev=0.1, who="z")
        # chewing: a wet crunch every time the jaw slams shut (1.55 Hz cycle)
        n_c = 0
        tc = (np.floor((rip + 0.3) * 1.55) + 0.88) / 1.55
        while tc < min(ce - 0.15, 30.0):
            if tc > rip + 0.45:
                s1 = chew_sl[(n_c * 3 + bi) % len(chew_sl)]
                s2 = wet_sl[(n_c * 5 + 2 * bi) % len(wet_sl)]
                M.place(s1, tc - 0.02, -1 + rng.uniform(-1.5, 1.5), rev=0.1, who="z")
                M.place(s2, tc, -4 + rng.uniform(-2, 2), rev=0.1, who="z")
                if n_c % 3 == 1:     # gristle crackles now and then
                    c = cr_on[(n_c + bi) % len(cr_on)]
                    M.place(fade(crunch[int((c - 0.01) * SR): int((c + 0.2) * SR)], 0.002, 0.06), tc + 0.03,
                            -9, who="z")
                n_c += 1
            tc += 1 / 1.55
        # smacking / slurping bed + low satisfied growls while chewing
        c_a, c_b = rip + 0.4, min(ce, 30.0)
        n = int((c_b - c_a) * SR)
        M.place(fade(juicy[int(5.0 * bi * SR): int(5.0 * bi * SR) + n], 0.3, 0.3), c_a, -10, rev=0.1, who="z")
        g = gr_eat if bi == 0 else gr_eat2
        M.place(fade(fft_filter(g, None, 2500), 0.2, 0.4), c_a + 0.6, -14, who="z")
        M.place(fade(fft_filter(g, None, 2500), 0.2, 0.4), c_a + 2.6, -16, who="z")
    for ts in SWALLOWS:
        if ts < 30:
            M.place(fade(swallow, 0.01, 0.2), ts - 0.18, -4, rev=0.05, who="z")

    # ---- the victim's voice: one actor, cut on his breaths, on the story beats (scripts/voice.py)
    vox, spans = V.render(T, L.d)
    M.place_track(vox, 0.0, +2, rev=0.28, who="v")
    # ---- struggle foley, on the same bursts that drive his body (voice.bursts)
    env, _ = V.envelope(vox, spans)
    grab = norm(L("237047"), -20)
    grab_on = [t for t in onsets(grab, rise=10, min_gap=0.35) if t < 36]
    scrape = norm(L("329514"), -20)
    scr_on = onsets(scrape, rise=9, min_gap=0.4)
    scuff = norm(L("437946"), -20)
    scuff_on = [t for t in onsets(scuff, rise=12, min_gap=0.3) if t < 33]
    slap = norm(L("812359"), -18)
    for k, (tb, side, a) in enumerate(V.bursts(env, T)):
        g0 = grab_on[(k * 3) % len(grab_on)]
        M.place(fade(grab[int((g0 - 0.02) * SR): int((g0 + 0.45) * SR)], 0.01, 0.15), tb - 0.02, -12, pan=0.15 * side, rev=0.1, who="v")
        s0 = scr_on[k % len(scr_on)]
        M.place(fade(fft_filter(scrape[int((s0 - 0.02) * SR): int((s0 + 0.5) * SR)], 120, 6000), 0.01, 0.2), tb + 0.08, -12, pan=-0.25 * side, rev=0.12, who="v")
        if k % 2 == 0:
            c0 = scuff_on[(k * 5) % len(scuff_on)]
            M.place(fade(fft_filter(scuff[int((c0 - 0.01) * SR): int((c0 + 0.18) * SR)], None, 3000), 0.003, 0.08), tb + 0.42, -14, rev=0.1, who="v")
        if k % 3 == 0:
            M.place(fade(slap, 0.002, 0.1), tb, -15, rev=0.05, who="v")
    # his hand finds the wound: a wet press
    M.place(fade(slime[int(4.0 * SR): int(4.6 * SR)], 0.05, 0.3), 17.75, -17, who="v")
    return M.master()


def write_wav(path, x):
    y = (np.clip(x, -1, 1) * 32767).astype(np.int16).T.copy()
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(y.tobytes())


if __name__ == "__main__":
    timing = json.load(open(sys.argv[1]))
    lib = Lib(sys.argv[2])
    out = build(timing, lib)
    write_wav(sys.argv[3], out)
    print("WROTE", sys.argv[3], out.shape)
