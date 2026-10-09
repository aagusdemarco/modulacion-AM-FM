from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import hilbert, resample_poly, welch
from scipy.special import jv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parent.parent
ASSETS = RAIZ / "assets"
FIGS = ASSETS / "figs"
AUDIO = ASSETS / "audio"

FS = 400_000
FS_AUDIO = 48_000
DURACION = 5.0

AC = 100.0
AM_TONO = 50.0
F_TONO = 440.0
FM_AUDIO = 5_000.0
A_AUDIO = 50.0

FC_TONO = 20_000.0
FC_AUDIO_AM = 50_000.0
FC_AUDIO_ANG = 80_000.0

C_AZUL, C_NARANJA, C_AQUA, C_AMARILLO, C_MAGENTA = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
C_TEXTO, C_GRIS = "#333333", "#8a8a8a"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": "#b0b0b0",
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.2, "legend.frameon": False, "legend.fontsize": 8,
    "text.color": C_TEXTO, "axes.labelcolor": C_TEXTO,
    "xtick.color": C_TEXTO, "ytick.color": C_TEXTO,
})


def eje_t(dur: float, fs: float = FS) -> np.ndarray:
    return np.arange(int(round(dur * fs))) / fs


def potencia(x: np.ndarray) -> float:
    return float(np.mean(np.asarray(x) ** 2))


def db(x: float) -> float:
    return 10 * np.log10(x)


def coma(v: float, d: int = 1) -> str:
    return f"{v:.{d}f}".replace(".", ",")


def filtro_ideal(x, fs, f1, f2):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    X[(f < f1) | (f > f2)] = 0
    return np.fft.irfft(X, len(x))


def derivador(x, fs):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    return np.fft.irfft(1j * 2 * np.pi * f * X, len(x))


def integrador(x, fs):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    H = np.zeros_like(f, dtype=complex)
    H[1:] = 1 / (1j * 2 * np.pi * f[1:])
    return np.fft.irfft(H * X, len(x))


def envolvente(x):
    return np.abs(hilbert(x))


def mod_am(m, t, fc, Ac=AC):
    return (Ac + m) * np.cos(2 * np.pi * fc * t)


def mod_fm(m, t, fc, fd, Ac=AC, fs=FS):
    fase = 2 * np.pi * fd * integrador(m, fs)
    return Ac * np.cos(2 * np.pi * fc * t + fase)


def mod_pm(m, t, fc, kp, Ac=AC):
    return Ac * np.cos(2 * np.pi * fc * t + kp * m)


def demod_am(r, fs, fm):
    y = filtro_ideal(envolvente(r), fs, 0, fm)
    return y - y.mean()


def discriminador(r, fs, fc):
    lim = r / np.maximum(envolvente(r), 1e-12)
    E = envolvente(derivador(lim, fs))
    return E / (2 * np.pi) - fc


def demod_fm(r, fs, fc, fd, fm):
    return filtro_ideal(discriminador(r, fs, fc), fs, 0, fm) / fd


def demod_pm(r, fs, fc, kp, fm):
    df = discriminador(r, fs, fc)
    y = 2 * np.pi * integrador(df, fs) / kp
    y = filtro_ideal(y, fs, 0, fm)
    return y - y.mean()


def mensaje_tono(t):
    return AM_TONO * np.cos(2 * np.pi * F_TONO * t)


def mensaje_audio(ruta=ASSETS / "imperial_march.wav"):
    x, fs0 = sf.read(ruta)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x[: int(DURACION * fs0)]
    x = x - x.mean()
    x = filtro_ideal(x, fs0, 0, FM_AUDIO)
    g = np.gcd(int(FS), int(fs0))
    x = resample_poly(x, FS // g, fs0 // g)[: int(DURACION * FS)]
    x = x - x.mean()
    return A_AUDIO * x / np.max(np.abs(x))


def a_wav(nombre, y, fs=FS):
    g = np.gcd(int(fs), FS_AUDIO)
    z = resample_poly(y, FS_AUDIO // g, int(fs) // g)
    z = 0.9 * z / np.max(np.abs(z))
    AUDIO.mkdir(parents=True, exist_ok=True)
    sf.write(AUDIO / f"{nombre}.wav", z.astype(np.float32), FS_AUDIO)


def fidelidad_db(m, y, fs=FS, borde=0.05):
    k = int(borde * fs)
    m, y = m[k:-k], y[k:-k]
    return db(potencia(m) / potencia(m - y))


def lineas(x, fs):
    N = len(x)
    X = np.fft.rfft(x) / N
    f = np.fft.rfftfreq(N, 1 / fs)
    return f, np.abs(X)


def dep(x, fs, nperseg=2 ** 15):
    f, S = welch(x, fs, nperseg=nperseg, window="hann")
    return f, S


def ancho_banda_potencia(x, fs, fc, frac=0.98):
    X = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / fs)
    total = X.sum()
    d = np.abs(f - fc)
    orden = np.argsort(d)
    acum = np.cumsum(X[orden]) / total
    return 2 * d[orden][np.searchsorted(acum, frac)]


def disenio_fdm(m2):
    dmax = np.max(np.abs(np.gradient(m2, 1 / FS)))
    kp_audio = 0.1
    df_pm = kp_audio * dmax / (2 * np.pi)
    BA = FM_AUDIO
    d = {
        "AM":    {"tono": {"fc": FC_TONO, "BT": 2 * F_TONO},
                  "audio": {"fc": FC_AUDIO_AM, "BT": 2 * BA}},
        "FM_BA": {"tono": {"fc": FC_TONO, "fd": 0.2 * F_TONO / AM_TONO, "BT": 2 * 1.2 * F_TONO},
                  "audio": {"fc": FC_AUDIO_ANG, "fd": 0.2 * BA / A_AUDIO, "BT": 2 * 1.2 * BA}},
        "FM_BW": {"tono": {"fc": FC_TONO, "fd": 5 * F_TONO / AM_TONO, "BT": 2 * 6 * F_TONO},
                  "audio": {"fc": FC_AUDIO_ANG, "fd": 5 * BA / A_AUDIO, "BT": 2 * 6 * BA}},
        "PM":    {"tono": {"fc": FC_TONO, "kp": 5 / AM_TONO, "BT": 2 * 6 * F_TONO},
                  "audio": {"fc": FC_AUDIO_ANG, "kp": kp_audio, "BT": 2 * (df_pm + BA),
                            "df": df_pm}},
    }
    return d


def modular(esq, p, m, t):
    if esq == "AM":
        return mod_am(m, t, p["fc"])
    if esq.startswith("FM"):
        return mod_fm(m, t, p["fc"], p["fd"])
    return mod_pm(m, t, p["fc"], p["kp"])


def demodular(esq, p, r, fm):
    W = 1.02 * fm
    if esq == "AM":
        return demod_am(r, FS, W)
    if esq.startswith("FM"):
        return demod_fm(r, FS, p["fc"], p["fd"], W)
    return demod_pm(r, FS, p["fc"], p["kp"], W)



def so_no_teorica(esq, p, si_ni_db, Pm, fm):
    if esq == "AM":
        g = 2 * Pm / (AC ** 2 + Pm)
    elif esq.startswith("FM"):
        g = 3 * p["BT"] * p["fd"] ** 2 * Pm / fm ** 3
    else:
        g = p["kp"] ** 2 * Pm * p["BT"] / fm
    return si_ni_db + db(g), g


NOMBRES = {"AM": "AM", "FM_BA": "FM banda angosta", "FM_BW": "FM banda ancha", "PM": "PM"}


def figuras_tono(res, t, m1, x1s):
    fc = FC_TONO
    x_am = x1s["AM"]
    Pm = potencia(m1)
    f, L = lineas(x_am, FS)
    k0 = np.argmin(np.abs(f - fc))
    PT = potencia(x_am)
    Pc_med = 2 * L[k0] ** 2
    E = envolvente(x_am)
    res["tono"] = {"Pm": Pm, "a": AM_TONO / AC, "AM": {
        "PT_teo": AC ** 2 / 2 + Pm / 2, "PT_med": PT, "Pc_med": Pc_med, "PB_med": PT - Pc_med,
        "E_teo": 100 * Pm / (AC ** 2 + Pm), "E_med": 100 * (PT - Pc_med) / PT, "BT": 2 * F_TONO,
        "linea_portadora": L[k0], "linea_lateral": L[np.argmin(np.abs(f - (fc + F_TONO)))],
        "a_med": (E.max() - E.min()) / (E.max() + E.min())}}
    betas = {"FM_BA": 0.2, "FM_BW": 5.0, "PM": 5.0}
    for k, beta in betas.items():
        f, L = lineas(x1s[k], FS)
        k0 = np.argmin(np.abs(f - fc))
        PT = potencia(x1s[k])
        res["tono"][k] = {"beta": beta, "df": beta * F_TONO, "PT_med": PT,
                          "Pc_teo": AC ** 2 / 2 * jv(0, beta) ** 2, "Pc_med": 2 * L[k0] ** 2,
                          "PB_med": PT - 2 * L[k0] ** 2, "BT_carson": 2 * (beta + 1) * F_TONO,
                          "B98_med": ancho_banda_potencia(x1s[k], FS, fc, 0.98)}

    fig, ax = plt.subplots(2, 1, figsize=(7, 4.2), sharex=True)
    sel = t < 5e-3
    ax[0].plot(t[sel] * 1e3, m1[sel], color=C_AZUL)
    ax[0].set_ylabel("m(t) [V]")
    ax[0].set_title("Mensaje del canal 1: tono de 440 Hz, Am = 50 V")
    ax[1].plot(t[sel] * 1e3, x_am[sel], color=C_AZUL, lw=0.5)
    ax[1].plot(t[sel] * 1e3, AC + m1[sel], color=C_NARANJA, lw=1.6)
    ax[1].plot(t[sel] * 1e3, -(AC + m1[sel]), color=C_NARANJA, lw=1.6)
    ax[1].set_ylabel("x_AM(t) [V]")
    ax[1].set_xlabel("t [ms]")
    ax[1].set_title("Señal AM (a = 0,5; fc = 20 kHz) y envolvente Ac + m(t)")
    fig.tight_layout(); fig.savefig(FIGS / "am_tono_tiempo.png"); plt.close(fig)

    f, L = lineas(x_am, FS)
    sel = np.abs(f - fc) < 1500
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.vlines(f[sel] / 1e3, 0, L[sel], color=C_AZUL, lw=2)
    for fx, lab in [(fc, "Ac/2 = 50 V"), (fc - F_TONO, "Am/4 = 12,5 V"), (fc + F_TONO, "Am/4 = 12,5 V")]:
        k = np.argmin(np.abs(f - fx))
        ax.annotate(lab, (fx / 1e3, L[k]), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8)
    ax.set_xlabel("f [kHz]")
    ax.set_ylabel("|X_AM(f)|  (bilateral) [V]")
    ax.set_ylim(0, 60)
    ax.set_title("Espectro AM del tono: portadora y bandas laterales en fc ± 440 Hz (B = 880 Hz)")
    fig.tight_layout(); fig.savefig(FIGS / "am_tono_espectro.png"); plt.close(fig)

    fig, axs = plt.subplots(3, 1, figsize=(7, 6.4), sharex=True)
    casos = [("FM banda angosta, β = 0,2 (fd = 1,76 Hz/V)", "FM_BA"),
             ("FM banda ancha, β = 5 (fd = 44 Hz/V)", "FM_BW"),
             ("PM, β = kp·Am = 5 (kp = 0,1 rad/V)", "PM")]
    for ax, (tit, k) in zip(axs, casos):
        beta = betas[k]
        f, L = lineas(x1s[k], FS)
        nmax = 1 if beta < 0.5 else 9
        sel = np.abs(f - fc) <= nmax * F_TONO + 1
        ax.vlines((f[sel] - fc) / 1e3, 0, L[sel], color=C_AZUL, lw=2, label="simulación")
        n = np.arange(-nmax, nmax + 1)
        ax.plot(n * F_TONO / 1e3, AC / 2 * np.abs(jv(n, beta)), "o", mfc="none",
                mec=C_NARANJA, ms=6, mew=1.2, label="(Ac/2)|Jn(β)|")
        BT = 2 * (beta + 1) * F_TONO if beta > 0.5 else 2 * F_TONO
        ax.axvspan(-BT / 2e3, BT / 2e3, color=C_AQUA, alpha=0.12, lw=0)
        ax.text(BT / 2e3, 45, f"  B_T = {BT:.0f} Hz" + (" (2fm)" if beta < 0.5 else " (Carson)"),
                fontsize=8, va="top")
        ax.set_title(tit)
        ax.set_ylabel("|X(f)| [V]")
        ax.set_ylim(0, 55)
        ax.set_xlim(-4.5, 4.5)
    axs[0].legend(loc="upper left")
    axs[-1].set_xlabel("f − fc [kHz]")
    fig.tight_layout(); fig.savefig(FIGS / "angular_tono_espectros.png"); plt.close(fig)


def experimento(res, eta=1e-3):
    t = eje_t(DURACION)
    m1 = mensaje_tono(t)
    m2 = mensaje_audio()
    a_wav("tono_original", m1)
    a_wav("audio_original", m2)
    d = disenio_fdm(m2)
    Pm1, Pm2 = potencia(m1), potencia(m2)
    res["audio"] = {"Pm": Pm2, "max": float(np.max(m2)), "min": float(np.min(m2)),
                    "a": float(-np.min(m2) / AC), "eta": eta,
                    "E_AM": 100 * Pm2 / (AC ** 2 + Pm2),
                    "dmax": float(np.max(np.abs(np.gradient(m2, 1 / FS))))}
    res["fdm"] = {}
    rng = np.random.default_rng(1)
    ruido = rng.normal(0, np.sqrt(eta / 2 * FS), len(t))

    fig, axs = plt.subplots(1, 2, figsize=(8, 2.8))
    axs[0].plot(t, m2, color=C_AZUL, lw=0.4)
    axs[0].set_xlabel("t [s]"); axs[0].set_ylabel("m2(t) [V]")
    axs[0].set_title("Marcha Imperial (5 s, mono, |m|max = 50 V)")
    f, S = dep(m2, FS)
    sel = f < 7000
    axs[1].semilogy(f[sel] / 1e3, S[sel], color=C_AZUL)
    axs[1].axvline(FM_AUDIO / 1e3, color=C_GRIS, ls=":", lw=0.8)
    axs[1].set_xlabel("f [kHz]"); axs[1].set_ylabel("DEP [V²/Hz]")
    axs[1].set_title("Espectro del audio (limitado a 5 kHz)")
    fig.tight_layout(); fig.savefig(FIGS / "audio_mensaje.png"); plt.close(fig)

    figE, axE = plt.subplots(4, 1, figsize=(7.5, 8), sharex=True)
    figD, axD = plt.subplots(4, 1, figsize=(7.5, 7), sharex=True)
    ventana = (t > 1.20) & (t < 1.26)
    titulos = {"AM": "AM", "FM_BA": "FM banda angosta (Δ = 0,2)",
               "FM_BW": "FM banda ancha (Δ = 5)", "PM": "PM (kp = 0,1 rad/V)"}
    x1s = {}
    for i, esq in enumerate(["AM", "FM_BA", "FM_BW", "PM"]):
        p1, p2 = d[esq]["tono"], d[esq]["audio"]
        x1, x2 = modular(esq, p1, m1, t), modular(esq, p2, m2, t)
        x1s[esq] = x1
        x = x1 + x2
        r = x + ruido
        out = {}
        for canal, p, xi, mi, fm, Pm in [("tono", p1, x1, m1, F_TONO, Pm1),
                                         ("audio", p2, x2, m2, FM_AUDIO, Pm2)]:
            lo, hi = p["fc"] - p["BT"] / 2, p["fc"] + p["BT"] / 2
            ri = filtro_ideal(r, FS, lo, hi)
            si = filtro_ideal(x, FS, lo, hi)
            ni = ri - si
            y_limpia = demodular(esq, p, si, fm)
            y = demodular(esq, p, ri, fm)
            SiNi = db(potencia(si) / potencia(ni))
            teo, g = so_no_teorica(esq, p, SiNi, Pm, fm)
            out[canal] = {
                **{k: float(v) for k, v in p.items()},
                "PT": potencia(xi),
                "B98": ancho_banda_potencia(xi, FS, p["fc"], 0.98) if esq != "AM" else p["BT"],
                "SiNi_dB": SiNi,
                "SoNo_dB": db(potencia(y_limpia) / potencia(y - y_limpia)),
                "SoNo_teo_dB": teo, "ganancia_teo": g,
                "fid_sin_ruido_dB": fidelidad_db(mi, y_limpia),
                "fid_con_ruido_dB": fidelidad_db(mi, y),
            }
            out[canal]["ganancia_med"] = 10 ** ((out[canal]["SoNo_dB"] - SiNi) / 10)
            a_wav(f"{canal}_{esq.lower()}_demodulado", y)
            if canal == "audio":
                axD[i].plot(t[ventana] * 1e3, mi[ventana], color=C_GRIS, ls="--", lw=1.2, label="original")
                axD[i].plot(t[ventana] * 1e3, y[ventana], color=C_AZUL, lw=1.0, label="demodulada")
                axD[i].set_title(f"{titulos[esq]}  —  S_o/N_o = {coma(out[canal]['SoNo_dB'])} dB")
                axD[i].set_ylabel("[V]")
        r2 = filtro_ideal(x2, FS, p1["fc"] - p1["BT"] / 2, p1["fc"] + p1["BT"] / 2)
        out["diafonia_audio_en_tono_W"] = potencia(r2)
        out["BT_total"] = (p2["fc"] + p2["BT"] / 2) - (p1["fc"] - p1["BT"] / 2)
        out["guarda"] = (p2["fc"] - p2["BT"] / 2) - (p1["fc"] + p1["BT"] / 2)
        res["fdm"][esq] = out

        f, S = dep(r, FS)
        sel = f < 160e3
        axE[i].semilogy(f[sel] / 1e3, S[sel], color=C_AZUL, lw=0.7)
        for p, lab in [(p1, "canal 1: tono"), (p2, "canal 2: audio")]:
            axE[i].axvspan((p["fc"] - p["BT"] / 2) / 1e3, (p["fc"] + p["BT"] / 2) / 1e3,
                           color=C_NARANJA, alpha=0.12, lw=0)
            axE[i].text(p["fc"] / 1e3, S[sel].max() * 3, lab, ha="center", fontsize=7)
        axE[i].set_ylim(eta / 20, S[sel].max() * 30)
        axE[i].set_title(titulos[esq] + f"  —  ancho de banda FDM ≈ {coma(out['BT_total']/1e3)} kHz")
        axE[i].set_ylabel("DEP unilat. [W/Hz]")
    axE[-1].set_xlabel("f [kHz]")
    figE.suptitle(f"Señal multicanal recibida (2 canales FDM + ruido blanco, η = {coma(eta, 3)} W/Hz)", fontsize=10)
    figE.tight_layout(); figE.savefig(FIGS / "fdm_espectros.png"); plt.close(figE)
    h, l = axD[0].get_legend_handles_labels()
    figD.legend(h, l, loc="lower center", ncols=2)
    axD[-1].set_xlabel("t [ms]")
    figD.suptitle("Canal de audio demodulado (fragmento de 60 ms)", fontsize=10)
    figD.tight_layout(rect=(0, 0.04, 1, 1)); figD.savefig(FIGS / "fdm_audio_demodulado.png"); plt.close(figD)

    res["audio"]["PT_AM"] = res["fdm"]["AM"]["audio"]["PT"]
    figuras_tono(res, t, m1, x1s)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    res = {}
    experimento(res)
    with open(ASSETS / "resultados.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False, default=float)
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=1, default=float, ensure_ascii=False))
