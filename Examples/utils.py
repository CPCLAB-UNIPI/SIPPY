import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["axes.grid"] = True

W_V = np.logspace(-3, 4, num=701)


def create_output_dir(script, subdirs: list = []):
    script_path = os.path.abspath(os.path.dirname(script))
    script_name = os.path.splitext(os.path.basename(script))[0]
    output_dir = os.path.join(script_path, "results", script_name)
    os.makedirs(output_dir, exist_ok=True)
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    return output_dir


def plot_response_compact(
    time,
    u,
    y,
    legend=None,
    title=None,
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(time, u)
    ax.plot(time, y)
    if legend is not None:
        ax.legend(legend)
    if title is not None:
        ax.set_title(title)
    return ax


def _make_title_legend(
    u_name,
):
    if not isinstance(u_name, list):
        u_name = [u_name]
    fun_of = ", ".join(u_name)
    inputs = " + ".join([r"sys$\cdot$" + f"{u}$_k$" for u in u_name])
    legend = [" + ".join([f"{u}(t)" for u in u_name]), "y(t)"]
    return (f"Time response y$_k$({fun_of}) = " + inputs, legend)


def plot_response(
    t,
    u,
    ys,
    legends,
    titles=["Input (identification data)", "Output (identification data)"],
):
    fig, axs = plt.subplots(2, 1, sharex=True)

    axs[0].plot(t, u, legends[0])
    axs[0].set_ylabel("Input GBN")
    axs[0].set_title(titles[0])

    for y, label in zip(ys, legends[1]):
        if isinstance(y, tuple):
            y = y[0]
        if y.ndim > 1 and y.shape[0] != t.shape[0]:
            y = y.T
        axs[1].plot(t, y, label=label)
    axs[1].set_ylabel("y(t)")
    axs[1].set_title(titles[1])
    axs[1].legend()

    axs[-1].set_xlabel("Time")
    return fig


def plot_responses(
    t,
    us,
    ys,
    us_names,
):
    if not len(us) == len(ys) == len(us_names):
        raise ValueError("All inputs must have the same length")
    fig, axs = plt.subplots(len(us), 1, sharex=True)
    for u, y, u_name, ax in zip(us, ys, us_names, axs):
        title, legend = _make_title_legend(u_name)
        ax = plot_response_compact(t, u, y, legend, title, ax)

    axs[-1].set_xlabel("Time")
    return fig


def plot_bode(om, mags, fis, legends):
    fig, axs = plt.subplots(2, 1, sharex=True)
    for mag in mags:
        axs[0].loglog(om, mag)
    axs[0].set_ylabel("Amplitude Ratio")
    axs[0].set_title("Bode Plot")

    for fi in fis:
        axs[1].semilogx(om, fi)
    axs[1].set_ylabel("Phase")
    axs[1].legend(legends)

    axs[-1].set_xlabel("w")
    return fig


def plot_comparison(t, Us, ylabels, legend=None, title=None):
    if not isinstance(Us, list):
        Us = [Us]
    m = Us[0].shape[0]
    fig, axs = plt.subplots(m, 1, sharex=True)
    if not isinstance(axs, np.ndarray):
        axs = [axs]
    for i in range(m):
        for U in Us:
            axs[i].plot(t, U[i, :])
        # axs[i].plot(t, U[i, :])
        axs[i].set_ylabel(ylabels[i])
        # axs[i].set_ylim([0.95 * np.amin(U[i, :]), 1.05 * np.amax(U[i, :])])

    if title is not None:
        axs[0].set_title(title)
    if legend is not None:
        axs[-1].legend(legend)
    axs[-1].set_xlabel("Time")
    return fig
