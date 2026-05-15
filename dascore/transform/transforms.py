"""Additional domain transforms for patches."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from dascore.constants import PatchType
from dascore.utils.patch import patch_function
from dascore.utils.time import to_float


@patch_function()
def stalta(
    patch: PatchType,
    short: float | None = None,
    long: float | None = None,
    absolute: bool = True,
    dim: str = "time",
) -> PatchType:
    """
    Compute the short-term / long-term average (STA/LTA) ratio along a patch dimension.

    Parameters
    ----------
    patch
        The input DASCore patch.
    short
        Short window length in seconds. If None, it defaults to 20 samples are assumed
    long
        Long window length in seconds. If None it defaults to 5*shortwindow
    absolute
        Uses the absolute value of the signal (default=True)
    dim
        Dimension along which to compute the STA/LTA ratio. Defaults to ``"time"``.

    Returns
    -------
    PatchType
        A new patch containing the STA/LTA ratio.

    """
    # these default values are heuristically derived and may not work in all cases
    if short is None:
        step = to_float(patch.get_coord(dim).step)
        short = 20 * step
    if long is None:
        long = 5 * short

    if absolute:
        patch = patch.abs()

    short_patch = patch.rolling(time=short).mean()
    long_patch = patch.rolling(time=long).mean()

    return (short_patch / long_patch).update(
        attrs={"data_type": "STALTA", "data_units": ""}
    )


@patch_function(required_dims=("time",), history="full")
def fbe_rms(
    patch: PatchType,
    corners: tuple[float, float] = (None, None),
    time: float | None = None,
    step: float | None = None,
    db: bool = True,
) -> PatchType:
    """
    Compute the rolling Root-Mean-Squared (RMS) of the Energy in a Frequency Band (FBE),
    Also commonly called the 'waterfall plot' in DAS-processing.
    This implementation is a wrapper to DAScore functionality:
        1) Apply a 'pass_filter' to the patch
        2) Apply rolling-funtction of window length "time"
        3) Calculate RMS value for each channel


    Parameters
    ----------
    patch
        Input DASCore patch.
    corners
        Two-element tuple with frequencies to calculate energy in.
    time
        window length in which to caluclate engergy.
    step
        time-step for rolling window. Defaults to sampling rate. This can be
        used for coarser sampling of the resulting patch
    db
        Return patch data in decibel [dB] instead of orginal units

    Returns
    -------
    PatchType
        A new patch containing FBE-RMS traces.
    """
    if step is None:
        step = to_float(patch.get_coord("time").step)

    if time is None:
        time = 20 * step

    if any(x is not None for x in corners):
        patch = patch.pass_filter(time=corners)

    fbe = ((patch**2).rolling(time=time, step=step).mean() ** 0.5).update(
        attrs={"data_type": "FBE_RMS"}
    )

    if db:
        fbe = (10 * fbe.log10()).update(
            attrs={"data_type": "FBE_RMS", "data_units": "dB"}
        )

    return fbe


@patch_function(required_dims=("time",), history="full")
def rolling_mean_frequency(
    patch: PatchType,
    winlen: float,
    step: float | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
) -> PatchType:
    """
    Compute rolling mean frequency along a dimension using one batched FFT pass.

    Parameters
    ----------
    patch
        Input DASCore patch.
    winlen
        Window length in seconds.
    step
        Step between windows in seconds. Defaults to `winlen`.
    fmin
        Optional lower frequency bound in Hz.
    fmax
        Optional upper frequency bound in Hz.

    Returns
    -------
    PatchType
        Patch containing rolling mean frequency. The output retains all
        non-reduced dimensions and replaces the input `dim` coordinate
        with window-center coordinates.
    """
    dim = "time"

    patch_t = patch.transpose(dim, ...)
    data = np.asarray(patch_t.data, dtype=float)

    coord = patch_t.get_coord(dim)
    dt = to_float(coord.step)
    if dt is None or dt <= 0:
        raise ValueError(
            f"Coordinate step for dim={dim!r} must be defined and positive."
        )

    if step is None:
        step = winlen

    nwin = int(round(winlen / dt))
    nstep = int(round(step / dt))

    if nwin < 2:
        raise ValueError("winlen is too small for the sampling interval.")
    if nstep < 1:
        raise ValueError("step is too small for the sampling interval.")
    if data.shape[0] < nwin:
        raise ValueError("Patch is shorter than the requested window length.")

    # shape: (n_windows_possible, ..., nwin) after sliding over axis 0
    win_view = sliding_window_view(data, window_shape=nwin, axis=0)

    # move window axis to position 1: (n_windows_possible, nwin, ...)
    win_view = np.moveaxis(win_view, -1, 1)

    # apply step
    win_view = win_view[::nstep]

    # FFT over window axis
    fft_vals = np.fft.rfft(win_view, axis=1)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(nwin, d=dt)

    # optional frequency limits
    mask = np.ones_like(freqs, dtype=bool)
    if fmin is not None:
        mask &= freqs >= fmin
    if fmax is not None:
        mask &= freqs <= fmax
    if not np.any(mask):
        raise ValueError("Frequency limits exclude all FFT bins.")

    freqs = freqs[mask]
    power = power[:, mask, ...]

    # weighted mean over frequency axis
    reshape = (1, freqs.size) + (1,) * (power.ndim - 2)
    freqs_b = freqs.reshape(reshape)

    numerator = np.sum(freqs_b * power, axis=1)
    denominator = np.sum(power, axis=1)

    mean_freq = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )

    # mean_freq shape: (n_windows, ...)
    # build window-center coordinates
    time_vals = patch_t.get_array(dim)
    starts = np.arange(0, data.shape[0] - nwin + 1, nstep)
    centers = starts + nwin // 2
    new_time = time_vals[centers]

    out = patch_t.new(
        data=mean_freq,
        coords={"time": new_time, "distance": patch.coords.get_array("distance")},
    )
    # patch.new(data=data, coords=new_coords, attrs=attrs)
    return out.transpose(*[d for d in patch.dims if d in out.dims]).update(
        attrs={"data_type": "Mean Frequency", "data_units": "Hz"}
    )


@patch_function()
def kurtosis(
    patch: PatchType,
    winlen: float,
    dim: str = "time",
    recursive: bool = True,
) -> PatchType:
    """
    Compute kurtosis along a patch dimension. Note that for best results
    normalize data, or convert to nano-strain(rate)


    Parameters
    ----------
    patch
        Input DASCore patch.
    winlen
        Window length in seconds.
    dim
        Dimension along which to compute kurtosis. Defaults to ``"time"``.
    recursive
        If True, use recursive pseudo-kurtosis. Otherwise use moving-window
        Pearson kurtosis.
        See Langet et. al (2014). Continuous Kurtosis-Based Migration for
        Seismic Event Detection and Location, with Application to Piton de la
        Fournaise Volcano, La Reunion. BSSA. 104. 229-246. 10.1785/0120130107.

    Returns
    -------
    PatchType
        A new patch with kurtosis traces.
    """

    def _validate_window(winlen: float, dt: float) -> int:
        """Convert window length in seconds to samples and validate."""
        if winlen <= 0:
            raise ValueError("winlen must be positive.")

        nwin = int(round(winlen / dt))
        if nwin < 2:
            raise ValueError("winlen is too small for the sampling interval.")

        return nwin

    def _moving_sum(x: np.ndarray, nwin: int) -> np.ndarray:
        """
        Moving sum along axis 0 using cumulative sums.

        Returns sums over centered windows with clipped boundaries, matching the
        original script's edge behavior approximately.
        """
        npts = x.shape[0]
        left = nwin // 2
        right = nwin - left

        starts = np.arange(npts) - left
        stops = np.arange(npts) + right

        starts = np.clip(starts, 0, npts)
        stops = np.clip(stops, 0, npts)

        csum = np.cumsum(x, axis=0)
        csum = np.concatenate(
            [np.zeros((1, *x.shape[1:]), dtype=x.dtype), csum], axis=0
        )

        return csum[stops, ...] - csum[starts, ...], (stops - starts)

    def _windowed_kurtosis(data: np.ndarray, nwin: int) -> np.ndarray:
        """
        Compute Pearson kurtosis in moving windows along axis 0.

        Uses raw moments from cumulative sums for speed.
        """
        x1 = data
        x2 = data**2
        x3 = data**3
        x4 = data**4

        s1, counts = _moving_sum(x1, nwin)
        s2, _ = _moving_sum(x2, nwin)
        s3, _ = _moving_sum(x3, nwin)
        s4, _ = _moving_sum(x4, nwin)

        counts = counts.reshape((-1,) + (1,) * (data.ndim - 1))

        m1 = s1 / counts
        m2 = s2 / counts
        m3 = s3 / counts
        m4 = s4 / counts

        # Central moments
        mu2 = m2 - m1**2
        mu4 = m4 - 4 * m1 * m3 + 6 * (m1**2) * m2 - 3 * m1**4

        out = np.divide(
            mu4,
            mu2**2,
            out=np.full_like(mu4, np.nan, dtype=float),
            where=mu2 > 0,
        )
        return out

    def _recursive_kurtosis(data: np.ndarray, dt: float, winlen: float) -> np.ndarray:
        """
        Recursive pseudo-kurtosis after Langet et al.-style formulation.

        Acts along axis 0.
        """
        fsamp = 1.0 / dt
        c = 1.0 - 1.0 / (fsamp * winlen)

        npts = data.shape[0]
        out = np.empty_like(data, dtype=float)

        # Per-channel initialization
        varx = np.std(data, axis=0)
        mean_value = np.zeros(data.shape[1:], dtype=float)
        var_value = np.zeros(data.shape[1:], dtype=float)
        kurt_value = np.zeros(data.shape[1:], dtype=float)

        varx2 = varx**2

        for i in range(npts):
            xi = data[i, ...]

            mean_value = c * mean_value + (1.0 - c) * xi
            var_value = c * var_value + (1.0 - c) * (xi - mean_value) ** 2

            norm_factor = np.where(var_value > varx, var_value**2, varx2)
            kurt_value = (
                c * kurt_value + (1.0 - c) * (xi - mean_value) ** 4 / norm_factor
            )
            out[i, ...] = kurt_value

        return out

    patch_t = patch.transpose(dim, ...)
    data = np.asarray(patch_t.data, dtype=float)

    coord = patch_t.get_coord(dim)
    dt = to_float(coord.step)
    if dt is None or dt <= 0:
        raise ValueError(
            f"Coordinate step for dim={dim!r} must be defined and positive."
        )

    if recursive:
        out = _recursive_kurtosis(data, dt=dt, winlen=winlen)
    else:
        nwin = _validate_window(winlen, dt)
        out = _windowed_kurtosis(data, nwin=nwin)

    return (
        patch_t.new(data=out)
        .transpose(*patch.dims)
        .update(attrs={"data_type": "Kurtosis", "data_units": ""})
    )


# %%
@patch_function()
def aic(
    patch: PatchType,
    dim: str = "time",
    fill_edges: bool = True,
) -> PatchType:
    """
    Compute a variance-based AIC picker curve along a patch dimension.

    This uses the classic split-point formulation

        AIC(k) = k * log(var_left) + (N - k) * log(var_right)

    where ``var_left`` is the variance of samples before ``k`` and
    ``var_right`` is the variance of samples from ``k`` onward.

    See: Maeda, N. (1985). A method for reading and checking phase times
    in autoprocessing system of seismic wave data, Zisin 38, no. 3, 365-379


    Parameters
    ----------
    patch
        Input DASCore patch.
    dim
        Dimension along which to compute the AIC curve.
    fill_edges
        If True, copy the nearest valid values into the first/last sample.

    Returns
    -------
    PatchType
        Patch containing the variance-based AIC curve with the same shape
        and dimensions as the input patch.
    """
    patch_t = patch.transpose(dim, ...)
    data = np.asarray(patch_t.data, dtype=float)

    npts = data.shape[0]
    if npts < 3:
        raise ValueError("variance_aic requires at least 3 samples.")

    # flatten non-target dims so we can process all traces together
    trailing_shape = data.shape[1:]
    ntraces = int(np.prod(trailing_shape)) if trailing_shape else 1
    data2d = data.reshape(npts, ntraces)

    # cumulative sums for left segment stats
    csum1 = np.cumsum(data2d, axis=0)
    csum2 = np.cumsum(data2d * data2d, axis=0)

    total_sum1 = csum1[-1:, :]
    total_sum2 = csum2[-1:, :]

    # split index k runs from 1 .. npts-2
    k = np.arange(1, npts - 1, dtype=float)[:, None]
    n_right = npts - k

    # left segment is data[:k]
    s1_left = csum1[:-2, :]
    s2_left = csum2[:-2, :]

    # right segment is data[k:]
    s1_right = total_sum1 - s1_left
    s2_right = total_sum2 - s2_left

    mean_left = s1_left / k
    mean_right = s1_right / n_right

    var_left = s2_left / k - mean_left**2
    var_right = s2_right / n_right - mean_right**2

    log_left = np.log(
        var_left,
        out=np.full_like(var_left, np.nan),
        where=var_left > 0,
    )
    log_right = np.log(
        var_right,
        out=np.full_like(var_right, np.nan),
        where=var_right > 0,
    )

    aic_mid = k * log_left + n_right * log_right

    out = np.full((npts, ntraces), np.nan, dtype=float)
    out[1:-1, :] = aic_mid
    out[np.isinf(out)] = np.nan

    if fill_edges and npts > 2:
        out[0, :] = out[1, :]
        out[-1, :] = out[-2, :]

    out = out.reshape(data.shape)

    return (
        patch_t.new(data=out)
        .transpose(*patch.dims)
        .update(attrs={"data_type": "AIC", "data_units": ""})
    )
