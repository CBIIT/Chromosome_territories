# Author: Adib Keikhosravi, Ph.D.
# Staff Scientist, Laboratory of Receptor Biology and Gene Expression, CCR, NCI
# National Institutes of Health
# Email: adib.keikhosravi@nih.gov
# License: MIT License

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SpacingZYX = tuple[float, float, float]

SUPPORTED_IMAGE_FILTER = (
    "Microscopy images (*.lif *.czi *.nd2 *.ims *.ome.tif *.ome.tiff *.tif *.tiff "
    "*.lsm *.zarr);;All files (*)"
)


@dataclass(frozen=True)
class SceneEntry:
    reader: str
    name: str
    scene_index: int = 0
    series_index: int = 0
    tile_index: int = 0
    fov_index: int = 0


def _finite_float(value: Any, default: float = 1.0) -> float:
    try:
        out = float(value)
        if np.isfinite(out) and out > 0:
            return out
    except Exception:
        pass
    return float(default)


def _as_gray2d(arr: Any) -> np.ndarray:
    out = np.asarray(arr)
    if out.ndim == 3:
        # Prefer first channel for RGB/RGBA frames returned by PIL/readlif.
        out = out[..., 0]
    if out.ndim != 2:
        out = np.squeeze(out)
    if out.ndim != 2:
        raise ValueError(f"Expected a 2D frame after squeezing, got shape {out.shape}.")
    return out


def _aics_scene_names(path: str | Path) -> list[str]:
    from aicsimageio import AICSImage

    img = AICSImage(str(path))
    scenes = [str(s) for s in img.scenes]
    return scenes if scenes else ["Scene 1"]


def _get_spacing_from_aics(img: Any) -> SpacingZYX:
    px = img.physical_pixel_sizes
    z = _finite_float(getattr(px, "Z", None), 1.0)
    y = _finite_float(getattr(px, "Y", None), 1.0)
    x = _finite_float(getattr(px, "X", None), y)
    return (z, y, x)


def _aics_load_scene(path: str | Path, scene_index_zero_based: int = 0) -> tuple[np.ndarray, SpacingZYX, str]:
    from aicsimageio import AICSImage

    img = AICSImage(str(path))
    scenes = list(img.scenes)
    if scenes:
        if scene_index_zero_based < 0 or scene_index_zero_based >= len(scenes):
            raise ValueError(
                f"Scene index {scene_index_zero_based} is outside the available range 0..{len(scenes) - 1}."
            )
        scene = scenes[scene_index_zero_based]
        img.set_scene(scene)
        scene_name = str(scene)
    else:
        scene_name = "Scene 1"
    spacing = _get_spacing_from_aics(img)

    # Prefer a single time point. AICS handles CZI, OME-TIFF, TIFF, many ND2 files, and several others.
    try:
        arr = np.asarray(img.get_image_data("CZYX", T=0))
    except Exception:
        try:
            arr = np.asarray(img.get_image_data("CZYX"))
        except Exception:
            arr = np.asarray(img.get_image_data("ZYX"))[None, ...]
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError(f"Expected CZYX data, but got shape {arr.shape}.")
    return np.asarray(arr), spacing, scene_name


def _lif_entries(path: str | Path) -> list[SceneEntry]:
    from readlif.reader import LifFile

    lif = LifFile(str(path))
    entries: list[SceneEntry] = []
    for series_index, lif_img in enumerate(lif.get_iter_image()):
        name = str(getattr(lif_img, "name", f"Series {series_index + 1}"))
        dims = getattr(lif_img, "dims", None)
        m_n = 1
        try:
            if dims is not None:
                if hasattr(dims, "m"):
                    m_n = int(dims.m)
                elif len(dims) > 4:
                    m_n = int(dims[4])
        except Exception:
            m_n = 1
        m_n = max(1, int(m_n or 1))
        for m_idx in range(m_n):
            tile_text = f", tile {m_idx + 1}" if m_n > 1 else ""
            entries.append(
                SceneEntry(
                    reader="readlif",
                    name=f"{series_index + 1}: {name}{tile_text}",
                    series_index=series_index,
                    tile_index=m_idx,
                )
            )
    return entries


def _lif_spacing(lif_img: Any) -> SpacingZYX:
    # readlif usually reports scale in pixels per micron. Convert to microns per pixel.
    x_um = y_um = z_um = 1.0
    try:
        scale = getattr(lif_img, "scale", None)
        scale_n = getattr(lif_img, "scale_n", None)
        x_candidates: list[float] = []
        y_candidates: list[float] = []
        z_candidates: list[float] = []
        if scale is not None:
            try:
                if len(scale) > 0:
                    x_candidates.append(float(scale[0]))
                if len(scale) > 1:
                    y_candidates.append(float(scale[1]))
                if len(scale) > 2:
                    z_candidates.append(float(scale[2]))
            except Exception:
                pass
        if isinstance(scale_n, dict):
            for key, target in [(1, x_candidates), (2, y_candidates), (3, z_candidates)]:
                if key in scale_n:
                    try:
                        target.append(float(scale_n[key]))
                    except Exception:
                        pass
        for candidates, default, setter in [
            (x_candidates, 1.0, "x"),
            (y_candidates, 1.0, "y"),
            (z_candidates, 1.0, "z"),
        ]:
            val = None
            for c in candidates:
                if np.isfinite(c) and c > 0:
                    val = 1.0 / c
                    break
            if val is None:
                val = default
            # Backward compatibility safeguard for unusual px/nm reports.
            if val > 10:
                val = val / 1000.0
            if setter == "x":
                x_um = float(val)
            elif setter == "y":
                y_um = float(val)
            else:
                z_um = float(val)
    except Exception:
        pass
    if not np.isfinite(x_um) or x_um <= 0:
        x_um = y_um
    if not np.isfinite(y_um) or y_um <= 0:
        y_um = x_um
    if not np.isfinite(z_um) or z_um <= 0:
        z_um = 1.0
    return (float(z_um), float(y_um), float(x_um))


def _lif_dims(lif_img: Any) -> tuple[int, int, int]:
    z_n = t_n = c_n = 1
    try:
        dims = getattr(lif_img, "dims", None)
        if dims is not None:
            z_n = int(dims.z) if hasattr(dims, "z") else int(dims[2])
            t_n = int(dims.t) if hasattr(dims, "t") else int(dims[3])
    except Exception:
        try:
            z_n = int(getattr(lif_img, "nz", 1) or 1)
        except Exception:
            z_n = 1
        try:
            t_n = int(getattr(lif_img, "nt", 1) or 1)
        except Exception:
            t_n = 1
    try:
        channels = getattr(lif_img, "channels", 1)
        c_n = int(channels) if not isinstance(channels, Sequence) else len(channels)
    except Exception:
        c_n = 1
    return max(1, z_n), max(1, t_n), max(1, c_n)


def _lif_load_scene(path: str | Path, scene_index_zero_based: int = 0) -> tuple[np.ndarray, SpacingZYX, str]:
    from readlif.reader import LifFile

    entries = _lif_entries(path)
    if scene_index_zero_based < 0 or scene_index_zero_based >= len(entries):
        raise ValueError(
            f"Scene index {scene_index_zero_based} is outside the available range 0..{len(entries) - 1}."
        )
    entry = entries[scene_index_zero_based]
    lif = LifFile(str(path))
    lif_imgs = list(lif.get_iter_image())
    lif_img = lif_imgs[entry.series_index]
    z_n, _t_n, c_n = _lif_dims(lif_img)
    frames: list[list[np.ndarray]] = []
    for c in range(c_n):
        ch_frames: list[np.ndarray] = []
        for z in range(z_n):
            try:
                frame = lif_img.get_frame(z=z, t=0, c=c, m=entry.tile_index)
            except TypeError:
                frame = lif_img.get_frame(z=z, t=0, c=c)
            ch_frames.append(_as_gray2d(frame))
        frames.append(ch_frames)
    arr = np.stack([np.stack(ch, axis=0) for ch in frames], axis=0)
    return arr, _lif_spacing(lif_img), entry.name


def _nd2_entries(path: str | Path) -> list[SceneEntry]:
    import nd2reader

    with nd2reader.ND2Reader(str(path)) as images:
        n_views = int(images.sizes.get("v", 1) or 1)
    return [SceneEntry(reader="nd2reader", name=f"FOV {i + 1}", fov_index=i) for i in range(max(1, n_views))]


def _nd2_load_scene(path: str | Path, scene_index_zero_based: int = 0) -> tuple[np.ndarray, SpacingZYX, str]:
    import nd2reader

    entries = _nd2_entries(path)
    if scene_index_zero_based < 0 or scene_index_zero_based >= len(entries):
        raise ValueError(
            f"Scene index {scene_index_zero_based} is outside the available range 0..{len(entries) - 1}."
        )
    fov = entries[scene_index_zero_based].fov_index
    with nd2reader.ND2Reader(str(path)) as images:
        sizes = images.sizes
        z_n = int(sizes.get("z", 1) or 1)
        c_n = int(sizes.get("c", 1) or 1)
        frames: list[list[np.ndarray]] = []
        for c in range(c_n):
            ch_frames: list[np.ndarray] = []
            for z in range(z_n):
                kwargs = {"t": 0, "z": z, "c": c}
                if "v" in sizes:
                    kwargs["v"] = fov
                ch_frames.append(_as_gray2d(images.get_frame_2D(**kwargs)))
            frames.append(ch_frames)
        arr = np.stack([np.stack(ch, axis=0) for ch in frames], axis=0)
        yx = _finite_float(images.metadata.get("pixel_microns", 1.0), 1.0)
        # nd2reader exposes z spacing inconsistently, so keep z=1.0 unless available.
        z_um = _finite_float(images.metadata.get("z_levels", [1.0])[1] - images.metadata.get("z_levels", [0.0])[0] if isinstance(images.metadata.get("z_levels"), list) and len(images.metadata.get("z_levels")) > 1 else 1.0, 1.0)
    return arr, (z_um, yx, yx), entries[scene_index_zero_based].name


def _ims_load_scene(path: str | Path, scene_index_zero_based: int = 0) -> tuple[np.ndarray, SpacingZYX, str]:
    if scene_index_zero_based != 0:
        raise ValueError("IMS fallback reader exposes a single scene at index 0.")
    from imaris_ims_file_reader.ims import ims

    images = ims(str(path))
    c_n = int(images.Channels)
    z_n = int(images.shape[2])
    frames: list[list[np.ndarray]] = []
    for c in range(c_n):
        ch_frames: list[np.ndarray] = []
        for z in range(z_n):
            ch_frames.append(_as_gray2d(images[0, c, z, :, :]))
        frames.append(ch_frames)
    arr = np.stack([np.stack(ch, axis=0) for ch in frames], axis=0)
    try:
        res = list(images.resolution)
        # Many IMS readers report X,Y,Z resolution or spacing; handle both by using last values if present.
        if len(res) >= 3:
            x_um, y_um, z_um = map(_finite_float, res[:3])
        else:
            x_um = y_um = z_um = _finite_float(res[-1] if res else 1.0, 1.0)
    except Exception:
        z_um = y_um = x_um = 1.0
    return arr, (float(z_um), float(y_um), float(x_um)), "Scene 1"


def _axes_to_czyx(arr: np.ndarray, axes: str) -> np.ndarray:
    axes = str(axes or "").upper()
    out = np.asarray(arr)
    # Drop unsupported dimensions by taking the first plane/time/position.
    for drop_axis in ["T", "S", "M", "V", "R", "I"]:
        while drop_axis in axes:
            ax = axes.index(drop_axis)
            out = np.take(out, 0, axis=ax)
            axes = axes[:ax] + axes[ax + 1 :]
    # If there is no metadata, infer common layouts.
    if not axes or len(axes) != out.ndim:
        if out.ndim == 2:
            axes = "YX"
        elif out.ndim == 3:
            # Ambiguous. For microscopy stacks this is most often ZYX.
            axes = "ZYX"
        elif out.ndim == 4:
            # Common ImageJ/OME export from this plugin is CZYX or ZCYX.
            axes = "CZYX"
        else:
            raise ValueError(f"Cannot infer axes for TIFF shape {out.shape}.")
    if "Y" not in axes or "X" not in axes:
        raise ValueError(f"TIFF axes must contain Y and X; got axes={axes!r}, shape={out.shape}.")
    if "C" not in axes:
        out = np.expand_dims(out, axis=0)
        axes = "C" + axes
    if "Z" not in axes:
        # Insert Z after C when possible.
        c_axis = axes.index("C")
        out = np.expand_dims(out, axis=c_axis + 1)
        axes = axes[: c_axis + 1] + "Z" + axes[c_axis + 1 :]
    order = [axes.index(ax) for ax in "CZYX"]
    return np.transpose(out, order).copy()


def _tiff_scene_names(path: str | Path) -> list[str]:
    from tifffile import TiffFile

    with TiffFile(str(path)) as tif:
        return [f"Series {i + 1}: axes {series.axes}, shape {tuple(series.shape)}" for i, series in enumerate(tif.series)] or ["Scene 1"]


def _tiff_load_scene(path: str | Path, scene_index_zero_based: int = 0) -> tuple[np.ndarray, SpacingZYX, str]:
    from tifffile import TiffFile

    with TiffFile(str(path)) as tif:
        if scene_index_zero_based < 0 or scene_index_zero_based >= len(tif.series):
            raise ValueError(
                f"Scene index {scene_index_zero_based} is outside the available range 0..{len(tif.series) - 1}."
            )
        series = tif.series[scene_index_zero_based]
        arr = series.asarray()
        axes = getattr(series, "axes", "")
        name = f"Series {scene_index_zero_based + 1}: axes {axes}, shape {tuple(series.shape)}"
    return _axes_to_czyx(arr, axes), (1.0, 1.0, 1.0), name


def get_scene_names(image_path: str | Path) -> list[str]:
    """Return scene names for AICS-compatible files plus LIF/ND2/IMS fallbacks."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    if suffix == ".lif":
        try:
            entries = _lif_entries(path)
            if entries:
                return [e.name for e in entries]
        except Exception:
            pass
    if suffix == ".nd2":
        try:
            entries = _nd2_entries(path)
            if entries:
                return [e.name for e in entries]
        except Exception:
            pass
    if suffix == ".ims":
        # AICS may not support IMS in all environments.
        try:
            return _aics_scene_names(path)
        except Exception:
            return ["Scene 1"]
    if suffix in {".tif", ".tiff"}:
        try:
            return _aics_scene_names(path)
        except Exception:
            return _tiff_scene_names(path)
    return _aics_scene_names(path)


def load_scene_channels(image_path: str | Path, scene_index_zero_based: int = 0) -> tuple[np.ndarray, SpacingZYX, str]:
    """Load one scene/FOV/tile as a CZYX NumPy array.

    The primary reader is AICSImageIO. File-specific fallbacks mirror the HiTIPS-style
    reader logic for LIF, ND2, and IMS files when AICS is unavailable or incomplete.
    """
    path = Path(image_path)
    suffix = path.suffix.lower()
    if suffix == ".lif":
        try:
            return _lif_load_scene(path, scene_index_zero_based)
        except Exception:
            # Fall back to AICS if readlif is unavailable or this file is better handled by AICS.
            return _aics_load_scene(path, scene_index_zero_based)
    if suffix == ".nd2":
        try:
            return _aics_load_scene(path, scene_index_zero_based)
        except Exception:
            return _nd2_load_scene(path, scene_index_zero_based)
    if suffix == ".ims":
        try:
            return _aics_load_scene(path, scene_index_zero_based)
        except Exception:
            return _ims_load_scene(path, scene_index_zero_based)
    if suffix in {".tif", ".tiff"}:
        try:
            return _aics_load_scene(path, scene_index_zero_based)
        except Exception:
            return _tiff_load_scene(path, scene_index_zero_based)
    return _aics_load_scene(path, scene_index_zero_based)
