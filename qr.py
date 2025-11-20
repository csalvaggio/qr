"""
QR code generation and rendering utilities.

This module provides tools for constructing QR codes from payload text,
manipulating their boolean module structure, and rendering them with a
variety of visual styles. The QR structure is generated once and stored as
a 2D boolean array, while rendering is performed explicitly in pixel space.
This enables fine-grained control over colors, textures, overlays, and
validation using OpenCV.

Functions
---------
make_qr
    Create a QRCodeImage from payload and configuration.
save_qr_png
    Generate and save a QR PNG in any supported rendering mode.

Classes
-------
QRSpec
    Immutable QR code configuration.
QRCodeImage
    QR code backed by a boolean module matrix with explicit rendering
    and validation utilities.

Features
--------
* Deterministic QR structure generation using the qrcode library.
* Explicit rendering into:
    - NumPy arrays (RGB)
    - PIL images
    - PNG bytes
* Support for classic foreground/background rendering.
* Textured rendering in which dark modules are filled from a user-supplied
  texture image.
* Centered overlay rendering with optional transparency.
* OpenCV-based decoding and validation (optional dependency).
* Convenience helpers for one-shot creation and saving.

Examples
--------
Create a QR specification and image:

>>> spec = QRSpec(
...     data="https://example.com",
...     box_size=10,
...     border=4,
...     ecc="Q",
... )
>>> qr = QRCodeImage(spec)

Render to an RGB NumPy array:

>>> arr = qr.render_array()
>>> arr.shape
(H, W, 3)

Save a standard PNG:

>>> qr.save_png("plain.png")

Render with a texture:

>>> import imageio
>>> tex = imageio.v3.imread("texture.jpg")
>>> qr.save_png("styled.png", texture=tex)

Apply a centered overlay:

>>> from PIL import Image
>>> logo = Image.open("logo.png")
>>> qr.save_png("with_logo.png", overlay=logo, relative_size=0.2)

Validate a QR code:

>>> qr.validate()
True

Decode an arbitrary image (requires OpenCV):

>>> text, ok = qr._decode_with_cv2(image=arr)

Notes
-----
OpenCV is optional and required only for decoding. If unavailable, decoding
raises RuntimeError.

The module intentionally abstracts away most of the underlying qrcode
library in order to provide consistent module-matrix-based rendering
behavior.

Author
------
Carl Salvaggio, Ph.D.
Digital Imaging and Remote Sensing Laboratory (DIRS)
Rochester Institute of Technology (RIT)

Contact
-------
carl.salvaggio@rit.edu
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

import qrcode
from qrcode.constants import (
    ERROR_CORRECT_L,
    ERROR_CORRECT_M,
    ERROR_CORRECT_Q,
    ERROR_CORRECT_H,
)


_ECC_MAP = {
    "L": ERROR_CORRECT_L,  # ~7% error correction
    "M": ERROR_CORRECT_M,  # ~15% (default)
    "Q": ERROR_CORRECT_Q,  # ~25%
    "H": ERROR_CORRECT_H,  # ~30%
}


Color = tuple[int, int, int]  # (R, G, B)
ImageLike = Union[np.ndarray, Image.Image]


@dataclass(frozen=True)
class QRSpec:
    """
    Immutable configuration specification for generating a QR code.

    Parameters
    ----------
    data : str
        Payload encoded into the QR code. Must be a non-empty string.
    box_size : int, optional
        Size, in pixels, of each QR code module. The default is 10.
    border : int, optional
        Width, in modules, of the quiet zone around the code. The
        default is 4, which is the minimum recommended by the QR
        standard.
    ecc : {'L', 'M', 'Q', 'H'}, optional
        Error-correction level. The value is case-insensitive and is
        normalized to uppercase. The default is 'M'.

        Levels
        ------
        'L'
            Low (approximately 7% codewords restored).
        'M'
            Medium (approximately 15%).
        'Q'
            Quartile (approximately 25%).
        'H'
            High (approximately 30%).

    Notes
    -----
    This class is frozen (immutable). After creation, attributes cannot
    be reassigned. The `ecc` value is validated and normalized to
    uppercase during ``__post_init__``.

    Raises
    ------
    ValueError
        If `data` is empty or whitespace.
    ValueError
        If `ecc` is not one of {'L', 'M', 'Q', 'H'}.
    """

    data: str
    box_size: int = 10
    border: int = 4
    ecc: str = "M"

    def __post_init__(self) -> None:
        if not isinstance(self.data, str) or not self.data.strip():
            raise ValueError("'data' must be a non-empty string")

        ecc_upper = self.ecc.upper()
        if ecc_upper not in _ECC_MAP:
            raise ValueError("'ecc' must be one of {'L', 'M', 'Q', 'H'}")

        # Store normalized ECC
        object.__setattr__(self, "ecc", ecc_upper)

    @property
    def ecc_level(self) -> int:
        """
        Integer error-correction level for this QR code configuration.

        Returns
        -------
        int
            Numeric ECC level corresponding to `ecc`, as defined
            in ``_ECC_MAP``.
        """
        return _ECC_MAP[self.ecc]


class QRCodeImage:
    """
    Generated QR code backed by a boolean module matrix.

    This class builds and renders a QR code using the configuration
    specified by a QRSpec instance. Internally, the QR code is
    represented as a 2D boolean matrix of modules, where True indicates
    a dark module and False a light module. Rendering to NumPy arrays,
    PIL images, and PNG bytes is performed on top of this matrix.

    Parameters
    ----------
    spec : QRSpec
        QR code configuration, including payload (`data`), module box
        size, border width, and error-correction level.

    Attributes
    ----------
    spec : QRSpec
        QR specification used to generate this image.
    matrix : numpy.ndarray
        Boolean 2D array representing the QR module grid with shape
        (rows, cols). True indicates a dark module.
    module_shape : tuple of int
        Shape of the QR module grid as (rows, cols).
    pixel_shape : tuple of int
        Shape of the rendered QR image in pixels as (height, width),
        including the quiet-zone border.

    Notes
    -----
    The underlying module matrix is generated once at initialization
    and reused for all subsequent rendering operations. The matrix is
    constructed using the qrcode library, with box size and border
    handled externally by this class.
    """

    def __init__(self, spec: QRSpec) -> None:
        self.spec = spec
        self._matrix = self._build_matrix()

    # ---------- Core matrix generation ----------

    def _build_matrix(self) -> np.ndarray:
        """
        Build the underlying boolean module matrix.

        Returns
        -------
        numpy.ndarray
            Boolean array of shape (rows, cols) where each element
            indicates whether the corresponding module is dark (True)
            or light (False).
        """
        qr = qrcode.QRCode(
            version=None,  # let the library pick
            error_correction=self.spec.ecc_level,
            box_size=1,
            border=0,
        )
        qr.add_data(self.spec.data)
        qr.make(fit=True)

        return np.array(qr.get_matrix(), dtype=bool)

    @property
    def matrix(self) -> np.ndarray:
        """
        Boolean module matrix.

        Returns
        -------
        numpy.ndarray
            Boolean array of shape (rows, cols). True indicates a dark
            module and False indicates a light module.
        """
        return self._matrix

    @property
    def module_shape(self) -> tuple[int, int]:
        """
        Shape of the QR module grid.

        Returns
        -------
        tuple of int
            Pair (rows, cols) giving the number of modules in each
            dimension.
        """
        return self._matrix.shape

    @property
    def pixel_shape(self) -> tuple[int, int]:
        """
        Shape of the rendered QR image in pixels.

        The pixel dimensions include the quiet-zone border and depend
        on both the module grid shape and the box size.

        Returns
        -------
        tuple of int
            Pair (height, width) of the rendered image in pixels.
        """
        rows, cols = self.module_shape
        h = (rows + 2 * self.spec.border) * self.spec.box_size
        w = (cols + 2 * self.spec.border) * self.spec.box_size
        return h, w

    # ---------- Rendering ----------

    def _full_mask(self) -> np.ndarray:
        """
        Construct the full-resolution boolean mask in pixel space.

        The mask is constructed by scaling the module matrix according
        to the box size and padding it with the configured border. It
        has the same spatial resolution as the rendered image.

        Returns
        -------
        numpy.ndarray
            Boolean array of shape (H, W) where True indicates pixels
            corresponding to dark modules in the QR code and False
            indicates background pixels (including the quiet-zone
            region).
        """
        rows, cols = self.module_shape
        box = self.spec.box_size
        border = self.spec.border

        # Scale module grid with Kronecker product
        scaled = np.kron(self.matrix, np.ones((box, box), dtype=bool))
        # Pad border in pixels (border modules * box_size pixels)
        pad = border * box
        full = np.pad(
            scaled,
            pad_width=pad,
            mode="constant",
            constant_values=False,
        )
        return full

    def render_array(
        self,
        *,
        fg: Color = (0, 0, 0),
        bg: Color = (255, 255, 255),
    ) -> np.ndarray:
        """
        Render the QR code to an RGB NumPy array.

        Parameters
        ----------
        fg : Color, optional
            Foreground color used for dark modules, as an (R, G, B)
            triple in the range 0–255. The default is (0, 0, 0).
        bg : Color, optional
            Background color used for non-module pixels (including the
            quiet-zone border), as an (R, G, B) triple in the range
            0–255. The default is (255, 255, 255).

        Returns
        -------
        numpy.ndarray
            Array of shape (H, W, 3) with dtype uint8 representing the
            rendered QR image in RGB format.
        """
        h, w = self.pixel_shape
        img = np.full((h, w, 3), bg, dtype=np.uint8)

        mask = self._full_mask()  # (H, W) bool
        img[mask] = fg
        return img

    def render_pil(
        self,
        *,
        fg: Color = (0, 0, 0),
        bg: Color = (255, 255, 255),
    ) -> Image.Image:
        """
        Render the QR code as a PIL image.

        Parameters
        ----------
        fg : Color, optional
            Foreground color used for dark modules, as an (R, G, B)
            triple. The default is (0, 0, 0).
        bg : Color, optional
            Background color used for non-module pixels, as an
            (R, G, B) triple. The default is (255, 255, 255).

        Returns
        -------
        PIL.Image.Image
            PIL image in RGB mode containing the rendered QR code.
        """
        arr = self.render_array(fg=fg, bg=bg)
        return Image.fromarray(arr, mode="RGB")

    def to_png_bytes(
        self,
        *,
        fg: Color = (0, 0, 0),
        bg: Color = (255, 255, 255),
    ) -> bytes:
        """
        Return PNG-encoded bytes of the rendered QR code.

        Parameters
        ----------
        fg : Color, optional
            Foreground color used for dark modules, as an (R, G, B)
            triple. The default is (0, 0, 0).
        bg : Color, optional
            Background color used for non-module pixels, as an
            (R, G, B) triple. The default is (255, 255, 255).

        Returns
        -------
        bytes
            PNG-encoded image data representing the rendered QR code.
        """
        img = self.render_pil(fg=fg, bg=bg)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _normalize_to_image_array(
        self,
        image: ImageLike,
        *,
        name: str = "image",
        allow_alpha: bool = False,
    ) -> np.ndarray:
        """
        Normalize a NumPy array or PIL Image into a uint8 array.

        Parameters
        ----------
        image : numpy.ndarray or PIL.Image.Image
            Input image. Supported forms are:
            - Grayscale array of shape (H, W).
            - RGB array of shape (H, W, 3).
            - RGBA array of shape (H, W, 4); the alpha channel is
              preserved if `allow_alpha` is True and dropped otherwise.
            - PIL Image in modes 'L', 'RGB', or 'RGBA'.
        name : str, optional
            Name used in error messages (for example, "texture",
            "overlay"). The default is "image".
        allow_alpha : bool, optional
            If False, the returned array is guaranteed to have shape
            (H, W, 3). If True, RGBA input is preserved and the array
            may have shape (H, W, 4). The default is False.

        Returns
        -------
        numpy.ndarray
            Array with dtype uint8 and shape (H, W, C), where C is:
            - 3 if `allow_alpha` is False.
            - 3 or 4 if `allow_alpha` is True, depending on input.

        Raises
        ------
        TypeError
            If `image` is not a NumPy array or PIL Image.
        ValueError
            If the input has an unsupported shape or channel count.
        """
        # --- PIL input ---
        if isinstance(image, Image.Image):
            if allow_alpha:
                # Preserve alpha if present
                if image.mode in ("RGBA", "LA"):
                    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8)
                else:
                    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
            else:
                arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
            return arr

        # --- NumPy input ---
        if not isinstance(image, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array or PIL.Image.Image")

        arr = np.asarray(image)
        if arr.ndim == 2:
            # Grayscale → RGB
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3:
            channels = arr.shape[2]
            if channels == 3:
                pass  # already RGB
            elif channels == 4:
                if not allow_alpha:
                    arr = arr[..., :3]  # drop alpha
                # else: keep RGBA as-is
            else:
                raise ValueError(
                    f"{name} array has unsupported channel count {channels}; "
                    f"expected 1, 3, or 4 channels."
                )
        else:
            raise ValueError(
                f"{name} array must be 2D (grayscale) or 3D (color); "
                f"got shape {arr.shape}"
            )

        return arr.astype(np.uint8)

    # ---------- Textured modules ----------

    def render_with_texture(
        self,
        texture: ImageLike,
        *,
        bg: Color = (255, 255, 255),
    ) -> np.ndarray:
        """
        Color the QR modules using a texture image.

        The texture image is resized to the module area only. For every
        dark QR module, the corresponding pixels are taken from the
        resized texture image. All non-module pixels (gaps and border)
        are set to `bg`.

        Visually, this produces:
        - Quiet zone (border) in the background color.
        - Gaps between modules in the background color.
        - Dark modules textured/colored by `texture`.

        Parameters
        ----------
        texture : numpy.ndarray or PIL.Image.Image
            Texture image as a grayscale, RGB, or RGBA source. If RGBA
            is provided, the alpha channel is ignored (modules are fully
            opaque). Values are expected in the range 0–255.
        bg : Color, optional
            Background color for non-module pixels (border and gaps), as
            an (R, G, B) triple. The default is (255, 255, 255).

        Returns
        -------
        numpy.ndarray
            RGB array of shape (Hq, Wq, 3) with dtype uint8 containing
            the composite image, where (Hq, Wq) is the QR pixel shape
            including the border.
        """
        H, W = self.pixel_shape

        # Start with pure bg color everywhere (border + gaps)
        out = np.full((H, W, 3), bg, dtype=np.uint8)

        rows, cols = self.module_shape
        box = self.spec.box_size
        border_px = self.spec.border * box

        # Module area in pixel coordinates
        mod_y0 = border_px
        mod_x0 = border_px
        mod_h = rows * box
        mod_w = cols * box

        # Normalize texture to an RGB uint8 array
        tex_arr = self._normalize_to_image_array(
            texture,
            name="texture",
            allow_alpha=False,  # always 3-channel for modules
        )

        # Resize texture to module area only
        tex_img = Image.fromarray(tex_arr, mode="RGB")
        tex_mod = tex_img.resize((mod_w, mod_h), Image.LANCZOS)
        tex_mod_arr = np.array(tex_mod, dtype=np.uint8)

        # Full mask including border
        full_mask = self._full_mask()  # (H, W) bool

        # Restrict mask to module area
        mask_mod = full_mask[mod_y0:mod_y0 + mod_h, mod_x0:mod_x0 + mod_w]

        # View into the module area of the output
        out_region = out[mod_y0:mod_y0 + mod_h, mod_x0:mod_x0 + mod_w]

        # For module pixels, take color from texture; gaps remain bg color
        out_region[mask_mod] = tex_mod_arr[mask_mod]

        return out

    # ---------- Centered overlay ----------

    def render_with_centered_overlay(
        self,
        overlay: ImageLike,
        *,
        fg: Color = (0, 0, 0),
        bg: Color = (255, 255, 255),
        relative_size: float = 0.15,
    ) -> np.ndarray:
        """
        Render the QR code and superimpose an image in its center.

        The QR code is first rendered with solid foreground/background
        colors, then the `overlay` image is resized to occupy a
        fraction of the QR's smaller dimension and pasted in the center.
        If the overlay has an alpha channel, it is used as a
        transparency mask.

        Parameters
        ----------
        overlay : numpy.ndarray or PIL.Image.Image
            Image to place at the center of the QR code. May be a NumPy
            array of shape (H, W) (grayscale) or (H, W, 3)/(H, W, 4)
            (RGB/RGBA), or a PIL image in modes 'L', 'RGB', or 'RGBA'.
        fg : Color, optional
            Foreground color used for dark modules when rendering the
            base QR, as an (R, G, B) triple. The default is (0, 0, 0).
        bg : Color, optional
            Background color for non-module pixels, as an (R, G, B)
            triple. The default is (255, 255, 255).
        relative_size : float, optional
            Maximum size of the overlay as a fraction of the QR code's
            smaller pixel dimension (for example, 0.15 corresponds to
            15% of the side length). Values are clipped to the range
            (0, 1]. The default is 0.15.

        Returns
        -------
        numpy.ndarray
            RGB array of shape (H, W, 3) with dtype uint8 representing
            the QR code with the centered overlay.

        Notes
        -----
        Covering too large a fraction of the QR code can impair
        decodability, even with high error-correction levels. Values
        of `relative_size` between approximately 0.2 and 0.3 are
        typically safe when using higher ECC levels (for example,
        'Q' or 'H').
        """
        # Clamp relative_size to (0, 1]
        relative_size = max(1e-6, min(1.0, float(relative_size)))

        # Render base QR as PIL image
        base = self.render_pil(fg=fg, bg=bg).convert("RGBA")
        qr_w, qr_h = base.size  # (width, height)

        # Normalize overlay to an array; preserve alpha if present
        overlay_arr = self._normalize_to_image_array(
            overlay,
            name="overlay",
            allow_alpha=True,
        )

        # Build a PIL image from the normalized array
        if overlay_arr.shape[2] == 4:
            overlay_img = Image.fromarray(overlay_arr, mode="RGBA")
        else:
            overlay_img = Image.fromarray(overlay_arr, mode="RGB")

        # Ensure we have an RGBA image for alpha compositing
        overlay_img = overlay_img.convert("RGBA")

        # Compute target overlay size
        overlay_w, overlay_h = overlay_img.size
        max_side = min(qr_w, qr_h) * relative_size
        scale = min(max_side / overlay_w, max_side / overlay_h, 1.0)
        new_w = max(1, int(round(overlay_w * scale)))
        new_h = max(1, int(round(overlay_h * scale)))

        overlay_resized = overlay_img.resize((new_w, new_h), Image.LANCZOS)

        # Center position
        x0 = (qr_w - new_w) // 2
        y0 = (qr_h - new_h) // 2

        # Paste with alpha
        base.paste(overlay_resized, (x0, y0), overlay_resized)

        # Return as RGB NumPy array
        return np.array(base.convert("RGB"), dtype=np.uint8)

    # ---------- Validation / decoding (with OpenCV) ----------

    def _decode_with_cv2(
        self,
        image: Optional[np.ndarray] = None,
    ) -> tuple[Optional[str], bool]:
        """
        Decode a QR image using OpenCV's QRCodeDetector.

        Parameters
        ----------
        image : numpy.ndarray, optional
            Image to decode as a NumPy array. If None, the plain
            rendered QR code from this object is used. If provided, it
            must be a color image of shape (H, W, 3), interpreted as
            RGB. The default is None.

        Returns
        -------
        tuple of (str or None, bool)
            Tuple (decoded_text, ok) where:
            - decoded_text is the decoded string, or None if no QR code
              was detected.
            - ok is True if OpenCV reported a successful decode and
              False otherwise.

        Raises
        ------
        RuntimeError
            If OpenCV (cv2) is not installed.
        ValueError
            If `image` is provided but does not have shape (H, W, 3).
        """
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "_decode_with_cv2 requires OpenCV (cv2) to be installed."
            ) from exc

        if image is None:
            # Use our own plain rendering (RGB)
            rgb = self.render_array()
        else:
            img = np.asarray(image)
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError("image must be (H, W, 3)")
            rgb = img

        # Ensure uint8
        rgb = rgb.astype(np.uint8, copy=False)

        # Assume input is RGB and convert to BGR for OpenCV
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        detector = cv2.QRCodeDetector()
        data, points, _ = detector.detectAndDecode(bgr)

        if points is None or not data:
            return None, False

        return data, True

    def validate(
        self,
        image: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Validate that a QR image decodes to this object's payload.

        This method decodes the given image (or the plain rendered QR if
        no image is provided) and checks whether the decoded text matches
        ``self.spec.data``.

        Parameters
        ----------
        image : numpy.ndarray, optional
            Image to validate as a NumPy array. If None, the plain
            rendered QR code from this object is validated. If
            provided, it must be a color image of shape (H, W, 3),
            interpreted as RGB. The default is None.

        Returns
        -------
        bool
            True if the QR code is successfully decoded and matches
            ``self.spec.data``, False otherwise.
        """
        decoded, ok = self._decode_with_cv2(image=image)
        return bool(ok and decoded == self.spec.data)

    # ---------- Convenience saving ----------

    def save_png(
        self,
        path: str | Path,
        *,
        fg: Color = (0, 0, 0),
        bg: Color = (255, 255, 255),
        texture: Optional[ImageLike] = None,
        overlay: Optional[ImageLike] = None,
        relative_size: float = 0.15,
    ) -> None:
        """
        Save the QR code as a PNG file.

        Parameters
        ----------
        path : str or pathlib.Path
            Output path for the PNG file.
        fg : Color, optional
            Foreground color used for dark modules when rendering the
            base QR, as an (R, G, B) triple. The default is (0, 0, 0).
        bg : Color, optional
            Background color for non-module pixels, as an (R, G, B)
            triple. The default is (255, 255, 255).
        texture : ImageLike, optional
            Texture image used to color dark modules. If provided and
            `overlay` is not provided, the QR is rendered with textured
            modules. The default is None.
        overlay : ImageLike, optional
            Image to be overlaid at the center of the QR code. If
            provided, this mode takes priority over `texture`. The
            default is None.
        relative_size : float, optional
            Maximum size of the overlay as a fraction of the QR code's
            smaller pixel dimension. See
            ``render_with_centered_overlay`` for details. The default
            is 0.15.

        Returns
        -------
        None
            This method does not return anything. The PNG file is
            written to `path`.

        Notes
        -----
        The rendering mode is selected by the following priority:
        1. If `overlay` is provided, centered overlay mode is used.
        2. Else if `texture` is provided, textured module mode is used.
        3. Otherwise, classic foreground/background rendering is used.
        """
        path = Path(path)

        if overlay is not None:
            comp = self.render_with_centered_overlay(
                overlay,
                relative_size=relative_size,
                fg=fg,
                bg=bg,
            )
            Image.fromarray(comp, mode="RGB").save(path, format="PNG")
            return

        if texture is not None:
            comp = self.render_with_texture(texture, bg=bg)
            Image.fromarray(comp, mode="RGB").save(path, format="PNG")
            return

        # Classic foreground/background QR rendering
        img = self.render_pil(fg=fg, bg=bg)
        img.save(path, format="PNG")


# ---------- Helper for creation ----------

def make_qr(
    data: str,
    *,
    box_size: int = 10,
    border: int = 4,
    ecc: str = "M",
) -> QRCodeImage:
    """
    Create a QRCodeImage from payload and configuration.

    Parameters
    ----------
    data : str
        Payload encoded into the QR code. Must be a non-empty string.
    box_size : int, optional
        Size, in pixels, of each QR code module. The default is 10.
    border : int, optional
        Width, in modules, of the quiet zone around the code. The
        default is 4.
    ecc : {'L', 'M', 'Q', 'H'}, optional
        Error-correction level. The default is 'M'. See QRSpec for
        details.

    Returns
    -------
    QRCodeImage
        Generated QRCodeImage instance built from the provided
        configuration.

    Raises
    ------
    ValueError
        If `data` is empty or `ecc` is invalid, as enforced by QRSpec.
    """
    spec = QRSpec(data=data, box_size=box_size, border=border, ecc=ecc)
    return QRCodeImage(spec)


# ---------- Helper for creation and saving ----------

def save_qr_png(
    path: str | Path,
    data: str,
    *,
    box_size: int = 10,
    border: int = 4,
    ecc: str = "M",
    fg: Color = (0, 0, 0),
    bg: Color = (255, 255, 255),
    texture: Optional[ImageLike] = None,
    overlay: Optional[ImageLike] = None,
    relative_size: float = 0.25,
) -> None:
    """
    Generate and save a QR PNG in any supported rendering mode.

    This is a convenience helper that constructs a QRCodeImage from
    `data` and saves it to `path` using one of the available rendering
    modes: classic foreground/background, textured modules, or centered
    overlay.

    Parameters
    ----------
    path : str or pathlib.Path
        Output path for the PNG file.
    data : str
        Payload encoded into the QR code.
    box_size : int, optional
        Size, in pixels, of each QR code module. The default is 10.
    border : int, optional
        Width, in modules, of the quiet zone around the code. The
        default is 4.
    ecc : {'L', 'M', 'Q', 'H'}, optional
        Error-correction level. The default is 'M'.
    fg : Color, optional
        Foreground color used for dark modules in classic/overlay
        modes, as an (R, G, B) triple. The default is (0, 0, 0).
    bg : Color, optional
        Background color used for non-module pixels, as an (R, G, B)
        triple. The default is (255, 255, 255).
    texture : ImageLike, optional
        Texture image used to color dark modules in textured mode. The
        default is None.
    overlay : ImageLike, optional
        Image to be overlaid at the center of the QR code in overlay
        mode. The default is None.
    relative_size : float, optional
        Maximum size of the overlay as a fraction of the QR code's
        smaller pixel dimension, passed through to
        ``QRCodeImage.save_png`` and ultimately to
        ``render_with_centered_overlay``. The default is 0.25.

    Returns
    -------
    None
        This function does not return anything. The PNG file is written
        to `path`.
    """
    qr = make_qr(data, box_size=box_size, border=border, ecc=ecc)
    qr.save_png(
        path,
        fg=fg,
        bg=bg,
        texture=texture,
        overlay=overlay,
        relative_size=relative_size,
    )


if __name__ == "__main__":
    url = "https://www.rit.edu/science/chester-f-carlson-center-imaging-science"
    qr = make_qr(url, box_size=20, border=4, ecc="H")

    # ---------- Render plain QR ----------
    qr.save_png("images/rit_qr_plain.png")
    try:
        print("Plain QR valid?", qr.validate())  # uses internal rendering
    except RuntimeError as e:
        print("Validation skipped (no OpenCV):", e)

    # ---------- Render textured QR ----------
    try:
        texture = Image.open("images/rit_tiger.png")
        qr.save_png("images/rit_qr_textured.png", texture=texture)
        # Validate (if OpenCV available)
        try:
            comp = qr.render_with_texture(texture)
            is_valid = qr.validate(image=comp)
            print("Textured QR valid?", is_valid)
        except RuntimeError as e:
            print("Textured QR validation skipped (no OpenCV):", e)
    except FileNotFoundError as e:
        print("Texture image not found/could not be opened:", e)

    # ---------- Render QR with overlay ----------
    try:
        overlay = Image.open("images/rit_tiger.png")
        relative_size = 0.375
        qr.save_png(
            "images/rit_qr_overlay.png",
            overlay=overlay,
            relative_size=relative_size,
        )
        # Validate (if OpenCV available)
        try:
            comp = qr.render_with_centered_overlay(
                overlay, relative_size=relative_size
            )
            is_valid = qr.validate(image=comp)
            print("Overlayed QR valid?", is_valid)
        except RuntimeError as e:
            print("Overlayed QR validation skipped (no OpenCV):", e)
    except FileNotFoundError as e:
        print("Overlay image not found/could not be opened:", e)

