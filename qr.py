"""
QR code generation utilities.

This module provides two main classes:

    * :class:`QRSpec` — an immutable specification for QR code generation,
      including payload data, module box size, border width, and
      error-correction level.

    * :class:`QRCodeImage` — a generated QR code backed by a boolean module
      matrix. The class supports rendering to NumPy arrays, PIL images,
      PNG bytes, and compositing QR modules onto background images.

Unlike high-level QR libraries that directly render to bitmaps, this module
separates **QR structure** from **QR rendering**. Internally, the QR code is
represented as a boolean matrix indicating dark/light modules. Rendering is
performed explicitly by scaling this matrix in pixel space and applying
foreground/background or composited colors.

The separation provides fine-grained control over:

    * module scaling
    * quiet-zone borders
    * foreground/background colors
    * textured or image-based module rendering
    * conversion to PIL, NumPy arrays, or encoded PNG bytes
    * validation and decoding using OpenCV

Examples
--------
Create a QR specification:

>>> spec = QRSpec(
...     data="https://example.com",
...     box_size=12,
...     border=4,
...     ecc="Q",
... )

Generate the QR code matrix and render to a NumPy RGB array:

>>> qr = QRCodeImage(spec)
>>> arr = qr.render_array()
>>> arr.shape
(??? , ???, 3)

Save as a PNG file:

>>> qr.save_png("example_qr.png")

Render with a textured background image:

>>> import imageio
>>> bg = imageio.v3.imread("texture.jpg")
>>> qr.save_png("styled_qr.png", background=bg)

Validate that a rendered QR code matches its payload:

>>> qr.validate()
True

Decode a custom image (using OpenCV if installed):

>>> text, ok = qr._decode_with_cv2(image=arr)
>>> text
'https://example.com'

Notes
-----
OpenCV is optional and required only for the decoding utilities.
If OpenCV is not installed, decoding will raise a ``RuntimeError``.

This module does not attempt to expose the full power of the underlying
``qrcode`` library; instead, it standardizes QR generation to produce a
canonical module matrix and delegates all rendering to user-controlled
logic.

Author
------
Carl Salvaggio, Ph.D.
Professor / Director of the Digital Imaging and Remote Sensing Laboratory (DIRS)
Rochester Institute of Technology (RIT)
Chester F. Carlson Center for Imaging Science (CIS)

Contact
-------
carl.salvaggio@rit.edu
"""


from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Tuple, Optional

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


Color = Tuple[int, int, int]  # (R, G, B)



@dataclass(frozen=True)
class QRSpec:
    """
    Immutable configuration specification for generating a QR code.

    Parameters
    ----------
    data : str
        The payload encoded into the QR code. Must be a non-empty string.
    box_size : int, optional
        The size (in pixels) of each QR code module. Defaults to 10.
    border : int, optional
        The width (in modules) of the quiet zone around the code.
        Defaults to 4, which is the minimum recommended by the QR standard.
    ecc : {'L', 'M', 'Q', 'H'}, optional
        Error-correction level. The value is case-insensitive and will
        be normalized to uppercase. Defaults to 'M'.

        Levels:
            * 'L' — Low (~7% codewords restored)
            * 'M' — Medium (~15%)
            * 'Q' — Quartile (~25%)
            * 'H' — High (~30%)

    Notes
    -----
    This class is frozen (immutable). After creation, attributes cannot
    be reassigned. The ``ecc`` value is validated and normalized to
    uppercase during ``__post_init__``.

    Raises
    ------
    ValueError
        If ``data`` is empty or whitespace, or
        If ``ecc`` is not one of ``{'L', 'M', 'Q', 'H'}``.
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
            The numeric ECC level corresponding to ``ecc``, as defined
            in ``_ECC_MAP``.
        """
        return _ECC_MAP[self.ecc]



class QRCodeImage:
    """
    Generated QR code backed by a boolean module matrix.

    This class builds and renders a QR code using the configuration
    specified by a :class:`QRSpec` instance. Internally, the QR code
    is represented as a 2D boolean matrix of modules, where ``True``
    corresponds to a dark module and ``False`` to a light module.
    Rendering to NumPy arrays, PIL images, and PNG bytes is performed
    on top of this matrix.

    Parameters
    ----------
    spec : QRSpec
        QR code configuration, including payload (`data`), module box
        size, border width, and error-correction level.

    Attributes
    ----------
    spec : QRSpec
        The QR specification used to generate this image.
    matrix : numpy.ndarray
        A 2D boolean array representing the QR module grid, with shape
        ``(rows, cols)``. ``True`` indicates a dark module.
    module_shape : tuple of int
        Shape of the QR module grid as ``(rows, cols)``.
    pixel_shape : tuple of int
        Shape of the rendered QR image in pixels as ``(height, width)``,
        including the quiet-zone border.

    Notes
    -----
    The underlying module matrix is generated once at initialization
    and reused for all subsequent rendering operations. The matrix is
    constructed using the `qrcode` library, with box size and border
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
            A 2D boolean array of shape ``(rows, cols)`` where each
            element indicates whether the corresponding module is dark
            (``True``) or light (``False``).
        """
        qr = qrcode.QRCode(
            version=None,  # let the library pick
            error_correction=self.spec.ecc_level,
            box_size=1,
            border=0,
        )
        qr.add_data(self.spec.data)
        qr.make(fit=True)

        matrix = np.array(qr.get_matrix(), dtype=bool)  # shape (rows, cols)
        return matrix

    @property
    def matrix(self) -> np.ndarray:
        """
        Boolean module matrix.

        Returns
        -------
        numpy.ndarray
            A 2D boolean array of shape ``(rows, cols)``. ``True``
            indicates a dark module, ``False`` a light module.
        """
        return self._matrix

    @property
    def module_shape(self) -> Tuple[int, int]:
        """
        Shape of the QR module grid.

        Returns
        -------
        tuple of int
            The pair ``(rows, cols)`` giving the number of modules in
            each dimension.
        """
        return self._matrix.shape

    @property
    def pixel_shape(self) -> Tuple[int, int]:
        """
        Shape of the rendered QR image in pixels.

        The pixel dimensions include the quiet-zone border and depend
        on both the module grid shape and the box size.

        Returns
        -------
        tuple of int
            The pair ``(height, width)`` of the rendered image in pixels.
        """
        rows, cols = self.module_shape
        h = (rows + 2 * self.spec.border) * self.spec.box_size
        w = (cols + 2 * self.spec.border) * self.spec.box_size
        return h, w

    # ---------- Rendering ----------

    def _full_mask(self) -> np.ndarray:
        """
        Full-resolution boolean mask in pixel space.

        The mask is constructed by scaling the module matrix according
        to the box size and padding it with the configured border. It
        has the same spatial resolution as the rendered image.

        Returns
        -------
        numpy.ndarray
            A 2D boolean array of shape ``(H, W)`` where ``True``
            indicates pixels corresponding to dark modules in the QR
            code (including the module area, excluding the quiet-zone
            background).
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
            Foreground color used for dark modules, as an RGB triple
            ``(R, G, B)`` in the range 0–255. Defaults to black.
        bg : Color, optional
            Background color used for non-module pixels (including the
            quiet-zone border), as an RGB triple. Defaults to white.

        Returns
        -------
        numpy.ndarray
            An array of shape ``(H, W, 3)`` with dtype ``uint8``
            representing the rendered QR image in RGB format.
        """
        h, w = self.pixel_shape
        # Start with background
        img = np.full((h, w, 3), bg, dtype=np.uint8)

        mask = self._full_mask()  # (H, W) bool

        # Set foreground where mask is True; NumPy broadcasts `fg` over channels
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
            Foreground color used for dark modules, as an RGB triple.
            Defaults to black.
        bg : Color, optional
            Background color used for non-module pixels, as an RGB
            triple. Defaults to white.

        Returns
        -------
        PIL.Image.Image
            A PIL image in RGB mode containing the rendered QR code.
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
            Foreground color used for dark modules, as an RGB triple.
            Defaults to black.
        bg : Color, optional
            Background color used for non-module pixels, as an RGB
            triple. Defaults to white.

        Returns
        -------
        bytes
            PNG-encoded image data representing the rendered QR code.
        """
        img = self.render_pil(fg=fg, bg=bg)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # ---------- Background compositing ----------

    def overlay_on_background(
        self,
        background: np.ndarray,
        *,
        bg: Color = (255, 255, 255),
    ) -> np.ndarray:
        """
        Composite the QR modules over a background image.

        The background image is resized to the *module area* only.
        For every dark QR module, the corresponding pixels are taken
        from the resized background. All non-module pixels (gaps and
        border) are set to ``bg``.

        Visually:
            - White (or `bg`) quiet zone (border)
            - White (or `bg`) gaps between modules
            - Modules themselves are textured/colored by `background`

        Parameters
        ----------
        background : numpy.ndarray
            Background image as an RGB array of shape ``(H, W, 3)`` or
            a grayscale array of shape ``(H, W)``. Values are expected
            in the range 0–255.
        bg : Color, optional
            Background color for non-module pixels (border and gaps),
            as an RGB triple. Defaults to white.

        Returns
        -------
        numpy.ndarray
            Composite RGB image as an array of shape ``(Hq, Wq, 3)``
            with dtype ``uint8``, where ``(Hq, Wq)`` is the QR pixel
            shape including border.
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

        # Ensure background is RGB
        if background.ndim == 2:
            background = np.stack([background] * 3, axis=-1)
        elif background.ndim != 3 or background.shape[2] != 3:
            raise ValueError("background must be (H, W) or (H, W, 3) array")

        # Resize background to module area ONLY
        bg_img = Image.fromarray(background.astype(np.uint8), mode="RGB")
        bg_mod = bg_img.resize((mod_w, mod_h), Image.LANCZOS)
        bg_mod_arr = np.array(bg_mod, dtype=np.uint8)

        # Full mask including border
        full_mask = self._full_mask()  # (H, W) bool

        # Restrict mask to module area
        mask_mod = full_mask[mod_y0:mod_y0 + mod_h, mod_x0:mod_x0 + mod_w]  # (mod_h, mod_w) bool

        # View into the module area of the output
        out_region = out[mod_y0:mod_y0 + mod_h, mod_x0:mod_x0 + mod_w]

        # For module pixels, take color from background; gaps remain bg color
        out_region[mask_mod] = bg_mod_arr[mask_mod]

        return out

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
            Image to decode as a NumPy array.
            If ``None``, the plain rendered QR code from this object is
            used. If provided, it must be an RGB or BGR image of shape
            ``(H, W, 3)``.

        Returns
        -------
        tuple of (str or None, bool)
            A pair ``(decoded_text, ok)`` where:
                * ``decoded_text`` is the decoded string, or ``None`` if
                  no QR code was detected.
                * ``ok`` is ``True`` if OpenCV reported a successful
                  decode, ``False`` otherwise.

        Raises
        ------
        RuntimeError
            If OpenCV (``cv2``) is not installed.
        ValueError
            If `image` is provided but does not have shape ``(H, W, 3)``.
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
            Image to validate as a NumPy array.
            If ``None``, the plain rendered QR code from this object is
            validated. If provided, it must be an RGB or BGR image of
            shape ``(H, W, 3)``.

        Returns
        -------
        bool
            ``True`` if the QR code is successfully decoded and matches
            ``self.spec.data``; ``False`` otherwise.
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
        background: Optional[np.ndarray] = None,
    ) -> None:
        """
        Save the QR code as a PNG file.

        If `background` is ``None``, a classic QR code is rendered with
        solid foreground modules on a uniform background. If `background`
        is provided, only the modules are colored using the resized
        background image, while border and gaps are set to ``bg``.

        Parameters
        ----------
        path : str or pathlib.Path
            Filesystem path where the PNG image will be written.
        fg : Color, optional
            Foreground color used for dark modules, as an RGB triple.
            Ignored when `background` is provided (module colors come
            from the background image). Defaults to black.
        bg : Color, optional
            Background color for non-module pixels (border and gaps),
            as an RGB triple. Defaults to white.
        background : numpy.ndarray, optional
            Background image as an RGB array of shape ``(H, W, 3)`` or
            grayscale array of shape ``(H, W)``. If provided, the QR
            modules are textured with this image using
            :meth:`overlay_on_background`.

        Returns
        -------
        None
        """
        path = Path(path)
        if background is None:
            img = self.render_pil(fg=fg, bg=bg)
            img.save(path, format="PNG")
        else:
            comp = self.overlay_on_background(background, bg=bg)
            Image.fromarray(comp, mode="RGB").save(path, format="PNG")



# ---------- Helper for one-liner ----------

def make_qr(
    data: str,
    *,
    box_size: int = 10,
    border: int = 4,
    ecc: str = "M",
) -> QRCodeImage:
    """One-shot helper to go from data to QRCodeImage."""
    spec = QRSpec(data=data, box_size=box_size, border=border, ecc=ecc)
    return QRCodeImage(spec)



if __name__ == "__main__":
    url = "https://www.rit.edu/science/chester-f-carlson-center-imaging-science"
    qr = make_qr(url, box_size=20, border=4, ecc="H")

    # Render plain QR, save, and validate
    qr.save_png("images/rit_qr_plain.png")
    try:
        print("Plain QR valid?", qr.validate())  # uses internal rendering
    except RuntimeError as e:
        print("Validation skipped (no OpenCV):", e)

    # ---------- OR ----------

    # Render sylized QR (modules colored by background image, border/gaps
    # white), save, and validate
    try:
        bg = np.array(Image.open("images/rit_tiger.png").convert("RGB"))
        qr.save_png("images/rit_qr_fancy.png", background=bg)

        # Validate the stylized image (if OpenCV available)
        try:
            comp = qr.overlay_on_background(bg)
            is_valid = qr.validate(image=comp)
            print("Colored-modules QR valid?", is_valid)
        except RuntimeError as e:
            print("Colored-modules validation skipped (no OpenCV):", e)
    except FileNotFoundError as e:
        print("Background image not found/could not be opened:", e)

