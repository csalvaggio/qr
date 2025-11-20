# qr - A QR Code Generation Utility Module

QR code generation and rendering utilities for Python.

This module provides tools for constructing QR codes from payload text, manipulating their boolean module structure, and rendering them with a variety of visual styles. The QR structure is generated once and stored as a 2D boolean array, while rendering is performed explicitly in pixel space. This enables fine-grained control over colors, textures, overlays, and validation using OpenCV (optional).

The core premise on which this is built is to **separate QR structure from QR rendering**:

- The QR content is stored as a boolean module matrix (`True` = dark module, `False` = light).
- Rendering to bitmaps (NumPy arrays, PIL images, PNG files, or stylized overlays) is handled explicitly by your code, so you stay in control of colors, compositing, and post-processing.

---

## Notes & Design Choices

- The underlying QR module matrix is generated once at initialization and reused for all rendering operations.
- The module intentionally abstracts away most of the `qrcode` library’s rendering in favor of explicit pixel-based control.
- All image normalization is funneled through `_normalize_to_image_array`, which accepts NumPy arrays and PIL `Image` objects in grayscale, RGB, or RGBA form.

---

## Features

- **Immutable QR specification** via `QRSpec` (data, box size, border, ECC level).
- **Matrix-based QR representation** via `QRCodeImage`.
- **Rendering**:
  - To RGB NumPy arrays.
  - To PIL `Image` objects.
  - Directly to PNG bytes or files.
- **Styles**:
  - Support for classic foreground/background rendering.
  - Textured modules using a background **texture image** (modules are “textured” by the background; border and gaps remain a clean solid color).
  - Centered overlay rendering with optional transparency.
- **Optional validation** (if OpenCV is installed):
  - Decode a QR and verify it matches the expected payload.
- **Helpers**
  - Convenience helpers for one-shot creation and saving.

---

## API Overview

The two classes that make up this basis for this module are:
### `QRSpec`

An immutable configuration for generating a QR code.

```python
spec = QRSpec(
    data: str,
    box_size: int = 10,
    border: int = 4,
    ecc: str = "M",  # 'L', 'M', 'Q', or 'H'
)
```

- `data` – Payload string (must be non-empty).
- `box_size` – Pixel size of each module.
- `border` – Quiet-zone width in **modules**.
- `ecc` – Error-correction level (`'L'`, `'M'`, `'Q'`, `'H'`).

Properties:

- `ecc_level` – Integer ECC constant for the underlying `qrcode` library.

---

### `QRCodeImage`

```python
qr = QRCodeImage(spec: QRSpec)
```

Attributes & properties:

- `spec` – The `QRSpec` used to generate this QR.
- `matrix` – Boolean 2D array of modules (`True` = dark).
- `module_shape` – `(rows, cols)` in module space.
- `pixel_shape` – `(height, width)` in pixel space (including border).

Key methods:

- `render_array(fg=(0,0,0), bg=(255,255,255)) -> np.ndarray`
- `render_pil(fg=(0,0,0), bg=(255,255,255)) -> PIL.Image.Image`
- `to_png_bytes(fg=(0,0,0), bg=(255,255,255)) -> bytes`
- `render_with_texture(texture, bg=(255,255,255)) -> np.ndarray`
- `render_with_centered_overlay(overlay, fg=(0,0,0), bg=(255,255,255), relative_size=0.15) -> np.ndarray`
- `save_png(path, fg=(0,0,0), bg=(255,255,255), texture=None, overlay=None, relative_size=0.15) -> None`
- `_decode_with_cv2(image: np.ndarray | None = None) -> tuple[str | None, bool]`
- `validate(image: np.ndarray | None = None) -> bool`

---

### Helper functions (one-liners)

```python
make_qr(
    data: str,
    *,
    box_size: int = 10,
    border: int = 4,
    ecc: str = "M",
) -> QRCodeImage
```

```python
save_qr_png(
    path: str | Path,
    data: str,
    *,
    box_size: int = 10,
    border: int = 4,
    ecc: str = "M",
    fg: Color = (0, 0, 0),
    bg: Color = (255, 255, 255),
    texture: ImageLike | None = None,
    overlay: ImageLike | None = None,
    relative_size: float = 0.25,
) -> None
```

---

## Requirements

- Python **3.9+** (uses modern type-hint syntax)
- Required packages:
  - [`numpy`](https://pypi.org/project/numpy/)
  - [`Pillow`](https://pypi.org/project/Pillow/)
  - [`qrcode`](https://pypi.org/project/qrcode/)
- Optional (for decoding/validation only):
  - [`opencv-python`](https://pypi.org/project/opencv-python/) (`cv2`)

Install the basic dependencies:

```bash
pip install qrcode[pil] numpy Pillow
```

Install the optional dependencies:

```bash
pip install opencv-python
```

---

## Examples

### High-level approaches:

---

##### Example #1 - Generate and save a plain QR code

```python
from qr import save_qr_png

payload_text = "https://www.rit.edu/science/chester-f-carlson-center-imaging-science"

save_qr_png("images/rit_qr_plain.png",
            payload_text,
            box_size=20,
            border=4,
            ecc="H")
```
<img src="images/rit_qr_plain.png" width="300">

##### Example #2 - Generate and save a textured-modules QR code

```python
from PIL import Image
from qr import save_qr_png

texture = Image.open("images/rit_tiger.png")
payload_text = "https://www.rit.edu/science/chester-f-carlson-center-imaging-science"

save_qr_png("rit_qr_textured.png",
            payload_text,
            box_size=20,
            border=4,
            ecc="H",
            texture=texture)
```
<img src="images/rit_qr_textured.png" width="300">

##### Example #3 - Generate and save a QR code with a central overlay

```python
from PIL import Image
from qr import save_qr_png

overlay = Image.open("images/rit_tiger.png")
payload_text = "https://www.rit.edu/science/chester-f-carlson-center-imaging-science"

save_qr_png("rit_qr_textured.png",
            payload_text,
            box_size=20,
            border=4,
            ecc="H",
            overlay=overlay,
            relative_size=0.375)
```
<img src="images/rit_qr_overlay.png" width="300">

&nbsp;
### Low-level approaches:

---

##### Example #4 - Generate, save, and validate a plain QR code

```python
from qr import QRSpec, QRCodeImage

payload_text = "https://www.rit.edu/science/chester-f-carlson-center-imaging-science"

box_size=20
border=4
ecc="H"
spec = QRSpec(data=payload_text, box_size=box_size, border=border, ecc=ecc)

qr = QRCodeImage(spec)

qr.save_png("images/rit_qr_plain.png")

try:
    print("Plain QR valid?", qr.validate())  # uses internal rendering
except RuntimeError as e:
    print("Validation skipped (no OpenCV):", e)
```
<img src="images/rit_qr_plain.png" width="300">

##### Example #5 - Generate, save, and validate a textured-module QR code

```python
from PIL import Image
from qr import QRSpec, QRCodeImage

payload_text = "https://www.rit.edu/science/chester-f-carlson-center-imaging-science"

spec = QRSpec(
    payload_text,
    box_size=20,
    border=4,
    ecc="H"
)

qr = QRCodeImage(spec)

texture = Image.open("images/rit_tiger.png")

qr.save_png("images/rit_qr_textured.png",
            texture=texture,
            bg=(255, 255, 255))

try:
    # Recreate the composited image as an array
    composited = qr.render_with_texture(texture, bg=(255, 255, 255))

    # Check that it decodes back to the original payload
    is_valid = qr.validate(image=composited)
    print("Styled QR valid?", is_valid)

except RuntimeError as e:
    # OpenCV not installed; decoding features are unavailable
    print("Validation skipped (no OpenCV):", e)
```
<img src="images/rit_qr_textured.png" width="300">

&nbsp;
### Miscellaneous "one-liner" approaches:

---

##### Example 6 - Get PNG bytes for a web response

```python
from qr import make_qr

payload_text = "https://www.rit.edu/science/chester-f-carlson-center-imaging-science"

png_bytes = make_qr(payload_text).to_png_bytes()

print(png_bytes)
```
```
b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\xc2\x00\x00\x01\xc2\x08\x02\x00\x00\x00\xf3z^\x12\x00\x00\t} ...
```

##### Example 7 – Embed QR directly into HTML (Base64)

```python
import base64
from qr import make_qr

png_bytes = make_qr("https://www.rit.edu/", box_size=10).to_png_bytes()
b64 = base64.b64encode(png_bytes).decode("ascii")

html_tag = f'<img src="data:image/png;base64,{b64}" />'

print(html_tag)
```
```
<img src="data:image/png;base64,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" />
```

---

## License

This project is licensed under the MIT License.  
Copyright (c) 2025 Carl Salvaggio.

See the [LICENSE](LICENSE) file for details.

---

## Contact

**Carl Salvaggio, Ph.D.**  
Email: carl.salvaggio@rit.edu

Chester F. Carlson Center for Imaging Science  
Rochester Institute of Technology  
Rochester, New York 14623  
United States
