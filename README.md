# QR Code Generation Utilities

This repository contains a small, self-contained Python module for generating and rendering QR codes using NumPy and Pillow, with optional validation/decoding via OpenCV.

The core idea is to **separate QR structure from QR rendering**:

- The QR content is stored as a boolean module matrix (`True` = dark module, `False` = light).
- Rendering to bitmaps (NumPy arrays, PIL images, PNG files, or stylized overlays) is handled explicitly by your code, so you stay in control of colors, compositing, and post-processing.

---

## Features

- **Immutable QR specification** via `QRSpec` (data, box size, border, ECC level).
- **Matrix-based QR representation** via `QRCodeImage`.
- **Rendering**:
  - To RGB NumPy arrays
  - To PIL `Image` objects
  - Directly to PNG bytes or files
- **Stylized overlays**:
  - Composite the QR modules over a background image
  - Modules are “textured” by the background; border and gaps remain a clean solid color
- **Optional validation** (if OpenCV is installed):
  - Decode a QR and verify it matches the expected payload

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
pip install numpy Pillow qrcode
```

&nbsp;
## Examples

### Example #1 - Generate and save a plain QR code

```python
from qr import make_qr

# Simple helper: Go straight from data -> QRCodeImage
qr = make_qr(
    "https://www.rit.edu/",
    box_size=12,        # pixels per module
    border=4,           # modules of quiet zone
    ecc="Q",            # ~25% error correction
)

# Render to an RGB NumPy array
arr = qr.render_array()
print(arr.shape)        # (height, width, 3)

# Save as a PNG file
qr.save_png("images/rit_qr_plain.png")
```

### Example #2 - Textured modules + validation

```python
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from qr import QRSpec, QRCodeImage

# Define an immutable QR specification
spec = QRSpec(
    "https://www.rit.edu/",
    box_size=16,
    border=4,
    ecc="H"        # ~30% error correction (highest rate)
)

# Build the QR code image object
qr = QRCodeImage(spec)

# Load a background/texture image as a NumPy array
bg_path = Path("images/rit_tiger.png")
background = iio.imread(bg_path)        # Shape (H, W, 3) or (H, W)

# Save a QR where only the modules are textured by `background`
qr.save_png(
    "images/rit_qr_fancy.png",
    background=background,
    bg=(255, 255, 255)        # Quiet zone and gaps stay white
)

# Optionally validate with OpenCV (if installed)
try:
    # Recreate the composited image as an array
    comp = qr.overlay_on_background(background, bg=(255, 255, 255))

    # Check that it decodes back to the original payload
    is_valid = qr.validate(image=comp)
    print("Styled QR valid?", is_valid)
    
except RuntimeError as exc:
    # OpenCV not installed; decoding features are unavailable
    print("Validation skipped (no OpenCV):", exc)
```

### Example 3 - Get PNG bytes for a web response [one liner]

```python
from qr import make_qr

png_bytes = make_qr("https://www.rit.edu/").to_png_bytes()

# ... Return "png_bytes" in a web API response, attach to email, etc.
```

### Example 4 – Embed QR directly into HTML (Base64) [one liner]

```python
import base64
from qr import make_qr

png_bytes = make_qr("https://www.rit.edu/", box_size=10).to_png_bytes()
b64 = base64.b64encode(png_bytes).decode("ascii")

html_tag = f'<img src="data:image/png;base64,{b64}" />'
print(html_tag)
```

&nbsp;
## License

This project is licensed under the MIT License.  
Copyright (c) 2025 Carl Salvaggio.

See the [LICENSE](LICENSE) file for details.

&nbsp;
## Contact

**Carl Salvaggio, Ph.D.**  
Email: carl.salvaggio@rit.edu

Chester F. Carlson Center for Imaging Science  
Rochester Institute of Technology  
Rochester, New York 14623  
United States