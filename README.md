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