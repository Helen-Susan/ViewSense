# Indian Currency Detection 🇮🇳💰 (YOLO26)

This project implements a real-time **Indian Currency Detection System** using the **YOLO26** architecture (via Ultralytics). It is designed to detect and classify Indian currency notes and coins (10, 20, 50, 100, 200, 500, 2000 INR) with high accuracy and speed, optimized for GPU training on Windows.

## 🚀 Features
- **Model**: YOLO26 (Medium) - `yolo26m.pt`
- **Task**: Object Detection (Bounding Boxes)
- **Dataset Source**: Roboflow (`yolo26` format)
- **Hardware Support**: NVIDIA GPU acceleration (CUDA 12.1)

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/anazmuhdd/main-project-sw.git
cd main-project-sw
```

### 2. Create Virtual Environment
It is recommended to use a virtual environment to manage dependencies.
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
This project requires **PyTorch with CUDA support** and the `ultralytics` library.

```bash
# Install core requirements
pip install -r requirements.txt
```

> **Note for Windows Users with NVIDIA GPUs:**
> If you encounter issues with CUDA detection, ensure you have the correct PyTorch version installed:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --upgrade --force-reinstall
> ```
> Also ensure `zlibwapi.dll` is in your `.venv/Scripts/` folder or CUDA `bin` directory if using cuDNN.

### 4. GPU Verification
Run the following python snippet to verify your GPU is detected:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

---

## 🏋️ Training the Model

The training script `yolo26train.py` handles dataset downloading (from Roboflow), analysis, and training loop.

### Run Training
```bash
python yolo26train.py
```

**What this script does:**
1.  Downloads the dataset from Roboflow in `yolo26` format.
2.  Analyzes the dataset (prints class counts and image distribution).
3.  Loads the **YOLO26m** (Medium) pretrained model.
4.  Starts training for **100 epochs** on **GPU 0** with batch size **16**.
5.  Saves results to `yolo26_currency_run/`.

---

## 🔍 Inference / Testing

Once training is complete, you can run inference on new images or videos using the CLI.

```bash
# Run inference on an image
yolo predict model=yolo26_currency_run/weights/best.pt source='path/to/image.jpg' show=True

# Run inference on a video
yolo predict model=yolo26_currency_run/weights/best.pt source='path/to/video.mp4' show=True

# Run inference on live webcam
yolo predict model=yolo26_currency_run/weights/best.pt source=0 show=True
```

---

## 📂 Project Structure
```
.
├── .gitignore          # Git ignore rules
├── currency_model.py   # (Legacy) Old TensorFlow script
├── requirements.txt    # Python dependencies
├── yolo26train.py      # Main YOLO26 training script
└── README.md           # Project documentation
```

## ⚠️ Common Issues
*   **"Could not locate zlibwapi.dll"**: Download `zlibwapi.dll` and place it in `C:\Windows\System32` or your virtual env `Scripts/` folder.
*   **CUDA Out of Memory**: Open `yolo26train.py` and reduce `batch=16` to `batch=8` or `batch=4`.
