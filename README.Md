# Printed Text Scanner 

**Project Features**  
- Modern GUI (PyQt6)  
- Load image from file  
- Live webcam feed  
- Click-and-drag ROI selection with live preview  
- Automatic OCR when ROI is selected  
- Real-time text overlay (bounding boxes + detected words on the image)  
- Clean extracted text display  



-### Installation & Run (Linux)

```bash
# 1. Install system packages
sudo apt update
sudo apt install tesseract-ocr libtesseract-dev python3-opencv -y

# 2. Setup project
mkdir printed-text-scanner && cd printed-text-scanner
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install pyqt6 opencv-python pytesseract numpy

# 3. Save the code as printed_text_scanner.py
# 4. Run
python3 printed_text_scanner.py