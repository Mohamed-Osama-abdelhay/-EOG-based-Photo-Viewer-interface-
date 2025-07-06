# Eye Movement Controlled Photo Viewer

This project is a Human-Computer Interaction (HCI) application that uses EOG (Electrooculography) signals to control a photo viewer interface. The system classifies eye movements (left, right, blink, etc.) to allow hands-free navigation through images.

## 🧠 Features

- Reads and preprocesses horizontal and vertical EOG signals.
- Extracts features using:
  - Wavelet decomposition
  - Autoregressive modeling
- Trains an SVM model for classification.
- Provides a GUI for browsing photos using eye movements.

## 🗂️ Project Structure

- `project.py` - Core logic for reading signals, preprocessing, feature extraction, and model training.
- `saved Features.py` - Utility for saving preprocessed data to Excel.
- `main.py` - GUI application that uses the trained model to navigate images based on eye movements.

## 🧪 Signal Classes

Supports classification for:
- `Up`
- `Down`
- `Right`
- `Left`
- `Blink`

## 🛠️ How It Works

1. **Data Input**: Reads signals from `.txt` files grouped by movement class.
2. **Preprocessing**:
   - Bandpass filtering
   - Downsampling
   - Normalization
   - DC offset removal
3. **Feature Extraction**:
   - Wavelet (db4)
   - Autoregressive coefficients
4. **Model Training**:
   - SVM with RBF kernel
   - Trained on extracted features with 75/25 train/test split
5. **GUI Usage**:
   - Upload two signal files.
   - Predictions navigate to previous or next images based on the detected movement.

## 🖥️ GUI Controls

- `Upload Signal 1` and `Upload Signal 2`: Load the horizontal and vertical signal files.
- `OK`: Classify the signals and navigate images (left or right).
- `<< Previous` and `Next >>`: Manual navigation for testing.

## 📁 Data File Expectations

- Signal files must be in plain text format containing integer values.
- File naming and structure should follow the convention used in `read_signals()` inside `project.py`.

## 📦 Dependencies

```bash
pip install numpy scipy pandas scikit-learn statsmodels pywavelets pillow openpyxl
