# Face Recognition Project

## Project Description

This project builds a face recognition system for a fixed set of 31 people using **ArcFace** embeddings extracted via the `deepface` library. Rather than training a classifier from scratch, the pipeline uses a pre-trained ArcFace backbone to convert each face into a 512-dimensional embedding, then performs recognition via cosine-distance nearest-neighbor matching against a reference embedding database. The recognition threshold is optimized on a held-out validation split (Macro F1), and the final system is evaluated on an unseen test split with accuracy, per-person precision/recall/F1, and a confusion matrix. ArcFace is additionally benchmarked against **Facenet** and **VGG-Face** backbones for comparison. The final reference embeddings, labels, backbone name, and tuned threshold are exported to a single reusable file, and a Gradio interface is included for interactive, real-time face recognition from an uploaded photo or webcam.

## How to Run the Code

1. Clone the repository:

```
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

2. Install dependencies: It's recommended to create a virtual environment (Python 3.9–3.12 required for TensorFlow/DeepFace compatibility):

```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

3. Run the Notebook: Open `Face_Recognition_Project_Finalize.ipynb` in Google Colab or a local Jupyter environment and execute all cells sequentially:

   - **Install dependencies** — `deepface`, `scikit-learn`, `opencv`, etc.
   - **Load the dataset** — one folder per person under `Dataset/Faces/`
   - **Create Reference / Validation / Test splits** — 70% / 15% / 15% per person, fixed random seed
   - **Build reference embeddings** — `DeepFace.represent()` with the ArcFace backbone (detect → align → embed)
   - **Optimize the recognition threshold** — swept from 0.20 to 0.60 on the validation split, selected by best Macro F1
   - **Evaluate on the test split** — classification report, confusion matrix, error inspection, sample predictions
   - **Compare backbones** — Facenet, ArcFace, and VGG-Face benchmarked on accuracy and runtime
   - **Save the model** — reference embeddings, labels, backbone name, and threshold exported to `face_recognition_model.joblib`

   Note: If running in Google Colab, grant Drive access if prompted, and make sure the dataset path (`Dataset/Faces/`) is accessible to the notebook.

4. Run the Gradio App: Once the notebook has been run and `face_recognition_model.joblib` has been generated, launch the interactive app:

```
pip install -r requirements.txt
python app_gradio.py
```

This opens a Gradio interface where you can upload a photo (or use your webcam), adjust the recognition threshold live, and get an instant identity prediction with a top-5 closest-matches table.

## Libraries Required

All necessary Python libraries are listed in `requirements.txt`. Install them using:

```
pip install -r requirements.txt
```

Key libraries used across the notebook and app include `deepface`, `opencv-python-headless`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `joblib`, and `gradio`.

## Project Structure

```
├── Face_Recognition_Project_Finalize.ipynb   # Data prep, embedding, threshold tuning, evaluation, backbone comparison
├── face_recognition_model.joblib             # Saved reference embeddings, labels, backbone name, and threshold
├── app_gradio.py                             # Gradio app for interactive face recognition
├── requirements.txt
├── README.md
└── Dataset/
    └── Faces/                                # One folder per person, ~34 images each (1,030 images, 31 people)
```

## Algorithm Implementation Summary

* **ArcFace** (DeepFace backbone): pre-trained embedding model, 512-dim output — **winning backbone**, selected via Test Macro F1
* **Facenet**: benchmarked as an alternative embedding backbone
* **VGG-Face**: benchmarked as an alternative embedding backbone
* **Recognition rule**: `1 − cosine similarity` between query embedding and each reference embedding; nearest reference wins if its distance is below the tuned threshold, otherwise the query is labeled `"Unknown"`
* **Threshold tuning**: swept 0.20–0.60 in steps of 0.01 on the validation split, selected by best Macro F1 (final value: `0.54`)

Each backbone was benchmarked independently on accuracy, precision, recall, and Macro F1, then compared in the notebook to select the final production model.
