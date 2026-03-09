# UFC ML UI

## Quick Start

1. Install dependencies:
   - `python -m pip install -r requirements.txt`
2. Launch the app:
   - `streamlit run ufc_ml_ui.py`

## Desktop App

- Launch the native desktop wrapper from source:
  - `.venv\Scripts\python.exe ufc_desktop_app.py`
- Build a Windows `.exe`:
  - `powershell -ExecutionPolicy Bypass -File .\build_desktop.ps1`
- The built executable will be created at:
  - `dist\UFC ML Desktop.exe`

## What the UI Does

- Lets you choose between:
  - `Use current training set` to load the saved model/features already on disk
  - `Run new training set` to scrape/build/train again and overwrite the saved outputs
- Scrape UFCStats events/fights (you control how many events)
- Build aligned fighter-profile training data
- Train the model and show:
  - accuracy
  - cross-validation stats
  - confusion matrix
  - classification report
  - held-out test prediction exports:
    - `ufc_test_predictions.csv`
    - `ufc_test_mistakes.csv`
- Select two fighters from UFCStats and run prediction
- Show model reasoning for the pick (top feature-based factors)
- Enter sportsbook American odds to calculate:
  - expected value (EV)
  - market edge versus implied probability
  - edge grades: `Pass`, `Lean`, `Playable`, `Strong`
  - full Kelly and fractional Kelly bankroll sizing
