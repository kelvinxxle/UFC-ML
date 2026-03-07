# UFC ML UI

## Quick Start

1. Install dependencies:
   - `python -m pip install -r requirements.txt`
2. Launch the app:
   - `streamlit run ufc_ml_ui.py`

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
- Select two fighters from UFCStats and run prediction
- Show model reasoning for the pick (top feature-based factors)
- Enter sportsbook American odds to calculate:
  - expected value (EV)
  - market edge versus implied probability
  - edge grades: `Pass`, `Lean`, `Playable`, `Strong`
  - full Kelly and fractional Kelly bankroll sizing
