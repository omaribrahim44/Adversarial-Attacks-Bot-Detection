# Adversarial Attacks on Graph-Based Bot Detection

This project builds a baseline Twitter bot classifier using profile + keyword-derived proxy features, then evaluates robustness under two adversarial attacks: test-time structural evasion and training-time graph poisoning.

## Dataset
- Source: Kaggle Twitter Bot Detection (50k rows).
- File: `bot_detection_data.csv` (tweet-level; includes synthetic `Hashtags`).
- Label: `Bot Label` (0 = human, 1 = bot).

## Approach
1. **Aggregate to user level** (one row per `Username`): mean `Retweet Count`/`Mention Count`, max `Follower Count`, first `Verified`, max `Bot Label`.
2. **Keyword proxy features** from `Hashtags`: `num_keywords`, `avg_kw_pop`, `max_kw_pop`, `sum_kw_pop`, `prop_rare_kw`, `kw_entropy`.
3. **Baseline model**: Random Forest (200 trees, class_weight="balanced"), features scaled with `StandardScaler`.
4. **Attacks**
   - Structural evasion (test-time): modify bot keywords to appear human-like; recompute proxy features; evaluate baseline.
   - Graph poisoning (train-time): flip ~40% of bot labels in training; retrain; evaluate on clean test.

## Key Results (from notebook run)
- Baseline accuracy ~0.53, bot recall ~0.64.
- Evasion: accuracy ~0.54, bot recall ~0.64 (minor drop).
- Poisoning: accuracy ~0.48, bot recall ~0.10 (severe collapse).

## Known Gaps / TODOs
- Keyword extraction for the co-keyword graph currently yields zero edges; inspect tokenizer/`Hashtags` parsing to ensure non-empty keyword sets before graph build.
- Performance comparison plot uses hard-coded metrics; recompute from actual reports.
- Evasion keyword pools are built from full data (includes test); rebuild from training-only to avoid leakage.
- Persist feature column order/metadata alongside saved artifacts to keep model+scaler usable.

## Files
- Notebook: `Final NoteBook.ipynb`
- Data: `bot_detection_data.csv`
- Models: `rf_baseline_model.joblib` (RF), `rf_scaler.joblib` (scaler)
- References: `Adversarial Attacks on Graphs.pdf`, `Final NoteBook.pdf`

## How to Run
1. Open `Final NoteBook.ipynb` in Jupyter (Python 3.x).
2. Install deps: `pip install pandas numpy scikit-learn networkx matplotlib seaborn tqdm joblib`.
3. Ensure `bot_detection_data.csv` is in the project root.
4. Run notebook cells in order; artifacts save to the project directory.

## Repro Tips
- Keep `random_state=42` to reproduce splits/attacks.
- If feature columns change, update the saved feature list and regenerate model+scaler.
