# Chaos-Aware Weather Forecasting — Streamlit app

Trained in Colab with TensorFlow 2.21.0, scikit-learn 1.6.1, Python 3.13.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free)
1. Create a GitHub repository and upload **all files in this folder** (keep them together, next to app.py).
2. Go to https://share.streamlit.io → *Create app* → pick the repository, branch, and `app.py`.
3. Under *Advanced settings* choose Python 3.13 (the same as Colab).
4. Deploy. The first build installs TensorFlow and takes a few minutes.

If `tensorflow-cpu==2.21.0` is not available for the chosen Python version, change that line to
`tensorflow==2.21.0`.

## Using your own data
Sidebar → *Upload CSV*. 10-minute readings with columns `date, T, p, rh, Tdew, wv, wd`
(Jena-style headers such as `T (degC)` also work), at least 25 hours long.
