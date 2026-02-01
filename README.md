# GelMA-Hydrogel-Mechanical- Property-Prediction
This project presents two machine learning models for predicting **hydrogel mechanical properties** .

## Features
- Predict storage modulus (G′) and loss modulus (G″)
- Classify critical stress range
- Uncertainty quantification using Bayesian inference
- Interactive web interface built with Streamlit

- ## Model
- Regression BNN: predicts G′ and G″
- Classification BNN: predicts critical stress category
- Implemented using `blitz-bayesian-deep-learning`

- ## Input Parameters
- GelMA concentration (%)
- LAP concentration (%)
- UV crosslinking time (s)

## How to Run
conda create -n GelMA python = 3.8
pip install -r requirements.txt
pip install streamlit
python BNN_MC_ANN.py
python Strain-StressClassifier.py
streamlit run app.py
