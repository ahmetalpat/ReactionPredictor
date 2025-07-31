---
title: Chemical Reaction Predictor
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
---

# 🧪 Chemical Reaction Predictor

This application predicts the products of chemical reactions using a state-of-the-art T5-based model.

## How to Use the App

1.  **Input Molecules**: You have two options:
    *   Use the **✍️ Chemical Drawing Tool** to draw the reactant and reagent molecules.
    *   Switch to the **⌨️ SMILES Text Input** tab and paste the SMILES strings directly.
2.  **Load Examples (Optional)**: Use the dropdown in the sidebar to load pre-defined example reactions to see how the app works.
3.  **Set Parameters**: In the sidebar, you can select the number of predictions you want to generate.
4.  **Predict**: Click the "Predict Product" button to see the results.

## About the Model

This application uses the `sagawa/ReactionT5v2-forward-USPTO_MIT` model, which has been fine-tuned for forward reaction prediction.

For more details about the model, please visit its page on the [Hugging Face Hub](https://huggingface.co/sagawa/ReactionT5v2-forward-USPTO_MIT).