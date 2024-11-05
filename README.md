# LLM-Enhanced Explainability Framework

This proof-of-concept framework integrates multiple Explainable AI (XAI) methods to enhance the interpretability of machine learning models through a combination of local explanations and retrieval-augmented generation (RAG). Specifically, it combines traditional explainability techniques, such as SHAP and counterfactual explanations, with large language models (LLMs) to synthesize and communicate explanations effectively. Below, we provide a brief overview of the main components of the framework:

## Framework Overview

The proposed framework consists of several key components to provide transparent insights into machine learning model predictions. Each component is summarized below:

- **Data Preparation**:
  - **Input**: Raw dataset (here: Breast Cancer dataset).
  - **Output**: Train and test splits, standardized feature values.

- **Deep Neural Network (DNN) Model**:
  - **Input**: Standardized training data.
  - **Output**: Trained model capable of predicting binary outcomes (e.g., benign or malignant).

- **DiCE Counterfactual Explanations**:
  - **Input**: Model predictions, data instances.
  - **Output**: Counterfactual explanations indicating minimal feature changes required to alter predictions.

- **SHAP Values**:
  - **Input**: Model and data instances.
  - **Output**: SHAP values indicating feature contributions to individual predictions.

- **Document Store and Embeddings**:
  - **Input**: Generated explanations (original instances, SHAP values, counterfactuals).
  - **Output**: Indexed documents with embeddings for information retrieval.

- **Retrieval-Augmented Generation (RAG)**:
  - **Input**: User queries, indexed documents.
  - **Output**: Responses synthesized using retrieved explanation documents and contextual knowledge.

## Features

- **Local Explanation Generation**: The framework uses SHAP and DiCE to generate feature-level explanations for each instance.
- **LLM-Driven Query Response**: A large language model is used to answer user queries regarding model predictions and explanations.
- **Interactive Interface**: The framework is equipped with a Gradio-based user interface for interactive exploration of explanations.

## Installation

Clone the repository and ensure all dependencies are installed. The main dependencies include `torch`, `transformers`, `shap`, `dice_ml`, `haystack`, and `gradio`.

