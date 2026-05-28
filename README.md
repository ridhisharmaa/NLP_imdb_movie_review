# IMDB Movie Review Sentiment Analysis

This project builds and serves a deep learning sentiment classifier for IMDB movie reviews. It uses TensorFlow/Keras to train a neural network on the built-in IMDB dataset and Streamlit to provide a simple web interface where users can enter a review and classify it as positive or negative.

## Features

- Trains a sentiment classification model using the Keras IMDB dataset.
- Uses word-index encoding, sequence padding, an embedding layer, and an LSTM-based classifier.
- Saves the trained model as `simple_rnn_imdb.keras`.
- Provides a Streamlit app in `main.py` for interactive predictions.
- Includes notebooks for embedding practice, model training, and prediction testing.

## Project Structure

```text
.
|-- .devcontainer/
|   `-- devcontainer.json
|-- embedding.ipynb
|-- main.py
|-- predictions.ipynb
|-- requirements.txt
|-- simple_rnn_imdb.keras
`-- textclassification.ipynb
```

## Files

- `main.py` - Streamlit application that loads the saved model and classifies user-entered reviews.
- `textclassification.ipynb` - Notebook for loading the IMDB dataset, preprocessing reviews, training the model, and saving it.
- `predictions.ipynb` - Notebook for loading the saved model and testing sentiment predictions.
- `embedding.ipynb` - Small notebook demonstrating one-hot encoding, padding, and embedding layers.
- `simple_rnn_imdb.keras` - Saved trained Keras model used by the app.
- `requirements.txt` - Python dependencies for running the project.
- `.devcontainer/devcontainer.json` - Dev container configuration for VS Code/Codespaces.

## Requirements

- Python 3.11 recommended
- TensorFlow 2.15.0
- Keras 2.15.0
- Streamlit
- NumPy

Install dependencies with:

```bash
pip install -r requirements.txt
```

## How to Run

From the project folder, run:

```bash
streamlit run main.py
```

Streamlit will start a local web app, usually at:

```text
http://localhost:8501
```

Enter a movie review in the text area and click `Classify` to get:

- Sentiment label: `Positive` or `Negative`
- Prediction score from the model

## Model Workflow

The training workflow in `textclassification.ipynb` follows these steps:

1. Load the IMDB dataset from `tensorflow.keras.datasets.imdb`.
2. Keep the top `10,000` most frequent words.
3. Pad each review to a sequence length of `200`.
4. Build a neural network with:
   - `Embedding(max_features, 128)`
   - `LSTM(64, dropout=0.2, recurrent_dropout=0.2)`
   - `Dropout(0.3)`
   - `Dense(1, activation='sigmoid')`
5. Compile the model with Adam optimizer and binary cross-entropy loss.
6. Train with early stopping on validation loss.
7. Save the model as `simple_rnn_imdb.keras`.

## Prediction Logic

`main.py` performs prediction by:

1. Loading the IMDB word index.
2. Cleaning the review text by lowercasing and removing punctuation.
3. Converting words to IMDB word-index values.
4. Padding the encoded review to length `200`.
5. Running the saved model.
6. Returning `Positive` when the prediction score is greater than `0.69`; otherwise returning `Negative`.

## Notes

- The saved model file must remain in the same folder as `main.py`.
- The app expects words to be available in the Keras IMDB vocabulary.
- Unknown or out-of-vocabulary words are mapped to the IMDB unknown token.
- If you retrain the model, save it again as `simple_rnn_imdb.keras` or update the filename in `main.py`.
