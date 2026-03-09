

## Project

AutoEIT: Automated Scoring and Transcription for Elicited Imitation Tasks

This repository contains my implementation for the **AutoEIT evaluation tests** for prospective **Google Summer of Code 2026** contributors.

The goal of this task is to explore methods for:

1. **Automatic Speech Recognition (ASR)** for transcribing Spanish audio recordings.
2. **Automated scoring** of elicited imitation responses using semantic similarity.

The implementation focuses on reproducible pipelines using modern NLP and speech recognition models.

---

# Repository Structure

```
AutoEIT-GSoC-Test
│
├── data/
│   ├── audio/                  # Audio recordings for transcription
│   ├── audiodata/              #CSV files with stimulus sentences with audio
│   └── transcription/        # CSV files with stimulus sentences
│
├── notebooks/
│   ├── AutoEIT_Transcription.ipynb
│   └── AutoEIT_Scoring.ipynb
│
├── scripts/
│   ├──  score_transcription.py  # Reproducible scoring pipeline
│   └──  audio_transcribe.py
│
├── results/
│   ├── AutoEIT_transcriptions.xlsx
│   └── AutoEIT_scored_output.csv
│
└── README.md
```

---

# Test I — Audio Transcription

## Objective

Generate transcriptions of Spanish elicited imitation task recordings.

Each participant recording contains approximately **30 stimulus sentences** that participants repeat.

The goal is to produce **sentence-level transcriptions** reflecting the **exact production of the learner**, including disfluencies.

Grammar or lexical errors produced by the learner are **not corrected**, only transcription errors are fixed.

---

## Method

Audio transcription was performed using the **OpenAI Whisper** automatic speech recognition model.

Whisper is a multilingual transformer-based ASR system trained on large-scale speech data and performs well on Spanish speech recognition.

Model used:

```
whisper-small
```

Reasons for choosing Whisper:

* strong multilingual speech recognition
* robust to accent variation
* reliable sentence-level transcription
* widely used in research environments

---

## Transcription Pipeline

1. Load audio recordings.
2. Use Whisper to generate transcription.
3. Extract text segments.
4. Align transcription with the **30 stimulus sentences** provided in the dataset.
5. Export results to Excel.

Example code:

```python
result = model.transcribe(audio_path, language="es")
full_text = result["text"]
```

Because Whisper segmentation may produce more segments than expected, the full transcription was generated and then aligned with the provided stimulus sentence list.

---

## Output

The transcription output is saved as:

```
results/AutoEIT_transcriptions.xlsx
```

Example structure:

| Sentence | Stimulus                                | Transcription                           |
| -------- | --------------------------------------- | --------------------------------------- |
| 1        | Me gustan las películas que acaban bien | Me gustan las peliculas que acaban bien |
| 2        | El niño juega en el parque              | El niño juega en el parque              |

---

# Test II — Automated Scoring

## Objective

Develop a reproducible script to automatically score learner responses using a **meaning-based rubric**.

The system compares:

```
Stimulus sentence
vs
Learner transcription
```

and assigns a sentence-level score.

---

## Method

Semantic similarity between the stimulus and transcription was computed using the **Sentence Transformers** framework.

Model used:

```
paraphrase-multilingual-MiniLM-L12-v2
```

This model produces multilingual sentence embeddings that allow comparison between Spanish sentences.

Similarity between embeddings is computed using **cosine similarity**.

---

## Scoring Rubric

The similarity score is mapped to the AutoEIT scoring rubric.

| Cosine Similarity | Score | Meaning           |
| ----------------- | ----- | ----------------- |
| ≥ 0.85            | 2     | Correct meaning   |
| 0.65 – 0.85       | 1     | Partially correct |
| < 0.65            | 0     | Incorrect meaning |

Example implementation:

```python
def assign_score(sim):
    if sim >= 0.85:
        return 2
    elif sim >= 0.65:
        return 1
    else:
        return 0
```

---

## Reproducible Scoring Script

The full scoring pipeline is implemented in:

```
scripts/score_transcription.py
```

Run the scoring pipeline:

```
python scripts/score_transcription.py
```

This script:

1. Loads all participant transcription files.
2. Cleans stimulus sentences.
3. Computes semantic similarity.
4. Assigns automated scores.
5. Exports results.

---

## Output

Final scoring results are saved in:

```
results/AutoEIT_scored_output.csv
```

Example output:

| Sentence | Stimulus                                | Transcription                           | Similarity | Predicted Score |
| -------- | --------------------------------------- | --------------------------------------- | ---------- | --------------- |
| 1        | Me gustan las películas que acaban bien | Me gustan las peliculas que acaban bien | 0.93       | 2               |

---

# Challenges and Observations

Several challenges were encountered during development:

### ASR segmentation

Whisper often splits audio into multiple segments based on pauses.
This resulted in more segments than the expected number of stimulus sentences.
To address this, the full transcription was generated and then aligned with the 30 stimulus sentences.

### Orthographic variation

Spanish accent differences (e.g., **películas vs peliculas**) may slightly affect similarity scores.

### Semantic similarity limitations

Embedding-based similarity captures overall meaning but may miss some grammatical differences.

Future improvements could incorporate:

* syntactic analysis
* word-level alignment
* grammar-sensitive scoring

---

# Dependencies

```
python >= 3.10
pandas
sentence-transformers
openai-whisper
torch
openpyxl
```

Install dependencies:

```
pip install pandas sentence-transformers openai-whisper torch openpyxl
```
