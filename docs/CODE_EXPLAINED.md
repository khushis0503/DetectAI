# DetectAI — Every File Explained (and Viva Preparation)

This document explains, in plain language, what every code file in DetectAI does, **which NLP syllabus topic each
part demonstrates**, and ends with **25 likely viva questions** answered with *our own* results.
Every number below was produced by running our code (sources: `results/*.json`, `backend/models/thresholds_*.json`).

---

## 0. Syllabus map — where each topic lives

| Syllabus topic | Unit | Where in DetectAI |
|---|---|---|
| Tokenization (sentence + word) | 1 | `preprocessing.py` (`word_tokenize`, `sent_tokenize`), `build_dataset.py` (sentence-boundary cuts), `detector.py` (sentence spans) |
| Sub-word tokenization (WordPiece) & morphology | 1 | Notebook step 6, `detector.py` (BERT tokenizer, chunking) |
| Ambiguity | 1 | `augment.py` (WordNet synonyms picked without context), notebook step 6/14 |
| NLP libraries (NLTK, spaCy, Hugging Face) | 1 | NLTK: `preprocessing.py`, `augment.py`; Hugging Face: notebook, `detector.py`, `build_dataset.py` (HC3 download). *spaCy: not used — see Q6 note* |
| Text cleaning | 2 | `preprocessing.py` (`clean_text`, `strip_markdown`), `build_dataset.py` (`fix_hc3_spacing`, quote normalisation) |
| Stopword removal | 2 | `preprocessing.py` (`classical_preprocess`), `train_baselines.py` (stopwords removed vs kept experiment) |
| Stemming & lemmatization | 2 | `preprocessing.py` (Porter stemmer, WordNet lemmatizer) |
| Bag of Words | 2 | `train_baselines.py` (`CountVectorizer` + Naive Bayes) |
| TF-IDF (word and character n-grams) | 2 | `train_baselines.py`, `detector.py` (coefficient × TF-IDF explanations) |
| Word embeddings vs contextual embeddings | 2 | BERT's input embedding layer (static) vs its 12 Transformer layers (contextual) — `detector.py` Gradient×Input, notebook steps 8 and 15 |
| POS tagging | 3 | `preprocessing.py` (`stylometric_features`, NLTK perceptron tagger, Penn Treebank tags) |
| Naive Bayes, Logistic Regression, SVM | 4 | `train_baselines.py` |
| Evaluation metrics (accuracy, precision, recall, F1, confusion matrix, ROC-AUC) | 4 | `train_baselines.py`, `robustness_test.py`, `calibrate.py`, notebook step 11 |
| Transformers and attention | 4 | Notebook steps 8 and 16 (last-layer self-attention analysis) |
| NLP applications | 5 | The whole project: AI-text detection in education, per-domain evaluation |
| Ethics and bias | 5 | 3-verdict design + 5% false-positive rule (`calibrate.py`), disclaimer, error analysis (notebook step 17), leak fixes, `saliency_check.py` |

---

## 1. The pipeline in one picture

```
 raw corpora ──► build_dataset.py ──► data/detectai_dataset.csv (8,895 texts, fixed train/val/test split)
                                            │
             augment.py ◄───────────────────┤  (edited AI copies, TRAIN only)
                                            │
   train_baselines.py ──► 5 classical models + figures + baseline_metrics.json
   robustness_test.py ──► robustness.json
   Colab notebook     ──► fine-tuned BERT + bert_metrics.json + figures
                                            │
   calibrate.py  ──► temperature, ensemble decision, verdict thresholds (validation set)
   saliency_check.py ──► are the highlights faithful?
                                            │
   backend/detector.py + app.py ──► http://127.0.0.1:8000  (frontend/)
```

---

## 2. File by file

### 2.1 `scripts/build_dataset.py` — building the dataset
**Rubric: Dataset Preparation & Preprocessing.**

| Part | What it does in plain words | Topic |
|---|---|---|
| `DOMAINS`, `AI_FOLDERS` | Reads ~1,000 human and ~1,000 ChatGPT texts from each Ghostbuster domain (student essays, Reddit WritingPrompts stories, Reuters news). AI texts are mixed from 5 different prompting styles so the model sees variety. Other folders in the corpus are not used. | Data collection |
| `list_text_files`, `is_token_dump` | Skips every `logprobs*` folder and any file that is a list of "token probability" lines instead of prose. A folder called `logprobs_babbage` slipped past a simple name check at first, and 129 of its files got into our first dataset version. They contain the GPT tokenizer's `Ġ` symbol — a perfect AI give-away. | Data cleaning, leak prevention |
| `normalise_quotes` | Turns curly quotes ’ “ ” into straight ones. Human source files had about 11× more curly apostrophes (3.33 vs 0.31 per 1,000 words; results/data_audit.json) — a formatting artefact, not style. | Text cleaning |
| `fix_hc3_spacing` | HC3 human answers were stored word-tokenised ("you 're", "end .", "( i.e. )"). We glue them back ("you're", "end.") for **both** classes, and remove a literal `\n` that only ChatGPT answers had, and Wikipedia maths markup. Before the fix, in the Reddit (ELI5) answers "space before punctuation" occurred 18 times per 1,000 characters in human answers vs 0.01 in ChatGPT answers — a huge shortcut. | Tokenization (detokenization), cleaning |
| `collect_hc3` | Takes a balanced 300 human + 300 ChatGPT answers from each HC3 subset (reddit_eli5, open_qa, wiki_csai, finance, medicine) = 1,500 + 1,500. This casual Q&A text is closest to what a teacher will type in the live demo. | Data collection, domain coverage |
| `cut_to_random_length` | Uses NLTK **sentence tokenization** to cut every text at a sentence boundary to a random length of 20–300 words. Both classes then have the same length distribution, so length cannot be a shortcut, and the model learns short texts too. | Sentence tokenization |
| HC3 length matching | Raw HC3 human answers are much shorter (median 83 vs 174 words). Each ChatGPT answer is cut to a length drawn from the human lengths, giving medians of 61 vs 63 words. | Bias/shortcut removal |
| filters + split | Drops texts under 20 or over 300 words and exact duplicates; makes a **stratified 70/15/15 split** (seed 42) on label × domain → 6,226 train / 1,334 validation / 1,335 test. | Dataset preparation |
| `strip_markdown` (last step) | Removes Reddit markdown (`*emphasis*`, `(URL_0)`, `> quote`), which appeared in 303 human texts but only 6 AI texts. Done **after** splitting so the rows and the split stay exactly the same as the version BERT v1 was trained on. | Cleaning, leak prevention |

**Result:** 8,895 texts — 4,490 human, 4,405 ChatGPT — in 4 domains: Q&A (HC3) 3,000, essay 1,992, wp 1,998, reuter 1,905.

### 2.2 `backend/preprocessing.py` — cleaning, classical NLP pipeline, style features
**Rubric: NLP Concepts & Methodology; Dataset Preparation & Preprocessing.**

| Function | What it does | Topic |
|---|---|---|
| `clean_text` | *Light* cleaning for BERT: Unicode normalisation (NFKC), HTML entities and tags, URLs, curly quotes, markdown, extra whitespace. **Keeps punctuation and stopwords** because BERT uses them as context. Used for training data *and* live input, so both are treated the same. | Text cleaning |
| `strip_markdown` | Removes `*`, `_emphasis_`, `(URL_0)` placeholders and `> ` quote markers. | Text cleaning |
| `classical_preprocess` | The classic pipeline for Bag-of-Words/TF-IDF: **lowercase → word tokenization** (`word_tokenize`) → keep alphabetic tokens → **stopword removal** (NLTK English list, optional) → **lemmatization** (WordNet: "studies" → "study") or **stemming** (Porter: "studies" → "studi"). The demo at the bottom of the file shows both. | Tokenization, stopwords, stemming vs lemmatization, morphology |
| `stylometric_features` | 13 hand-made writing-style numbers: average and standard deviation of sentence length, **type-token ratio** (distinct words ÷ all words), **hapax ratio** (words used once), average word length, stopword ratio, punctuation density, commas per sentence, and **POS-tag ratios** for nouns (NN*), verbs (VB*), adjectives (JJ*), adverbs (RB*) and pronouns (PRP*, WP*) using NLTK's averaged-perceptron **POS tagger** (Penn Treebank tagset). | POS tagging, stylometry |
| `TAGGER = PerceptronTagger()` | Loads the POS tagger once. `nltk.pos_tag` re-loads it on every call — fixing this made feature extraction about 10× faster (≈62 ms → ≈6 ms per text). | Engineering |
| `batch_*`, `load/save_stylo_cache` | Small wrappers so a saved scikit-learn pipeline can take raw text; a cache so POS tagging runs only once per text. | Implementation |

Typical training-set values (median human vs AI): sentence-length variation **6.2 vs 4.9** words (AI sentences are more uniform),
average word length **4.85 vs 5.18** letters, commas per sentence **1.00 vs 1.25**, adverbs **3.3% vs 2.3%**, adjectives **7.5% vs 8.7%**.

### 2.3 `scripts/augment.py` — edited AI copies (training data augmentation)

| Function | What it does | Topic |
|---|---|---|
| `synonym_replace` | Replaces ~15% of non-stopwords with a random **WordNet** synonym. The synonym is picked *without looking at context*, so it is often wrong — from our data: "advanced nursing **practice**" → "advanced nursing **recitation**", and "the utilization of **CIs**" (cochlear implants) → "the utilization of **Curie**". A live example of **lexical ambiguity**: WordNet lists every sense of a word and we do no word-sense disambiguation. | WordNet, ambiguity |
| `reorder_sentences` | Shuffles sentence order. | Sentence tokenization |
| `remove_fillers` | Deletes up to 4 filler words ("very", "additionally", "moreover"…). | — |
| `main` | Makes one edited copy of every AI text in the **train split only** (3,083 copies). Validation/test are never touched. | Data augmentation |

### 2.4 `scripts/train_baselines.py` — five classical models + EDA
**Rubric: Implementation; Results, Analysis & Evaluation.** Each model is a scikit-learn `Pipeline` that takes raw text, so it can be saved and reused by the website.

| Model | How it works (plain words) | Topic |
|---|---|---|
| **BoW + Naive Bayes** | `CountVectorizer` counts each word (**Bag of Words** — order is lost). Multinomial **Naive Bayes** uses Bayes' rule, P(class \| words) ∝ P(class) × Π P(word \| class), assuming words are independent given the class. | BoW, Naive Bayes |
| **TF-IDF + Logistic Regression** (stopwords removed / kept) | **TF-IDF** weighs a word by how often it appears in this text (TF) times how rare it is across all texts (IDF = log of N ÷ number of texts containing it), for single words and word pairs (1–2-grams). **Logistic Regression** learns one weight per feature; the weighted sum goes through a sigmoid to give P(AI). Trained twice to test whether stopwords help. | TF-IDF, n-grams, Logistic Regression, stopwords |
| **Char n-gram TF-IDF + SVM** | TF-IDF over 3–5-character pieces (catches spelling, punctuation and word endings). A **linear SVM** finds the boundary with the widest margin between classes; `CalibratedClassifierCV` converts its scores into probabilities. | Character n-grams, SVM |
| **Stylometric + LR** | The 13 style features (including POS ratios), standardised, into Logistic Regression. | POS tagging, stylometry |
| `evaluate` | Accuracy, precision, recall, F1 (AI = positive class), ROC-AUC, per-domain accuracy, confusion matrix. | Evaluation metrics |
| figures | Class balance, length distribution, style boxplots, confusion matrices, ROC curves, model comparison, per-domain accuracy, top TF-IDF words. | Analysis |
| `top_words` | The words with the largest negative (human) and positive (AI) LR weights. | Interpretability |

Top AI terms (stopwords kept): *upon, potential, these, "it is", introduction, "once upon", "this essay", including, significant, important to*.
Top human terms: *said, the, is, most, et, of, all, then, people, percent*.

### 2.5 `scripts/robustness_test.py` — light-editing attack
Applies the same three edits as `augment.py` (different random seed) to the **AI texts of the test set** and re-scores every saved model.
Human texts stay unchanged, so any change comes from edited AI text. Output: `results/robustness.json` + bar chart. **Topic:** robustness, ethics (evasion).

### 2.6 `scripts/calibrate.py` — turning probabilities into decisions
**Rubric: Results, Analysis & Evaluation; ethics.** Everything is chosen on the **validation** set; the test set is only reported.

| Step | What it does |
|---|---|
| Score cache | BERT on a CPU needs ~10 minutes for 2,669 texts, so scores are cached per model file. |
| **Temperature scaling** | Finds one number T that makes probabilities honest: p = sigmoid(log-odds ÷ T). It never changes the ranking or the 0.5 decision, so accuracy/F1/AUC are unchanged. BERT: **T = 2.45**, validation log-loss **0.264 → 0.146** (it was over-confident). Baseline: T = 0.64 (slightly under-confident). |
| **Ensemble** | p = w × main model + (1 − w) × char SVM; w picked on validation ROC-AUC, kept only if test ROC-AUC improves. Baseline: w = 0.55 → **kept** (test accuracy 0.890 → 0.899, AUC 0.967 → 0.971). BERT: best w = 1.0 → **not used**. |
| **Three verdicts** | "Likely AI" if p ≥ t_ai, where at most **5% of human validation texts** reach t_ai. "Likely Human" if p ≤ t_human, where at most 5% of AI texts fall below *and* p ≤ 0.5. In between: "Uncertain". BERT: t_ai = 0.898, t_human = 0.500. Baseline: 0.626 / 0.258. |
| Style reference | Median human / AI value of each style feature (training set) for the website's style panel. |

### 2.7 `scripts/saliency_check.py` — are the highlights trustworthy?
A **deletion test**: for correctly classified test texts, delete the 5 words the explanation marks as most important, and compare with
deleting 5 random words. Measured in log-odds (because BERT's raw probabilities are saturated near 0/1).
Results (`results/saliency_faithfulness.json`):

| Explanation | Texts | Evidence drop, top-5 deleted | Evidence drop, 5 random deleted | Top-5 beats random |
|---|---|---|---|---|
| BERT Gradient×Input | 147 | 0.29 | −0.13 | **56%** of texts |
| Baseline coefficient × TF-IDF | 140 | 0.97 | 0.04 | **100%** of texts |

**Topic:** explainability, ethics.

### 2.8 `scripts/plot_style.py`
Shared colours (blue = human, orange = AI, a colour-blind-checked palette in a fixed order) and fonts for every figure, so the report and slides look consistent.

### 2.9 `backend/detector.py` — the prediction engine
**Rubric: Implementation.**

| Part | What it does | Topic |
|---|---|---|
| `__init__` | Loads **BERT** if `backend/models/detectai-bert/` exists, otherwise the TF-IDF + LR baseline; always loads the char SVM; reads the matching `thresholds_<model>.json` (temperature, ensemble, thresholds). | Hugging Face Transformers |
| `_bert_chunks` | BERT reads at most 320 WordPiece tokens. Longer texts are split **between sentences** into chunks of ≤ 318 tokens (+ `[CLS]`, `[SEP]`). | WordPiece tokenization |
| `main_proba` | BERT: probability for each chunk, combined as a token-weighted average. Baseline: the scikit-learn pipeline. | Transformers / LR |
| `apply_temperature`, `score` | Temperature scaling, then the ensemble if calibration chose it. | Calibration |
| `_bert_word_scores` | **Gradient × Input**, exactly as in the notebook: take each token's *input embedding*, compute how the score (AI logit − human logit) changes when that embedding changes (the gradient), multiply by the embedding and sum. WordPieces are merged back into words. Positive = pushes to AI. | Word embeddings, explainability |
| `_tfidf_word_scores` | Baseline explanation: each word's **LR coefficient × its TF-IDF value** (exact for a linear model); bigram contributions are split between the two words; lemmas are mapped back to the original words. | TF-IDF, lemmatization |
| `predict` | Cleans each paragraph, scores the whole text, decides the verdict, computes **confidence** (distance from 50%, reduced for texts under 50 words), word highlights, **per-sentence** scores, **per-paragraph** (or per-chunk) breakdown, and style features. | Application |

### 2.10 `backend/app.py` — the web server (FastAPI)
- `POST /api/predict` — validates input (empty, under 5 words, over 3,000 words → clear error messages), calls the detector.
- `GET /api/health`, `GET /api/model-info` (model name, thresholds, temperature, test metrics).
- Enables CORS, downloads missing NLTK data on first start, and serves the `frontend/` folder at `/`.

### 2.11 `frontend/index.html`, `style.css`, `app.js` — the website
- **index.html:** text box with word counter, 3 sample buttons (real held-out test texts: casual human, formal human, ChatGPT), Analyze button, verdict card, gauge, highlighted words with legend, paragraph breakdown, sentence scores, writing-style panel, "How it works", disclaimer.
- **style.css:** large high-contrast type (19 px base) for a projector; blue/grey/orange for human/uncertain/AI; works on phones too.
- **app.js:** sends the text to `/api/predict` and draws the results: a semicircle gauge with the three verdict zones taken from the calibrated thresholds, word colours whose strength follows the explanation score, probability bars, and style tracks showing "this text" against the typical human and AI values. No external libraries or internet needed.

### 2.12 `run.bat` and `backend/requirements.txt`
`run.bat` creates the virtual environment and installs the pinned packages on first run, starts the server and opens the browser.
`requirements.txt` pins exact versions (CPU-only PyTorch).

### 2.13 `notebooks/DetectAI_BERT_Training_Colab.ipynb` — fine-tuning BERT
Every code cell is followed by a plain-language explanation in the notebook itself. Summary:

| Step | What it does | Topic |
|---|---|---|
| 1–2 | Installs pinned `transformers 5.19.0`, `accelerate`, `nltk`; checks the T4 GPU. | NLP libraries |
| 3 | All settings: lr 2e-5, batch 16, ≤ 4 epochs, warm-up 10%, weight decay 0.01, max length 320, seed 42. | Fine-tuning |
| 4–5 | Upload; checks the dataset is the cleaned version; uses the fixed split; augmented copies in training only. | Dataset preparation |
| 6 | **WordPiece demo** ("Unbelievably" → `un ##bel ##ie ##va ##bly`; `[CLS]`, `[SEP]`, token IDs) and token-length histogram: median 144 tokens, 95th percentile 355, 11.6% of texts longer than 320. | Tokenization, morphology |
| 7 | Input IDs, attention mask, labels; dynamic padding. | Preprocessing for Transformers |
| 8 | `bert-base-uncased` + classification head, Hugging Face `Trainer`, fp16, early stopping on validation F1. | **Transformers, self-attention, contextual embeddings, transfer learning** |
| 9 | Training curves (loss, validation F1/accuracy). | Evaluation |
| 10 | Saves model + tokenizer. | — |
| 11 | Test metrics, confusion matrix, ROC curve. | Evaluation metrics |
| 12–13 | Comparison with the 5 baselines; per-domain accuracy. | Model comparison, applications |
| 14 | Robustness to light editing (same edits and seed as Phase 1). | Robustness, WordNet, ambiguity |
| 15 | **Gradient × Input** word saliency for an AI and a human example; global word importance over 300 test texts. | Explainability |
| 16 | **Last-layer attention** heat-map, where `[CLS]` attends, with the "attention is not explanation" caveat. | Attention |
| 17 | Error analysis: false positives vs false negatives, most confident mistakes, accuracy by length. | Ethics and bias |
| 18 | Saves `bert_metrics.json`, zips everything and downloads it in 80 MB parts (+ `copy /b` instructions). | — |

---

## 3. Concept box: word embeddings vs contextual embeddings
- **Bag of Words / TF-IDF:** a word is just a column in a big table. "bank" is the same column in "river bank" and "bank loan"; similar words ("big", "large") are unrelated columns.
- **Static word embeddings** (Word2Vec, GloVe): each word gets one dense vector, so similar words get similar vectors — but "bank" still has *one* vector for both meanings. In BERT, the **input embedding layer** is exactly this kind of lookup table (one vector per WordPiece). Our Gradient×Input explanation is computed on these input embeddings.
- **Contextual embeddings** (BERT): after 12 layers of self-attention, each token's vector depends on the whole sentence, so "bank" near "river" and "bank" near "loan" end up different. This is what lets BERT notice *how* something is said, not just *which* words appear.

---

## 4. Key results (for quick reference)

| Model (test set, 1,335 texts) | Acc | Prec | Rec | F1 | AUC | Q&A | essay | reuter | wp |
|---|---|---|---|---|---|---|---|---|---|
| BoW + Naive Bayes | 0.813 | 0.818 | 0.802 | 0.810 | 0.912 | 0.676 | 0.716 | 0.965 | 0.973 |
| TF-IDF + LR, stopwords removed | 0.870 | 0.844 | 0.903 | 0.873 | 0.954 | 0.751 | 0.880 | 0.965 | 0.947 |
| TF-IDF + LR, stopwords kept | 0.890 | 0.853 | 0.939 | 0.894 | 0.967 | 0.802 | 0.893 | 0.969 | 0.943 |
| Char n-gram + SVM | 0.882 | 0.955 | 0.799 | 0.870 | 0.956 | 0.769 | 0.933 | 0.965 | 0.920 |
| Stylometric + LR | 0.662 | 0.611 | 0.877 | 0.720 | 0.759 | 0.607 | 0.542 | 0.706 | 0.823 |
| **BERT v1 (Colab)** | **0.945** | 0.915 | **0.979** | **0.946** | **0.993** | **0.904** | 0.923 | **0.986** | **0.987** |

- BERT confusion matrix: 614 humans correct, **60 humans flagged as AI**, **14 AI missed**, 647 AI caught. Of the 60 false positives, 36 are short Q&A answers and 17 essays; the median misclassified text is only 54 words long.
- BERT trained for 10.5 minutes on a T4; validation F1 rose every epoch (0.915 → 0.925 → 0.942 → 0.957), so epoch 4 was kept.
- Robustness (accuracy, original → AI texts lightly edited): BERT 0.945 → 0.948; TF-IDF LR 0.890 → 0.894; char SVM 0.882 → 0.891.
- Last-layer attention from `[CLS]`: 32.5% goes to punctuation tokens, 2.1% to `[SEP]`, 1.8% to `[CLS]`, 63.6% to words.
- Website verdicts on the test set with BERT: humans → 91.0% Likely Human, 2.8% Uncertain, **6.2% Likely AI**; AI → **95.2% Likely AI**, 2.7% Uncertain, 2.1% Likely Human.

---

## 5. Viva preparation — 25 likely questions

**1. What problem does DetectAI solve, and why does it matter?**
Teachers and editors increasingly receive text that may have been written by ChatGPT. DetectAI estimates the probability that a text is AI-generated and shows *why*, so a human can make an informed judgement. It is an NLP text-classification application (Unit 5) with an ethics focus: it is designed to avoid falsely accusing people.

**2. Where does your data come from, and why two datasets?**
Ghostbuster gives formal text in three domains (essays, Reddit stories, Reuters news); HC3 gives casual question answers. The live demo uses casual typed text and pasted chatbot answers, which look like HC3, not like news. Adding HC3 exposed a real weakness: it is the hardest domain for every model (best baseline 0.802, BERT 0.904). In total: 8,895 texts, 4,490 human and 4,405 ChatGPT.

**3. What data leaks did you find, and how did you fix them?**
Six shortcuts that a model could use without understanding writing style:
1. Token-probability files (lists of tokens with the GPT `Ġ` space symbol, not prose) hidden in a `logprobs_babbage` subfolder of the AI essays; 129 of them got into our first dataset version → filtered out by folder name and by content.
2. Curly apostrophes about 11× more frequent in human source files → normalised.
3. HC3 human answers stored word-tokenised ("you 're") → detokenised for both classes.
4. A literal `\n` only in ChatGPT answers → removed.
5. Raw HC3 human answers much shorter (median 83 vs 174 words) → lengths matched (final medians 63 vs 61 words).
6. Reddit markdown (`*word*`, `(URL_0)`) in 303 human vs 6 AI texts → removed after the split, so the split did not change.

The lesson: always check *what* a classifier could be learning, not only how accurate it is.

**4. Why did you cut every text to a random 20–300 words?**
ChatGPT and humans write different lengths. If we kept natural lengths, a model could simply learn "long = AI". Cutting both classes to the same random length distribution (at sentence boundaries) removes that shortcut and also teaches the model to handle short texts like the ones in a live demo.

**5. How did you split the data and avoid tuning on the test set?**
A fixed, stratified 70/15/15 split on label × domain with seed 42: 6,226 train, 1,334 validation, 1,335 test. Augmented copies go into training only. Early stopping, temperature, ensemble weight and thresholds are all chosen on validation. The test set is used once, for reporting. One exception: the decision whether to *keep* the ensemble was checked on the test set, as the brief required; we state that openly.

**6. Explain your preprocessing. Why is it different for TF-IDF and BERT?**
For TF-IDF: lowercase → word tokenization (NLTK) → keep alphabetic tokens → optional stopword removal → WordNet lemmatization, because a count-based model benefits from merging word forms. For BERT: only light cleaning (Unicode, URLs, HTML, markdown, whitespace), keeping stopwords, punctuation and word forms, because BERT reads them as context and has its own WordPiece tokenizer. *(We used NLTK and Hugging Face; spaCy offers the same tokenization/POS/lemmatization, but NLTK covered everything we needed with fewer dependencies.)*

**7. Why did you keep stopwords?**
Our experiment: TF-IDF + LR with stopwords **kept** scored accuracy 0.890 / F1 0.894, versus 0.870 / 0.873 with stopwords **removed**. For authorship and style, function words are the signal: our top AI features include *"it is"*, *"these"*, *"upon"*, *"important to"*, and top human features include *"the"*, *"is"*, *"of"*, *"so"*. Stopword removal is useful for topic classification, not for style detection.

**8. Stemming or lemmatization — which and why?**
Lemmatization (WordNet): "studies" → "study", a real dictionary word. The Porter stemmer chops suffixes by rule ("studies" → "studi", "quickly" → "quickli"), which is faster but produces non-words and merges unrelated forms. Our preprocessing file shows both side by side.

**9. What is the difference between Bag of Words and TF-IDF?**
BoW counts how often each word occurs. TF-IDF multiplies that count (term frequency) by the inverse document frequency, log(N ÷ number of documents containing the word), so words that appear everywhere get less weight and distinctive words get more. Both ignore word order. In our results BoW + Naive Bayes reached 0.813 accuracy versus 0.890 for TF-IDF + LR.

**10. How does Naive Bayes work, and why was it your weakest word model?**
It applies Bayes' rule and assumes every word is independent of the others given the class, then multiplies word probabilities. That assumption is false for language, and raw counts over-weight frequent words. It did fine on news and stories (0.965, 0.973) but poorly on essays (0.716) and Q&A (0.676), where both classes share vocabulary and style differences are subtle.

**11. Compare Logistic Regression and SVM in your results.**
TF-IDF + LR had better recall (0.939) and F1 (0.894). The character n-gram SVM had by far the best precision (0.955): only 25 of 674 humans were wrongly flagged. LR learns a probability with a sigmoid; a linear SVM finds the widest-margin boundary and needed calibration to output probabilities. Averaging them (55/45) gave the best baseline: accuracy 0.899, ROC-AUC 0.971.

**12. What did the stylometric model show?**
On its own it is weak (accuracy 0.662), but it is easy to interpret. Median AI text vs human text: less variation in sentence length (4.9 vs 6.2 words), longer words (5.18 vs 4.85 letters), more commas per sentence (1.25 vs 1.00), more adjectives (8.7% vs 7.5%) and fewer adverbs (2.3% vs 3.3%). These match the common impression that ChatGPT writes evenly and formally. The website shows these features as supporting evidence, not as the decision.

**13. What is POS tagging, and how did you use it?**
Part-of-speech tagging labels each word with its grammatical role (noun, verb, adjective…). We used NLTK's averaged-perceptron tagger with Penn Treebank tags (NN, VB, JJ, RB, PRP) and turned the tags into ratios: the share of nouns, verbs, adjectives, adverbs and pronouns in a text. These are 5 of our 13 stylometric features. Loading the tagger once instead of per call made it about 10× faster.

**14. What is the difference between word embeddings and contextual embeddings?**
A static word embedding (Word2Vec, GloVe, or BERT's own input embedding layer) gives each word one fixed vector, the same in every sentence. A contextual embedding (BERT's output after its Transformer layers) changes with the surrounding words, so "bank" in "river bank" and in "bank loan" get different vectors. TF-IDF has neither: each word is an independent column.

**15. How does BERT work? What are Transformers and self-attention?**
BERT is a 12-layer Transformer encoder pre-trained on billions of words to predict masked words. In each layer, **self-attention** lets every token compute a weighted mix of all other tokens' vectors (queries, keys and values; 12 heads look for different relationships). After 12 layers, the `[CLS]` token's vector summarises the text. We added one classification layer and fine-tuned the whole network for 4 epochs (lr 2e-5, batch 16, warm-up 10%, weight decay 0.01). That is **transfer learning**: we reuse BERT's knowledge of English.

**16. What is WordPiece tokenization, and why max length 320?**
WordPiece splits rare words into frequent sub-word pieces from a ~30,000-piece vocabulary: "Unbelievably" → `un ##bel ##ie ##va ##bly` (`##` means "continues the previous piece"), so no word is ever unknown. On our data the median text is 144 tokens and the 95th percentile 355; 11.6% of texts exceed 320. 320 keeps most texts whole while staying fast on a free T4 (attention cost grows with the square of the length). In training, longer texts were truncated. In the website we split them into ≤ 320-token chunks at sentence boundaries and average the chunk scores.

**17. Why does BERT beat TF-IDF?**
BERT scored accuracy 0.945, F1 0.946, ROC-AUC 0.993, against 0.890, 0.894 and 0.967 for the best single baseline. The largest gain is on short, casual Q&A answers (0.904 vs 0.802), where individual words are not distinctive but phrasing is. TF-IDF sees only which words and word pairs appear. BERT reads order and context, recognises smooth generic phrasing and sentence structure, and brings pre-trained knowledge of language. BERT also misses far fewer AI texts: 14 false negatives versus 40 for TF-IDF + LR.

**18. Which evaluation metric matters most here?**
Accuracy alone hides the type of mistake. In an education setting a **false positive** (a human flagged as AI) is the costly error, so precision and the false-positive rate on human texts matter most. BERT's confusion matrix: 60 false positives, 14 false negatives; its precision (0.915) is lower than its recall (0.979). That is exactly why we added the 5% rule and the Uncertain zone. ROC-AUC (0.993) shows how well BERT *ranks* texts regardless of threshold.

**19. What is Gradient × Input, and can it be trusted?**
For each token, we take its input embedding vector, compute the gradient of (AI score − human score) with respect to that vector, and multiply the two element by element, then sum. The result is a first-order estimate of how much that token contributed: positive means it pushed towards AI. We tested it with a deletion test on 147 test texts. Deleting the top-5 highlighted words reduced BERT's evidence on average (−0.29 log-odds, while 5 random words *increased* it by 0.13), but beat random deletion in only **56%** of individual texts. For the linear baseline, coefficient × TF-IDF is exact and wins 100% of the time. So BERT's highlights are a useful rough guide, not a precise account. The website says so under the highlights.

**20. Why is "attention not explanation"?**
Jain & Wallace (2019) showed that very different attention patterns can produce the same prediction. Attention weights show where the model *looks*, not what *causes* its decision (Wiegreffe & Pinter, 2019 add nuance). In our BERT, 32.5% of the `[CLS]` token's last-layer attention goes to punctuation, which looks more like "parking" attention (Clark et al., 2019) than evidence. We therefore show attention only as a look inside the Transformer and use gradient-based saliency for explanations.

**21. Why three verdicts and the 5% false-positive rule?**
A probability near the boundary should not be presented as an accusation. "Likely AI" is reserved for scores that at most 5% of human validation texts reach, which caps the chance of wrongly accusing a human. "Likely Human" needs the model to at least lean human (p ≤ 0.5). Everything in between is honestly labelled "Uncertain". On the unseen test set, 6.2% of human texts got "Likely AI" with BERT (4.8% with the baseline). A threshold set on one sample lands slightly differently on another, so 5% is a target, not a guarantee. 95.2% of AI texts were still labelled "Likely AI".

**22. What is temperature scaling, and why did you need it?**
Fine-tuned neural networks are over-confident: BERT gave probabilities like 0.999 even for texts it got wrong. Temperature scaling divides the model's log-odds by one number T, fitted on validation data. For BERT T = 2.45, which halved the validation log-loss (0.264 → 0.146). It does not change accuracy or ranking, only how honest the displayed percentages are. Without it, the "Likely AI" threshold would have been 0.995 and the Uncertain zone would have disappeared.

**23. Why does the website not combine BERT with the SVM?**
We tuned the mixing weight on validation ROC-AUC. For BERT the best weight was 100% BERT, so the SVM added nothing. For the baseline the ensemble helped (accuracy 0.890 → 0.899, ROC-AUC 0.967 → 0.971), so the baseline fallback uses it. We keep an ensemble only when the data shows it helps.

**24. How robust is the detector to editing or paraphrasing?**
We edited the AI test texts: about 15% synonym swaps, shuffled sentences, fillers removed. Accuracy did not drop: BERT 0.945 → 0.948, TF-IDF LR 0.890 → 0.894. But this is optimistic. The training set contained edited copies made the same way, and WordNet swaps create odd phrasing ("advanced nursing recitation") that the models learned to associate with AI. Real paraphrasing tools or careful human rewriting were not tested and would very likely lower accuracy.

**25. What are the main limitations and ethical risks?**
- **Generators:** only ChatGPT (GPT-3.5-era) text; newer models (such as GPT-4 or Gemini) are untested.
- **Language:** English only.
- **Short texts:** most errors are short, casual answers (median misclassified length 54 words), so we warn under 50 words.
- **Formal writers:** formal human writing and possibly non-native writers are flagged more often. 17 of BERT's 60 false positives are human essays.
- **Explanations:** the word highlights are approximate (56% deletion-test win rate).
- **Training data:** BERT v1 was trained before the markdown cleanup (same split), so its test score may be slightly flattering. Re-scored on the cleaned test texts it still reaches 0.944 accuracy.
- **Use:** the tool outputs a probability, not proof, and must never be the only basis for an academic decision.
