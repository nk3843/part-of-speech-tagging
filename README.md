# Part-of-speech tagging for source-code identifiers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nk3843/part-of-speech-tagging/blob/main/POS.ipynb)

Tags each word in a program identifier name, such as a method name split into words, with its grammatical
role, and compares recurrent neural network architectures for the task in Keras.

```
Input:  ares  expand  name  for  response
Tags:   PRE   V       N     P    N
```

Tags include `N` noun, `NM` noun modifier, `V` verb, `VM` verb modifier, `P` preposition, and `PRE` for a
leading prefix or abbreviation (10 tags in total).

## Approach

- **Data:** 1,335 tagged identifier names, split 72% train / 13% validation / 15% test.
- **Encoding:** words and tags tokenized with Keras, padded to 100 positions, tags one-hot encoded.
- **Embeddings:** 300-dimensional Google News word2vec vectors (gensim), compared with randomly initialized ones.
- **Models:** an embedding layer, a recurrent layer of 64 units, and a per-position softmax (`TimeDistributed`):

| Model | Embedding | Test accuracy* |
|---|---|---|
| SimpleRNN | random, frozen | (compared on validation only) |
| SimpleRNN | random, trained | (compared on validation only) |
| SimpleRNN | word2vec, fine-tuned | 98.1% |
| LSTM | word2vec, fine-tuned | 97.3% |
| GRU | word2vec, fine-tuned | 98.2% |
| Bidirectional LSTM | word2vec, fine-tuned | 98.0% |

## \*A note on these numbers

The accuracy is measured over all 100 positions, including padding. Identifiers are only 3-5 words long, so most
positions are padding, and a model predicting "padding" everywhere already scores about 97% (the validation
accuracy of every model after its first epoch). The numbers above therefore overstate how well real words are
tagged, and the differences between models are too small to rank them.

A fair comparison would:

1. **Mask padding** (`Embedding(mask_zero=True)`) or compute accuracy only on real tokens.
2. **Report per-tag precision, recall, and F1**, since rarer tags like `VM` and `P` matter most.
3. **Add a baseline**, such as the most frequent tag for each word.

## Run it

Open the notebook in Colab (badge above). It expects two files in Google Drive:

- `data.txt`: tab-separated identifier and tags, one per line (not included in this repo)
- `GoogleNews-vectors-negative300.bin`: the pretrained word2vec vectors

Built with Keras/TensorFlow, gensim, scikit-learn, pandas, and matplotlib (2021).
