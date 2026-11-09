# A short primer on miniature GPTs

A language model is a function that assigns a probability to the next token in a sequence. Training adjusts the function so that the tokens that actually appear in the data become more likely. Generation samples from those probabilities one token at a time.

## Tokens

Text is split into tokens before it reaches the model. A tokenizer might map `the` to one token, ` running` to another, and a rare name to three pieces. Byte-level tokenizers start from the 256 possible bytes. They learn merges that combine frequent neighbors into longer units. Because every byte has an id, any text can be encoded without an unknown token.

The vocabulary size is the number of distinct token ids. A small model might use one thousand ids; a large one uses tens of thousands. A larger vocabulary shortens sequences but makes the output layer wider.

## Embeddings

Each token id indexes a row in an embedding table, producing a vector. Positional embeddings add information about where the token sits in the window. The model has no built-in sense of order without them, because attention itself is permutation invariant.

## Attention

Attention lets each position build a weighted average of earlier positions. Queries ask what a position is looking for, keys describe what each position offers, and values carry the content. The weights come from the similarity between a query and each key.

A causal mask prevents a position from attending to the future. This is what makes the model usable for next-token prediction: during training, every position predicts the token that follows it, using only the tokens before it.

Multi-head attention runs several attention operations in parallel. Each head can specialize. One head may track nearby words, another may track repeated names, another may track punctuation.

## Blocks

A transformer block applies layer normalization, attention, another normalization, and a small feed-forward network. Residual connections add the input of each sublayer to its output. Stacking blocks increases the depth and the model's ability to compose features.

The feed-forward network is usually four times wider than the model dimension. It mixes information within each position after attention has mixed information across positions.

## The language-model head

The final hidden vector at each position is projected to the vocabulary size. Those logits become probabilities with a softmax. Training uses cross-entropy between the predicted distribution and the actual next token.

Many small models tie the output projection to the token embedding matrix. Tying saves parameters and often helps small models share structure.

## Training

Training samples a batch of windows from a token stream. For each window, the input is the tokens and the target is the same window shifted by one. The forward pass computes logits and loss. Backpropagation computes gradients. An optimizer such as AdamW updates the parameters. Warmup gradually increases the learning rate, and a cosine schedule decays it toward the end.

Gradient clipping keeps a single bad batch from taking a huge step. Dropout randomly hides activations during training and is disabled at inference. Weight decay discourages large weights.

A validation split measures the model on tokens it did not train on. When validation loss stops improving, more training is usually not helping.

## Sampling

Given a prompt, the model produces a distribution over the next token. Greedy decoding picks the most likely token. Temperature divides the logits before the softmax: lower values sharpen the distribution, higher values flatten it. Top-k keeps only the k most likely tokens. Repetition penalties discourage tokens that appeared recently.

Sampling matters. A model can be accurate and still produce dull text with greedy decoding. A temperature near zero is useful for tests; a temperature between `0.7` and `1.0` is more natural for prose.

## Context windows

The context window is the maximum sequence length the model was trained to handle. A model trained with two hundred fifty-six positions has an embedding row for each position. Feeding it a longer sequence would require positions it never learned.

ChainForge treats the context window as a hard budget. It reserves space for the answer, then for the reasoning draft, then fits as much history and question as possible. Everything that does not fit is trimmed and reported.

## Fine-tuning

Fine-tuning starts from a trained model and continues training on a smaller, targeted dataset. It is cheaper than training from scratch and usually produces better results for a narrow task.

The tokenizer must stay the same during fine-tuning, otherwise the embedding rows would no longer correspond to the same tokens. ChainForge locks the parent tokenizer and architecture and records the parent in the new archive.

A lower learning rate protects the pretrained weights. A few hundred steps on a few thousand examples can change style noticeably.

## What small models can and cannot do

A one-million-parameter model trained on a small corpus can learn vocabulary, formatting, punctuation, and local phrasing. It can imitate a question-and-answer template. It can memorize passages from the corpus.

It cannot reliably perform multi-step arithmetic, track long-range facts, or reason about the world. Those abilities emerge from much larger models trained on much larger and more diverse data. Treat a miniature model as a craft project and a teaching tool, not as a substitute for a production assistant.

## A practical checklist

1. Collect clean text in the style you want.
2. Remove duplicates and boilerplate.
3. Start with a smoke run to confirm the pipeline works.
4. Increase steps while validation loss improves.
5. Test the model with short, specific prompts.
6. Save the archive and write down what data produced it.
7. Fine-tune from the archive when you add better examples.
