# The ChainForge manual

ChainForge is a small workshop for training a custom miniature language model. You bring documents, the workshop turns them into tokens, and a transformer learns to predict the next token. The result is stored in a `.cgpt` archive that you can chat with, download, import, and fine-tune.

The interface has three tabs.

The Train tab accepts a dataset of documents. You can drop text files, Markdown notes, Word documents, PDFs with a text layer, CSV files, JSON, source code, or any readable text file. ChainForge extracts the text, removes duplicates, and writes one long token stream. Training then samples random windows from that stream and updates the model weights.

The Chat tab loads a model from the vault. It writes a short reasoning draft inside the thinking block, then writes an answer. Both halves of the reply are limited by the token budget. A context meter shows how many tokens the prompt used, how many the reasoning used, how many the answer used, and how large the context window is.

The Save tab lists every model in the vault. You can download an archive, import a new one, or continue training from an existing one. Bundled models are read-only.

## Training modes

A major run starts from random weights and learns a new tokenizer from your documents. This is the right choice for a new domain or a fresh base model.

A fine-tune run starts from an existing archive. It reuses the parent tokenizer and the parent architecture, so the vocabulary and context window do not change. Fine-tuning is the cheap way to teach an existing model a new style, a new product, or a new set of facts.

## Choosing steps

One step processes a batch of token windows and performs one optimizer update. More steps let the model see more of the data, but too many steps on a small corpus causes memorization: training loss keeps falling while validation loss starts to rise.

A smoke run uses twenty steps. A standard run uses three hundred steps. A deep run uses one thousand two hundred steps or more. The right number depends on the corpus size. A good habit is to start small, look at the loss curve, and increase the steps only while validation loss is still improving.

## Choosing a batch size

The batch size is the number of independent windows used for one update. Larger batches give smoother gradients but require more memory and more time per step. On a small CPU runner, a batch of eight to sixteen windows is usually comfortable.

## Learning rate

The learning rate controls how far each update moves the weights. Major runs start around `4e-4`. Fine-tunes start lower, around `1e-4`, because the model already knows useful things and large updates can damage them. If the loss jumps around or becomes not a number, lower the rate.

## The loss

Cross-entropy loss measures surprise. A loss of `ln(vocab_size)` is what a model that guesses uniformly would score. Lower is better. Training loss is measured on the data the model is learning from. Validation loss is measured on held-out tokens. The gap between them tells you whether the model is generalizing or memorizing.

## Tokens and the token budget

A token is a chunk of text. Common words become one token, rare words split into pieces, and every byte can be represented. The tokenizer is learned from your corpus with byte-pair encoding.

The context window is the maximum number of tokens the model can look at at once. The prompt, the reasoning draft, and the answer must all fit inside it. ChainForge reserves room for generation first, then keeps as much history and question text as possible. When space runs out, the oldest history is removed first, then the beginning of the current question. The reply metadata reports how many tokens were trimmed.

The reasoning budget limits the scratchpad. The answer budget limits the final response. If a phase hits its budget, generation stops and the stop reason says `budget`. If the model emits its end marker, the stop reason is `endthink` or `end`.

## The reasoning draft

The model is trained on a protocol with a visible thinking block. The draft is useful because it gives the model room to lay out a plan before committing to an answer. It is not a window into hidden computation. A small model can write confident notes that are incomplete or wrong. Treat the draft as a working note, not as proof.

## The model archive

A `.cgpt` archive is a zip file with four parts: a manifest, a tokenizer, safetensors weights, and training metadata. It contains no pickle, so importing one cannot execute code. The manifest records the architecture, parameter count, checksum, creation date, and lineage. Imported archives are checked for unexpected members, path traversal, oversized entries, checksum mismatches, and architecture limits.

Fine-tuning records the parent model name and identifier. That lineage makes it possible to see how a model was produced and to continue the chain later.

## Data preparation

Good data is consistent, clean, and relevant. Remove navigation menus, boilerplate, and duplicated pages. Prefer documents that demonstrate the style you want the model to produce. If you want question-and-answer behavior, include question-and-answer examples in the training corpus.

ChainForge accepts many formats but does not perform OCR. A scanned PDF has no text layer and will be rejected. Convert it to text with an OCR tool first. Legacy `.doc` files should be saved as `.docx` or plain text.

## Running offline

Everything runs locally. Documents stay in the data directory, weights stay on disk, and chat requests go to the local server. There are no external API calls. If you set an access token, browsers must send it with API requests; the token is stored in local storage and never written into a model archive.

## Common problems

If training fails with "no usable text was extracted", every uploaded file was rejected or empty. Check the reasons in the extraction report.

If the dataset is too small, the trainer refuses to start. A context window of two hundred fifty-six tokens needs at least a few times that many tokens to sample from.

If the answer stops immediately, the model may have produced its end marker at once. This is common for a model trained for only a few steps. More steps, more data, or a larger answer budget can help.

If the model repeats itself, the corpus may be too small or the model may have memorized a phrase. Lower the temperature, train longer on more varied text, or fine-tune on better examples.

If validation loss rises while training loss falls, the model is memorizing. Stop earlier, add more documents, or reduce the number of steps.
