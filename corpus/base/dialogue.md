<|user|>
What is ChainForge?
<|assistant|>
<|think|>
The user wants a definition. I should say it is a local trainer for small GPT models, describe the three tabs, and mention the portable archive.
<|endthink|>
ChainForge is a local workshop for training a miniature GPT on your own documents. You upload text, Markdown, Word files, PDFs, or code, choose a number of steps, and the trainer learns a small transformer. The web interface has Train, Chat, and Save tabs, and every finished model is stored as a portable .cgpt archive.
<|end|>

<|user|>
How do I train on my own notes?
<|assistant|>
<|think|>
Give the concrete path through the UI and the CLI. Keep it short and mention extraction.
<|endthink|>
Open the Train tab, drop your notes into the document area, and choose Major if you are starting fresh. Set the step count, press Start training, and watch the loss curve. The trainer extracts the text, removes duplicates, learns a tokenizer, and packs everything into one token stream. When the run finishes, the model appears in the vault and can be selected in Chat.
<|end|>

<|user|>
What is the difference between a major run and a fine-tune?
<|assistant|>
<|think|>
Major starts from random weights and a new tokenizer. Fine-tune continues an archive and keeps its tokenizer and architecture. I should mention the lower learning rate.
<|endthink|>
A major run starts from random weights and learns a new tokenizer from the new documents. A fine-tune starts from an existing .cgpt archive, reuses its tokenizer and context window, and usually uses a lower learning rate so the existing knowledge is not destroyed. Choose major for a new domain and fine-tune to adapt a model you already like.
<|end|>

<|user|>
What does the training loss mean?
<|assistant|>
<|think|>
Explain cross-entropy as surprise and mention validation loss and memorization.
<|endthink|>
The training loss is cross-entropy: the average surprise of the model when it predicts the next token in the training data. Lower is better. Validation loss is measured on held-out tokens. If training loss keeps falling while validation loss rises, the model is memorizing rather than generalizing, and you probably want more data or fewer steps.
<|end|>

<|user|>
Why does my model repeat the same phrase?
<|assistant|>
<|think|>
Likely causes: small corpus, memorization, high temperature, repetition penalty, or an undertrained model. Offer practical fixes.
<|endthink|>
Repetition usually comes from a small or repetitive corpus, or from a model that has memorized a phrase. Try adding more varied documents, lowering the temperature, smoothing the sampling with top-k, and checking whether validation loss is rising. A model trained for only a few dozen steps often produces broken loops because it has not learned enough structure yet.
<|end|>

<|user|>
What is a token budget?
<|assistant|>
<|think|>
The context window is finite. The budget splits it among prompt, reasoning, and answer, and trims the oldest text.
<|endthink|>
A token budget is a hard limit on how many tokens may be used. The context window is shared by the prompt, the reasoning draft, and the answer. ChainForge reserves room for generation first, then fits as much history and question as possible. When space runs out, it removes the oldest history first and reports how many tokens were trimmed.
<|end|>

<|user|>
Is the reasoning draft real chain-of-thought?
<|assistant|>
<|think|>
Be honest. It is a visible scratchpad that conditions the answer, not a faithful trace of internal computation.
<|endthink|>
No. The reasoning draft is a scratchpad the model was trained to write before answering, and its answer is conditioned on that text. It is visible and useful, but it is not a verified record of hidden computation. Small models can write confident notes that are incomplete or wrong, so treat the draft as a working note rather than proof.
<|end|>

<|user|>
Which files can I upload?
<|assistant|>
<|think|>
List the main supported formats and the important exclusions.
<|endthink|>
You can upload plain text, Markdown, reStructuredText, CSV, JSON, YAML, HTML, XML, source code, Word .docx files, notebooks, and PDFs that have a text layer. Images, audio, video, archives, and legacy .doc files are not supported. Scanned PDFs need OCR before they can be used, because ChainForge does not perform OCR.
<|end|>

<|user|>
What is inside a .cgpt file?
<|assistant|>
<|think|>
Describe the four members and the no-pickle safety property.
<|endthink|>
A .cgpt file is a zip archive with a manifest, a tokenizer, safetensors weights, and training metadata. The manifest records the architecture, parameter count, checksum, creation date, and lineage. Imported archives are validated for unexpected members, path traversal, oversized entries, checksum mismatches, and architecture limits. No pickle is used, so importing a model cannot execute code.
<|end|>

<|user|>
How many steps should I use?
<|assistant|>
<|think|>
There is no universal answer. Suggest starting small and watching validation loss.
<|endthink|>
Start small and let the validation curve guide you. A smoke run with twenty steps checks that the pipeline works. A standard run with three hundred steps is a reasonable first model. Increase toward one thousand or more when you have a larger corpus and validation loss is still improving. More steps on a tiny corpus mostly buy memorization.
<|end|>

<|user|>
Does ChainForge send my documents to a server?
<|assistant|>
<|think|>
The app is local; documents stay on the machine running the server. No external API calls.
<|endthink|>
No. Training and inference happen in the local server process. Documents are written to the local data directory, weights stay on disk, and chat requests go to the local API. There are no external API calls or telemetry. If you deploy the app to a host, your documents are stored on that host instead.
<|end|>

<|user|>
Can a tiny model answer hard questions?
<|assistant|>
<|think|>
Set expectations honestly. It can learn style and format, not reliable general reasoning.
<|endthink|>
Usually not. A model with one to twelve million parameters trained on a small corpus can learn your vocabulary, formatting, and local style, and it can imitate a question-and-answer template. It will not reliably do multi-step arithmetic or reason about the world. For stronger answers you need more clean data, longer training, a larger architecture, and realistically a much larger model.
<|end|>

<|user|>
What happens when training is cancelled?
<|assistant|>
<|think|>
Cancellation is cooperative. The trainer checks between steps, stops, and no archive is exported.
<|endthink|>
Cancellation is cooperative. ChainForge checks for a cancel request between optimizer steps, stops the run, and reports the job as cancelled. No archive is exported from a cancelled run, so the vault is unchanged. A step that is already in progress is allowed to finish before the run stops.
<|end|>

<|user|>
What is the base model for?
<|assistant|>
<|think|>
It is a starter model trained on the seed corpus. Use it to try chat and as a parent for fine-tuning.
<|endthink|>
The base model is a starter trained on the bundled seed corpus. It lets you try the Chat tab and the reasoning protocol before you train on your own data, and it can serve as a parent for fine-tuning. Because the seed corpus is small, the base model mostly demonstrates the pipeline; train a major run on your documents for a model that knows your domain.
<|end|>

<|user|>
How do I keep a model I trained?
<|assistant|>
<|think|>
Download the archive from Save. Hosted storage may be ephemeral. Re-import to continue.
<|endthink|>
Open the Save tab, find the model in the vault, and download the .cgpt file. Keep that file somewhere durable. Hosted deployments may use ephemeral disk, so downloading is the reliable way to preserve a model. To continue later, import the archive and start a fine-tune run from it.
<|end|>
