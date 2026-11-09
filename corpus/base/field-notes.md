# Field notes from the workshop

The first model is always a surprise. It begins by producing punctuation and spaces, then fragments of words, then sentences that sound almost right. The loss curve falls and flattens. Somewhere in the flat part, the model starts quoting the corpus back at you. That is the moment to look at validation loss instead of training loss.

A small corpus teaches vocabulary quickly and structure slowly. If you give the model a hundred pages of product documentation, it will learn the product's nouns and the shape of its headings. It will not learn to be a careful assistant. If you want a careful assistant, you need examples of careful assistance: questions, answers, corrections, refusals, and short explanations.

The tokenizer is the first place where a dataset reveals itself. If the documents are full of identifiers like `user_profile_v2`, the tokenizer will spend vocabulary on pieces of those identifiers. That is fine when the model needs to talk about them and wasteful when the corpus is mostly prose with a few code samples. Clean data is cheaper than a bigger vocabulary.

Training is a conversation between the data and the schedule. A learning rate that is too high makes the loss bounce or explode. A learning rate that is too low makes the loss fall with painful slowness. Warmup helps the first few steps, and cosine decay helps the last few. Gradient clipping is a seatbelt: it does nothing most of the time and saves the run when a batch goes badly.

Batches are not just a performance knob. A batch of one gives a noisy gradient and a jumpy curve. A larger batch averages that noise away, which can hide problems as easily as it can reveal them. On a CPU, a batch of eight to sixteen windows is a good compromise between smoothness, memory, and time.

The context window is a budget, not a suggestion. When a prompt is too long, something must go. ChainForge takes the oldest history first. This keeps the latest question intact, which is usually what matters. If the question itself is longer than the remaining room, the beginning of the question is trimmed. The reply metadata records the count, so a strange answer can be traced back to a truncated prompt.

The reasoning draft is a useful habit for the model and a useful reminder for the builder. A model that writes a short plan before answering tends to produce more coherent answers, especially when the plan names the entities and the shape of the response. But the draft is not a debugger. It is text sampled from the same distribution as the answer, and it can be wrong in the same ways.

Fine-tuning is where small models become useful. A base model trained on general text can be adapted to a narrow task with a few hundred examples. Keep the learning rate low, keep the examples consistent, and stop before the model starts repeating the examples verbatim. The archive records the parent, so a chain of adaptations is visible later.

The archive is deliberately boring. A zip, a manifest, a tokenizer, and a tensor file. Boring is good for a format that people will move between machines. The absence of pickle is not a stylistic choice; it means an imported model cannot smuggle code into the loader. The checksum catches corruption. The allowlist catches everything else.

Deployment changes the meaning of storage. On a laptop, the data directory sits on a real disk. On a free hosting tier, it may disappear when the container restarts. Download finished models. Keep the source documents. Treat the hosted vault as a workbench, not a library.

The best test of a model is a question you actually care about. Not "hello", not "write a poem", but the thing you would ask a colleague. If the answer is wrong, the corpus is the first suspect. If the answer is formatted badly, the examples are the first suspect. If the answer is empty, the budget or the stop marker is the first suspect.

A model is a mirror with a delay. It reflects the words you fed it, rearranged by probabilities. When the reflection looks wrong, look at what you fed it. Most of the time the mirror is doing exactly what it was taught.

The workshop is quiet in the morning. The fans spin up, the loss curve starts, and the log fills one line at a time. Somewhere in the middle of the run, the model stops being a random function and starts being a bad writer. That is progress. A bad writer with your vocabulary is the first step toward a useful one.

A good checklist for every run:

- Did the extraction report accept the files I expected?
- Is the dataset large enough for the context window?
- Is the learning rate in the right range for the mode?
- Does validation loss improve for a while?
- Does the model answer a short prompt with the expected format?
- Have I downloaded the archive?

If any answer is no, fix that before adding more steps. More steps cannot repair a dataset that was never extracted, and they cannot make a learning rate that is ten times too high behave.

The last lesson is about patience. A miniature model learns in minutes, but learning what to feed it takes longer. Every good run in the workshop started with someone reading their own documents closely, removing the boilerplate, and writing a few examples of the behavior they wanted. The model did the rest, badly at first and then a little better.
