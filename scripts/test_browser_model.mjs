import * as tf from '@tensorflow/tfjs';
import {
  S,
  buildTokenizer,
  createModel,
  decode,
  encode,
  exportModelData,
  generateText,
  importModelData,
  prepareTrainingText,
  trainModel,
} from '../pages/engine.js';

await tf.setBackend('cpu');
await tf.ready();
console.log('backend', tf.getBackend());
console.log('gelu', typeof tf.gelu);
console.log('softmaxCrossEntropy', typeof tf.losses.softmaxCrossEntropy);

const source = [
  'ChainForge is a small model that learns in the browser.',
  'User: What is ChainForge?',
  'Assistant: ChainForge is a tiny language model that trains on your words locally.',
  'User: Does it need a server?',
  'Assistant: No. The training loop runs in JavaScript in this page.',
].join('\n\n').repeat(8);
const prepared = prepareTrainingText([source]);
const tokenizer = buildTokenizer([prepared], 96);
const ids = encode(prepared, tokenizer);
console.log('chars', prepared.length, 'tokens', ids.length, 'vocab', tokenizer.vocabSize);

const model = createModel(tokenizer, { ctx: 32, dModel: 32, heads: 4, layers: 1 });
console.log('params', model.parameterCount);

const startTensors = tf.memory().numTensors;
const history = await trainModel(model, ids, {
  steps: 18,
  batch: 3,
  learningRate: 0.01,
  time: 32,
  onStep: ({ step, loss }) => {
    if (step === 1 || step % 6 === 0) console.log('step', step, 'loss', loss.toFixed(4), 'tensors', tf.memory().numTensors);
  },
});
console.log('start loss', history[0].toFixed(4), 'end loss', history.at(-1).toFixed(4));
console.log('tensor growth', tf.memory().numTensors - startTensors);
if (!Number.isFinite(history.at(-1))) throw new Error('non-finite loss');
if (history.length !== 18) throw new Error('training stopped early');

const prompt = encode(`${S.bos}${S.user}What is ChainForge?${S.assistant}`, tokenizer);
const generated = await generateText(model, tokenizer, prompt, {
  maxTokens: 20,
  temperature: 0.7,
  topK: 20,
  stopIds: [tokenizer.stoi.get(S.end)],
});
const text = decode(generated, tokenizer);
console.log('generated:', JSON.stringify(text));
if (!text.length) throw new Error('generation produced no visible text');

const saved = await exportModelData(model, { steps: 18 });
const json = JSON.stringify(saved);
console.log('model json bytes', json.length);
const restored = await importModelData(JSON.parse(json));
const prompt2 = encode(`${S.bos}${S.user}Does it need a server?${S.assistant}`, tokenizer);
const generated2 = await generateText(restored.model, restored.model.tokenizer, prompt2, {
  maxTokens: 16,
  temperature: 0,
  stopIds: [restored.model.tokenizer.stoi.get(S.end)],
});
console.log('restored generated:', JSON.stringify(decode(generated2, restored.model.tokenizer)));

model.dispose();
restored.model.dispose();
console.log('OK');
