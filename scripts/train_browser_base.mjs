import fs from 'node:fs';
import path from 'node:path';
import {
  buildTokenizer,
  createModel,
  encode,
  exportModelData,
  importModelData,
  prepareTrainingText,
  trainModel,
} from '../pages/engine.js';

const root = process.cwd();
const corpusDir = path.join(root, 'corpus', 'base');
const out = process.env.CF_OUT || path.join(root, 'pages', 'base-model.json');
const targetSteps = Number.parseInt(process.env.CF_STEPS || '240', 10);
const batch = Number.parseInt(process.env.CF_BATCH || '4', 10);
const learningRate = Number.parseFloat(process.env.CF_LR || '0.004');
const resume = process.env.CF_RESUME === '1';
const config = {
  ctx: Number.parseInt(process.env.CF_CTX || '96', 10),
  dModel: Number.parseInt(process.env.CF_DMODEL || '48', 10),
  heads: Number.parseInt(process.env.CF_HEADS || '4', 10),
  layers: Number.parseInt(process.env.CF_LAYERS || '1', 10),
};

const files = fs.readdirSync(corpusDir).filter((name) => name.endsWith('.md')).sort();
if (!files.length) throw new Error('No seed corpus files found.');
const seedTexts = files.map((name) => fs.readFileSync(path.join(corpusDir, name), 'utf8'));
const prepared = prepareTrainingText(seedTexts);
const tokenizer = buildTokenizer([prepared], 180);
const ids = encode(prepared, tokenizer);

let model = null;
let alreadyDone = 0;
let previousLoss = null;
if (resume && fs.existsSync(out)) {
  const saved = JSON.parse(fs.readFileSync(out, 'utf8'));
  const restored = await importModelData(saved);
  model = restored.model;
  alreadyDone = Number(saved.meta?.steps) || 0;
  previousLoss = Number(saved.meta?.loss) || null;
  console.log(`resumed ${alreadyDone} steps from ${out}`);
} else {
  model = createModel(tokenizer, config);
}

const remaining = Math.max(0, targetSteps - alreadyDone);
console.log(JSON.stringify({
  files,
  chars: prepared.length,
  tokens: ids.length,
  vocab: tokenizer.vocabSize,
  params: model.parameterCount,
  config: model.config,
  targetSteps,
  remaining,
  device: 'cpu',
}, null, 2));

if (remaining > 0) {
  const started = Date.now();
  let lastCheckpoint = alreadyDone;
  await trainModel(model, ids, {
    steps: remaining,
    batch,
    learningRate,
    time: model.config.ctx,
    onStep: async ({ step, loss, history }) => {
      const completed = alreadyDone + step;
      previousLoss = loss;
      if (step === 1 || step % 10 === 0 || step === remaining) {
        const elapsed = (Date.now() - started) / 1000;
        const rate = step / Math.max(0.001, elapsed);
        console.log(`step ${completed}/${targetSteps} loss ${loss.toFixed(4)} ${rate.toFixed(2)} steps/s`);
      }
      if (step % 40 === 0 && step < remaining) {
        const checkpoint = await exportModelData(model, {
          name: 'ChainForge Seed',
          kind: 'base',
          steps: completed,
          loss,
          history: history.slice(-40),
          chars: prepared.length,
          vocab: tokenizer.vocabSize,
          params: model.parameterCount,
          seedFiles: files,
          note: 'Tiny character-level transformer trained on the bundled ChainForge corpus.',
        });
        fs.writeFileSync(out, JSON.stringify(checkpoint));
        lastCheckpoint = completed;
        console.log(`checkpoint ${completed} steps (${(fs.statSync(out).size / 1e6).toFixed(2)} MB)`);
      }
    },
  });
  previousLoss = previousLoss ?? null;
  console.log(`finished training pass (last checkpoint at step ${lastCheckpoint})`);
}

const final = await exportModelData(model, {
  name: 'ChainForge Seed',
  kind: 'base',
  steps: targetSteps,
  loss: previousLoss,
  chars: prepared.length,
  vocab: tokenizer.vocabSize,
  params: model.parameterCount,
  seedFiles: files,
  note: 'Tiny character-level transformer trained on the bundled ChainForge corpus. Runs entirely in the browser when you fine-tune it.',
});
fs.writeFileSync(out, JSON.stringify(final));
fs.writeFileSync(
  path.join(root, 'base-summary.json'),
  JSON.stringify({
    steps: targetSteps,
    loss: previousLoss,
    chars: prepared.length,
    tokens: ids.length,
    vocab: tokenizer.vocabSize,
    params: model.parameterCount,
    config: model.config,
    bytes: fs.statSync(out).size,
  }, null, 2),
);
console.log(`wrote ${out} (${(fs.statSync(out).size / 1e6).toFixed(2)} MB)`);
model.dispose();
