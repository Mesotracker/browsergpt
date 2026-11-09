import * as tf from '@tensorflow/tfjs';

/**
 * ChainForge Browser Edition
 * A tiny character-level transformer that is defined, trained, and sampled
 * entirely in JavaScript. The same module runs in the browser and in Node so
 * the committed base model and the in-page trainer share one implementation.
 */

export const S = Object.freeze({
  pad: '\uE000',
  unk: '\uE001',
  bos: '\uE002',
  user: '\uE003',
  assistant: '\uE004',
  end: '\uE005',
  think: '\uE006',
  endthink: '\uE007',
});

export const SPECIAL_ORDER = [
  S.pad,
  S.unk,
  S.bos,
  S.user,
  S.assistant,
  S.end,
  S.think,
  S.endthink,
];

const SPECIAL_SET = new Set(SPECIAL_ORDER);

export const SPECIAL_LABELS = Object.freeze({
  [S.pad]: '<pad>',
  [S.unk]: '<unk>',
  [S.bos]: '<bos>',
  [S.user]: '<|user|>',
  [S.assistant]: '<|assistant|>',
  [S.end]: '<|end|>',
  [S.think]: '<|think|>',
  [S.endthink]: '<|endthink|>',
});

export const DEFAULT_CONFIG = Object.freeze({
  ctx: 64,
  dModel: 48,
  heads: 4,
  layers: 1,
  mlp: 4,
});

export function isSpecial(ch) {
  return SPECIAL_SET.has(ch);
}

export function buildTokenizer(texts, maxChars = 240) {
  const counts = new Map();
  for (const text of texts) {
    for (const ch of String(text || '')) {
      if (SPECIAL_SET.has(ch)) continue;
      counts.set(ch, (counts.get(ch) || 0) + 1);
    }
  }
  let chars = [...counts.keys()];
  chars.sort((a, b) => {
    const diff = counts.get(b) - counts.get(a);
    if (diff) return diff;
    return a.codePointAt(0) - b.codePointAt(0);
  });
  if (chars.length > maxChars) chars = chars.slice(0, maxChars);
  const itos = [...SPECIAL_ORDER, ...chars];
  return {
    itos,
    stoi: new Map(itos.map((ch, i) => [ch, i])),
    vocabSize: itos.length,
  };
}

export function tokenizerFromData(data) {
  if (!data || !Array.isArray(data.itos) || data.itos.length < SPECIAL_ORDER.length) {
    throw new Error('Invalid tokenizer data.');
  }
  const itos = data.itos.map(String);
  return {
    itos,
    stoi: new Map(itos.map((ch, i) => [ch, i])),
    vocabSize: itos.length,
  };
}

export function tokenizerToData(tokenizer) {
  return { itos: [...tokenizer.itos] };
}

export function encode(text, tokenizer) {
  const unk = tokenizer.stoi.get(S.unk);
  const ids = [];
  for (const ch of String(text || '')) {
    const id = tokenizer.stoi.get(ch);
    ids.push(id === undefined ? unk : id);
  }
  return ids;
}

export function decode(ids, tokenizer, { keepSpecial = false } = {}) {
  let out = '';
  for (const id of ids) {
    const ch = tokenizer.itos[id];
    if (ch === undefined) continue;
    if (SPECIAL_SET.has(ch)) {
      if (keepSpecial) out += SPECIAL_LABELS[ch] || '';
      continue;
    }
    out += ch;
  }
  return out;
}

export function convertChatMarkers(text) {
  return String(text || '')
    .replace(/<\|user\|>/g, `${S.bos}${S.user}`)
    .replace(/<\|assistant\|>/g, S.assistant)
    .replace(/<\|think\|>/g, S.think)
    .replace(/<\|endthink\|>/g, S.endthink)
    .replace(/<\|end\|>/g, S.end);
}

export function prepareTrainingText(seedTexts = [], uploadTexts = []) {
  const seed = seedTexts
    .map((text) => convertChatMarkers(text))
    .join('\n\n');
  const docs = uploadTexts
    .map((text) => convertChatMarkers(text))
    .join('\n\n');
  let text = `${seed}${docs ? `\n\n${docs}` : ''}`.replace(/\r\n?/g, '\n');
  if (!text.startsWith(S.bos)) text = S.bos + text;
  return text;
}

function assertTensor(tensor, label) {
  if (!tensor) throw new Error(`Forward pass failed at ${label}.`);
  return tensor;
}

export class TinyGPT {
  constructor(tokenizer, config = {}) {
    if (!tokenizer || !Array.isArray(tokenizer.itos)) {
      throw new Error('A tokenizer is required.');
    }
    this.tokenizer = tokenizer;
    this.config = {
      ...DEFAULT_CONFIG,
      ...config,
      vocabSize: tokenizer.itos.length,
    };
    const { ctx, dModel, heads, layers, vocabSize } = this.config;
    if (dModel % heads !== 0) throw new Error('dModel must be divisible by heads.');
    this.config.dHead = dModel / heads;
    this.vars = [];
    this.weights = {};

    const add = (name, initial, trainable = true) => {
      // Keep logical names in our own map. tfjs variable names are global, so
      // passing the logical name would break loading a second model in a page.
      const variable = tf.variable(initial, trainable);
      this.weights[name] = variable;
      this.vars.push(variable);
      return variable;
    };
    const rand = (shape, std = 0.02) => tf.randomNormal(shape, 0, std);
    const zeros = (shape) => tf.zeros(shape);
    const ones = (shape) => tf.ones(shape);

    add('tok', rand([vocabSize, dModel]));
    add('pos', rand([ctx, dModel], 0.01));

    for (let i = 0; i < layers; i++) {
      const p = `l${i}`;
      add(`${p}.wq`, rand([dModel, dModel]));
      add(`${p}.wk`, rand([dModel, dModel]));
      add(`${p}.wv`, rand([dModel, dModel]));
      add(`${p}.wo`, rand([dModel, dModel]));
      add(`${p}.bq`, zeros([dModel]));
      add(`${p}.bk`, zeros([dModel]));
      add(`${p}.bv`, zeros([dModel]));
      add(`${p}.bo`, zeros([dModel]));
      add(`${p}.g1`, ones([dModel]));
      add(`${p}.b1ln`, zeros([dModel]));
      add(`${p}.w1`, rand([dModel, dModel * this.config.mlp]));
      add(`${p}.b1`, zeros([dModel * this.config.mlp]));
      add(`${p}.w2`, rand([dModel * this.config.mlp, dModel]));
      add(`${p}.b2`, zeros([dModel]));
      add(`${p}.g2`, ones([dModel]));
      add(`${p}.b2ln`, zeros([dModel]));
    }

    add('gfin', ones([dModel]));
    add('bfin', zeros([dModel]));
    // An explicit output projection avoids a batched-matmul gradient bug in
    // tfjs when the token embedding is reused as a transposed output matrix.
    add('wout', rand([dModel, vocabSize]));
    add('bout', zeros([vocabSize]));

    const mask = new Float32Array(ctx * ctx);
    for (let row = 0; row < ctx; row++) {
      for (let col = 0; col < ctx; col++) {
        if (col > row) mask[row * ctx + col] = -1e9;
      }
    }
    this.mask = tf.tensor4d(mask, [1, 1, ctx, ctx]);
  }

  get parameterCount() {
    return this.vars.reduce((total, v) => total + v.size, 0);
  }

  layerNorm(x, gamma, beta) {
    const mean = tf.mean(x, -1, true);
    const centered = tf.sub(x, mean);
    const variance = tf.mean(tf.square(centered), -1, true);
    const norm = tf.mul(centered, tf.rsqrt(tf.add(variance, 1e-5)));
    return tf.add(tf.mul(norm, gamma), beta);
  }

  linear(x, weight, bias = null) {
    // tfjs's BatchMatMul gradient does not reduce a broadcast batch dimension
    // for a rank-3 x rank-2 product. Flattening batch*time keeps both operands
    // 2-D, so the gradient for every projection has the correct weight shape.
    const inFeatures = x.shape[x.shape.length - 1];
    const flat = tf.reshape(x, [-1, inFeatures]);
    let out = tf.matMul(flat, weight);
    if (bias) out = tf.add(out, bias);
    return tf.reshape(out, [...x.shape.slice(0, -1), weight.shape[1]]);
  }

  gelu(x) {
    // TensorFlow.js does not expose a GELU op in every build. This is the
    // standard tanh approximation and remains differentiable for training.
    const inner = tf.mul(
      tf.add(x, tf.mul(tf.pow(x, 3), 0.044715)),
      Math.sqrt(2 / Math.PI),
    );
    return tf.mul(tf.mul(x, 0.5), tf.add(1, tf.tanh(inner)));
  }

  forward(ids) {
    const [batch, time] = ids.shape;
    const { dModel, heads, layers } = this.config;
    const dHead = dModel / heads;
    if (time > this.config.ctx) throw new Error('Input exceeds the context window.');

    let x = tf.gather(this.weights.tok, ids);
    const positions = this.weights.pos.slice([0, 0], [time, dModel]);
    x = tf.add(x, tf.expandDims(positions, 0));

    for (let i = 0; i < layers; i++) {
      const p = this.weights;
      const prefix = `l${i}`;
      const residual = x;
      let h = this.layerNorm(x, p[`${prefix}.g1`], p[`${prefix}.b1ln`]);

      const q = this.linear(h, p[`${prefix}.wq`], p[`${prefix}.bq`]);
      const k = this.linear(h, p[`${prefix}.wk`], p[`${prefix}.bk`]);
      const v = this.linear(h, p[`${prefix}.wv`], p[`${prefix}.bv`]);

      const qh = tf.transpose(
        assertTensor(tf.reshape(q, [batch, time, heads, dHead]), 'q reshape'),
        [0, 2, 1, 3],
      );
      const kh = tf.transpose(tf.reshape(k, [batch, time, heads, dHead]), [0, 2, 1, 3]);
      const vh = tf.transpose(tf.reshape(v, [batch, time, heads, dHead]), [0, 2, 1, 3]);

      const scores = tf.matMul(qh, kh, false, true);
      const scaled = tf.div(scores, Math.sqrt(dHead));
      const mask = this.mask.slice([0, 0, 0, 0], [1, 1, time, time]);
      const weights = tf.softmax(tf.add(scaled, mask), -1);
      const attended = tf.matMul(weights, vh);
      const merged = tf.reshape(tf.transpose(attended, [0, 2, 1, 3]), [batch, time, dModel]);
      const projected = this.linear(merged, p[`${prefix}.wo`], p[`${prefix}.bo`]);
      x = tf.add(residual, projected);

      const residual2 = x;
      h = this.layerNorm(x, p[`${prefix}.g2`], p[`${prefix}.b2ln`]);
      h = this.linear(h, p[`${prefix}.w1`], p[`${prefix}.b1`]);
      h = this.gelu(h);
      h = this.linear(h, p[`${prefix}.w2`], p[`${prefix}.b2`]);
      x = tf.add(residual2, h);
    }

    x = this.layerNorm(x, this.weights.gfin, this.weights.bfin);
    return this.linear(x, this.weights.wout, this.weights.bout);
  }

  loss(ids, targets) {
    const logits = this.forward(ids);
    const [batch, time] = ids.shape;
    const flatLogits = tf.reshape(logits, [batch * time, this.config.vocabSize]);
    const flatTargets = tf.reshape(targets, [batch * time]);
    const oneHot = tf.oneHot(flatTargets, this.config.vocabSize);
    return tf.losses.softmaxCrossEntropy(oneHot, flatLogits);
  }

  dispose() {
    for (const variable of this.vars) variable.dispose();
    this.mask.dispose();
  }

  async getWeightData() {
    const data = {};
    for (const [name, variable] of Object.entries(this.weights)) {
      const values = await variable.data();
      data[name] = Array.from(values);
    }
    return data;
  }

  async loadWeightData(data) {
    for (const [name, variable] of Object.entries(this.weights)) {
      const values = data[name];
      if (!Array.isArray(values)) throw new Error(`Missing weights: ${name}`);
      if (values.length !== variable.size) {
        throw new Error(`Bad weight shape for ${name}: expected ${variable.size}, got ${values.length}.`);
      }
      const next = tf.tensor(values, variable.shape);
      variable.assign(next);
      next.dispose();
    }
  }
}

export function createModel(tokenizer, overrides = {}) {
  return new TinyGPT(tokenizer, overrides);
}

export function randomWindow(data, time) {
  if (data.length < time + 2) {
    throw new Error(`Need at least ${time + 2} tokens to train. Add more text.`);
  }
  const start = Math.floor(Math.random() * (data.length - time - 1));
  return {
    input: data.slice(start, start + time),
    target: data.slice(start + 1, start + time + 1),
  };
}

/**
 * Train in place. Yielding to the event loop between steps keeps the page
 * responsive on the CPU backend and lets the UI repaint the loss chart.
 */
export async function trainModel(
  model,
  tokenData,
  {
    steps = 80,
    batch = 4,
    learningRate = 0.003,
    time = null,
    onStep = null,
    shouldStop = null,
  } = {},
) {
  const T = time || model.config.ctx;
  const optimizer = tf.train.adam(learningRate);
  const history = [];
  for (let step = 1; step <= steps; step++) {
    if (shouldStop && shouldStop()) break;
    const inputs = [];
    const targets = [];
    for (let b = 0; b < batch; b++) {
      const window = randomWindow(tokenData, T);
      inputs.push(window.input);
      targets.push(window.target);
    }
    const xs = tf.tensor2d(inputs, [batch, T], 'int32');
    const ys = tf.tensor2d(targets, [batch, T], 'int32');
    let lossTensor = null;
    try {
      lossTensor = optimizer.minimize(() => model.loss(xs, ys), true, model.vars);
      if (!lossTensor) throw new Error('The optimizer returned no loss.');
      const value = (await lossTensor.data())[0];
      if (!Number.isFinite(value)) {
        throw new Error('Training diverged (non-finite loss). Try a lower learning rate.');
      }
      history.push(value);
      if (onStep) await onStep({ step, loss: value, history });
    } finally {
      if (lossTensor) lossTensor.dispose();
      xs.dispose();
      ys.dispose();
    }
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  return history;
}

export function sampleFromLogits(logits, { temperature = 0.8, topK = 40, ban = [] } = {}) {
  const values = Array.from(logits);
  if (temperature <= 0) {
    let best = 0;
    for (let i = 1; i < values.length; i++) {
      if (values[i] > values[best]) best = i;
    }
    return best;
  }
  const banned = new Set(ban);
  for (const id of banned) {
    if (id >= 0 && id < values.length) values[id] = -1e9;
  }
  const scaled = values.map((v) => v / Math.max(0.05, temperature));
  const order = scaled
    .map((value, index) => ({ value, index }))
    .sort((a, b) => b.value - a.value)
    .slice(0, Math.max(1, Math.min(topK, scaled.length)));
  let max = -Infinity;
  for (const entry of order) max = Math.max(max, entry.value);
  let total = 0;
  for (const entry of order) {
    entry.weight = Math.exp(entry.value - max);
    total += entry.weight;
  }
  let roll = Math.random() * total;
  for (const entry of order) {
    roll -= entry.weight;
    if (roll <= 0) return entry.index;
  }
  return order[0].index;
}

/**
 * Sample one token at a time. No KV cache is used, which keeps this readable
 * and small enough to fit in a single static page.
 */
export async function generateText(
  model,
  tokenizer,
  promptIds,
  {
    maxTokens = 80,
    temperature = 0.8,
    topK = 40,
    stopIds = [],
    banIds = [],
    onToken = null,
    shouldStop = null,
  } = {},
) {
  const ctx = model.config.ctx;
  let ids = promptIds.slice(-ctx);
  const generated = [];
  const stop = new Set(stopIds);
  const banned = new Set([
    tokenizer.stoi.get(S.pad),
    tokenizer.stoi.get(S.unk),
    tokenizer.stoi.get(S.bos),
    tokenizer.stoi.get(S.user),
    tokenizer.stoi.get(S.assistant),
    ...banIds,
  ]);
  for (const id of stop) {
    banned.delete(id);
  }
  for (let i = 0; i < maxTokens; i++) {
    if (shouldStop && shouldStop()) break;
    const input = ids.slice(-ctx);
    // Pad on the right to a fixed context length. Real tokens cannot attend to
    // future pads because of the causal mask, and the constant shape prevents
    // WebGL from recompiling kernels for every new sequence length.
    const padId = tokenizer.stoi.get(S.pad);
    const padded = input.concat(new Array(Math.max(0, ctx - input.length)).fill(padId));
    const tensor = tf.tensor2d([padded], [1, padded.length], 'int32');
    let logits = null;
    let values = null;
    try {
      // Keep the full logits tensor, but dispose the activation graph. Reading
      // a sliced WebGL view has proven unreliable; index the last position in
      // JavaScript after the readback instead.
      logits = tf.tidy(() => model.forward(tensor));
      const flat = await logits.data();
      const offset = (input.length - 1) * model.config.vocabSize;
      values = flat.subarray(offset, offset + model.config.vocabSize);
    } finally {
      if (logits) logits.dispose();
      tensor.dispose();
    }
    const next = sampleFromLogits(values, { temperature, topK, ban: [...banned] });
    ids.push(next);
    generated.push(next);
    if (onToken) await onToken(next, i);
    if (stop.has(next)) break;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  return generated;
}

export async function exportModelData(model, meta = {}) {
  return {
    format: 'chainforge-browser-transformer',
    version: 1,
    createdAt: new Date().toISOString(),
    config: { ...model.config },
    tokenizer: tokenizerToData(model.tokenizer),
    meta,
    weights: await model.getWeightData(),
  };
}

export async function importModelData(data) {
  if (!data || data.format !== 'chainforge-browser-transformer') {
    throw new Error('Not a ChainForge browser model.');
  }
  const tokenizer = tokenizerFromData(data.tokenizer);
  const model = createModel(tokenizer, data.config || {});
  await model.loadWeightData(data.weights || {});
  return { model, meta: data.meta || {} };
}

export function paramsForConfig(vocabSize, config = DEFAULT_CONFIG) {
  const { ctx, dModel, layers, mlp } = { ...DEFAULT_CONFIG, ...config };
  const perLayer = 4 * dModel * dModel + 4 * dModel + 8 * mlp * dModel * dModel + 4 * dModel + 4 * dModel;
  return vocabSize * dModel + ctx * dModel + layers * perLayer + 2 * dModel + vocabSize;
}
