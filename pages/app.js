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
  paramsForConfig,
  prepareTrainingText,
  trainModel,
} from './engine.js';

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
const MAX_FILES = 24;
const MAX_FILE_CHARS = 300_000;
const MAX_TOTAL_CHARS = 1_200_000;
const DEFAULT_CONFIG = { ctx: 96, dModel: 48, heads: 4, layers: 1 };
const SEED_FILES = ['chainforge-manual.md', 'dialogue.md', 'field-notes.md', 'gpt-primer.md'];
const ACTIVE_KEY = 'chainforge_browser_active';
const DB_NAME = 'chainforge-browser';
const DB_STORE = 'models';
const FALLBACK_SEED = `<|user|>
What is ChainForge?
<|assistant|>
<|think|>
The user wants a short definition.
<|endthink|>
ChainForge is a small language-model workshop that runs in the browser.
<|end|>

<|user|>
Does it need a server?
<|assistant|>
<|think|>
No server is involved.
<|endthink|>
No. Training and chat happen locally in this page.
<|end|>`;

const state = {
  ready: false,
  seedTexts: [],
  files: [],
  mode: 'finetune',
  baseEntry: null,
  savedModels: [],
  models: [],
  modelCache: new Map(),
  activeModelId: 'base',
  training: false,
  cancel: false,
  losses: [],
  trainStarted: 0,
  chat: [],
  streaming: false,
  stopChat: false,
  idb: null,
  idbAvailable: true,
  memoryRecords: [],
  renderQueued: false,
};

document.addEventListener('DOMContentLoaded', init);

async function init() {
  $('#start-train').disabled = true;
  bindTabs();
  bindTrain();
  bindChat();
  bindSave();
  drawChart();
  window.addEventListener('resize', drawChart);

  await initBackend();
  await Promise.allSettled([loadSeedCorpus(), loadBaseModel()]);
  await loadVault();
  refreshModelViews();
  restoreActiveModel();
  state.ready = true;
  updateDatasetMeta();
  syncTrainButtons();
  await refreshArchitecturePreview();
  setTab(['train', 'chat', 'save'].includes(location.hash.slice(1)) ? location.hash.slice(1) : 'train', false);
}

/* ------------------------------------------------------------------ engine */

async function initBackend() {
  const dot = $('#health-dot');
  const label = $('#health-label');
  try {
    await tf.setBackend('webgl');
    await tf.ready();
    if (tf.getBackend() !== 'webgl') throw new Error('WebGL backend was not selected.');
  } catch (error) {
    console.warn('WebGL backend unavailable, using CPU.', error);
    try {
      await tf.setBackend('cpu');
      await tf.ready();
    } catch (cpuError) {
      console.error(cpuError);
      dot.className = 'status-dot offline';
      label.textContent = 'engine failed';
      toast('TensorFlow.js could not start in this browser.', 'error');
      return;
    }
  }
  dot.className = 'status-dot online';
  label.textContent = `${tf.getBackend()} · local`;
  window.addEventListener('beforeunload', () => {
    if (state.baseEntry?.model) state.baseEntry.model.dispose();
    for (const entry of state.modelCache.values()) {
      if (entry.model && entry !== state.baseEntry) entry.model.dispose();
    }
  });
}

function setEngineStatus(text) {
  const label = $('#health-label');
  if (label) label.textContent = text;
}

/* ------------------------------------------------------------------- seed */

async function loadSeedCorpus() {
  const results = await Promise.allSettled(
    SEED_FILES.map(async (name) => {
      const response = await fetch(`./corpus/${name}`, { cache: 'no-cache' });
      if (!response.ok) throw new Error(`${name} (${response.status})`);
      return response.text();
    }),
  );
  const texts = results.filter((result) => result.status === 'fulfilled').map((result) => result.value);
  state.seedTexts = texts.length ? texts : [FALLBACK_SEED];
  const chars = state.seedTexts.reduce((sum, text) => sum + text.length, 0);
  $('#seed-meta').textContent = `Bundled seed corpus (${chars.toLocaleString()} chars)`;
  if (texts.length !== SEED_FILES.length) {
    $('#seed-meta').textContent = `Fallback seed corpus (${chars.toLocaleString()} chars)`;
  }
  updateDatasetMeta();
}

/* --------------------------------------------------------------- base load */

async function loadBaseModel() {
  $('#base-status').textContent = 'Loading the bundled seed model…';
  try {
    const response = await fetch('./base-model.json', { cache: 'no-cache' });
    if (!response.ok) throw new Error(`base-model.json (${response.status})`);
    const data = await response.json();
    const restored = await importModelData(data);
    const previous = state.baseEntry;
    state.baseEntry = {
      id: 'base',
      name: data.meta?.name || 'ChainForge Seed',
      kind: 'base',
      readOnly: true,
      createdAt: data.createdAt || new Date().toISOString(),
      meta: data.meta || {},
      config: restored.model.config,
      data,
      model: restored.model,
    };
    if (previous?.model && previous !== state.baseEntry) previous.model.dispose();
    state.modelCache.set('base', state.baseEntry);
    $('#base-status').textContent = `Starter ready: ${fmtCount(restored.model.parameterCount)} params · ${data.meta?.steps || 0} steps · loss ${Number(data.meta?.loss).toFixed(2)}.`;
    $('#param-preview').textContent = fmtCount(restored.model.parameterCount);
    $('#context-preview').textContent = restored.model.config.ctx;
    $('#vocab-preview').textContent = restored.model.tokenizer.vocabSize;
    refreshModelViews();
  } catch (error) {
    console.error(error);
    state.baseEntry = null;
    $('#base-status').textContent = 'Starter model unavailable. Train a fresh model from the seed corpus.';
    if (state.savedModels.length === 0) {
      state.mode = 'fresh';
      $$('.seg[data-mode]').forEach((button) => button.classList.toggle('is-active', button.dataset.mode === 'fresh'));
      $('#finetune-field').classList.add('hidden');
    }
    toast('Starter model could not be loaded; fresh training is still available.', 'error');
    refreshModelViews();
  }
}

async function reloadBaseModel() {
  if (state.training) return;
  await loadBaseModel();
  await loadVault();
  refreshModelViews();
  if (state.activeModelId === 'base' && state.baseEntry) {
    state.chat = [];
    renderChat();
  }
  await refreshArchitecturePreview();
  toast('Starter model reloaded.', 'ok');
}

/* ------------------------------------------------------------------ tabs */

function bindTabs() {
  $$('.tab').forEach((tab) => {
    tab.addEventListener('click', () => setTab(tab.dataset.tab));
    tab.addEventListener('keydown', (event) => {
      if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return;
      const tabs = $$('.tab');
      const index = tabs.indexOf(tab);
      const next = tabs[(index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length];
      next.focus();
      setTab(next.dataset.tab);
    });
  });
}

function setTab(name, updateHash = true) {
  $$('.tab').forEach((tab) => {
    const active = tab.dataset.tab === name;
    tab.classList.toggle('is-active', active);
    tab.setAttribute('aria-selected', String(active));
  });
  $$('.panel').forEach((panel) => panel.classList.toggle('hidden', panel.id !== `panel-${name}`));
  if (updateHash) {
    history.replaceState(null, '', `#${name}`);
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    window.scrollTo({ top: 0, behavior: reduce ? 'auto' : 'smooth' });
  }
  if (name === 'save') renderModelList();
}

/* ----------------------------------------------------------------- utils */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function setText(selector, text) {
  const node = $(selector);
  if (node) node.textContent = text;
}

function fmtCount(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '—';
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(Math.round(n));
}

function fmtBytes(bytes) {
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(2)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(1)} kB`;
  return `${bytes} B`;
}

function fmtDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '—';
  const total = Math.round(seconds);
  if (total < 60) return `${total}s`;
  return `${Math.floor(total / 60)}m ${total % 60}s`;
}

function clampInt(value, min, max, fallback) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, parsed));
}

function errorMessage(error) {
  return error?.message || String(error) || 'Something went wrong.';
}

function toast(message, kind = 'info') {
  const node = $('#toast');
  node.textContent = message;
  node.className = `toast show ${kind}`;
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => {
    node.className = 'toast';
  }, 4400);
}

function safeName(name) {
  return String(name || 'chainforge-model').replace(/[^a-z0-9._-]+/gi, '-').replace(/^-+|-+$/g, '').slice(0, 80) || 'chainforge-model';
}

/* ------------------------------------------------------------------- idb */

function openDatabase() {
  if (state.idb) return Promise.resolve(state.idb);
  return new Promise((resolve, reject) => {
    if (!('indexedDB' in window)) {
      reject(new Error('This browser does not expose IndexedDB.'));
      return;
    }
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(DB_STORE)) db.createObjectStore(DB_STORE, { keyPath: 'id' });
    };
    request.onsuccess = () => {
      state.idb = request.result;
      resolve(state.idb);
    };
    request.onerror = () => reject(request.error || new Error('Could not open browser storage.'));
  });
}

function dbRequest(mode, action) {
  return openDatabase().then((db) => new Promise((resolve, reject) => {
    const transaction = db.transaction(DB_STORE, mode);
    const store = transaction.objectStore(DB_STORE);
    const request = action(store);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error('Browser storage request failed.'));
  }));
}

async function loadVault() {
  try {
    const records = await dbRequest('readonly', (store) => store.getAll());
    state.savedModels = records.map((record) => ({
      ...record,
      readOnly: false,
      kind: record.kind || 'user',
    }));
    state.idbAvailable = true;
  } catch (error) {
    console.warn(error);
    state.idbAvailable = false;
    state.savedModels = state.memoryRecords.slice();
  }
  refreshModelViews();
}

async function saveRecord(record) {
  if (state.idbAvailable) {
    try {
      await dbRequest('readwrite', (store) => store.put(record));
      return true;
    } catch (error) {
      console.warn(error);
      state.idbAvailable = false;
    }
  }
  state.memoryRecords = state.memoryRecords.filter((item) => item.id !== record.id);
  state.memoryRecords.push(record);
  toast('Browser storage is unavailable. This model will disappear when the tab closes.', 'error');
  return false;
}

async function deleteRecord(id) {
  if (state.idbAvailable) {
    try {
      await dbRequest('readwrite', (store) => store.delete(id));
    } catch (error) {
      console.warn(error);
    }
  }
  state.memoryRecords = state.memoryRecords.filter((item) => item.id !== id);
}

/* ------------------------------------------------------------- model view */

function refreshModelViews() {
  state.models = [
    ...(state.baseEntry ? [state.baseEntry] : []),
    ...state.savedModels.filter((model) => model.id !== 'base'),
  ];
  populateModelSelects();
  renderModelList();
}

function populateModelSelects() {
  const chatSelect = $('#chat-model');
  const parentSelect = $('#parent-model');
  const entries = state.models;
  const currentChat = state.activeModelId;
  const currentParent = parentSelect.value;
  chatSelect.replaceChildren();
  parentSelect.replaceChildren();
  if (!entries.length) {
    chatSelect.append(new Option('No models yet', ''));
    parentSelect.append(new Option('No parent available', ''));
    chatSelect.disabled = true;
    parentSelect.disabled = true;
    return;
  }
  chatSelect.disabled = false;
  parentSelect.disabled = false;
  for (const entry of entries) {
    const label = `${entry.name}${entry.kind === 'base' ? ' · starter' : ''}`;
    chatSelect.append(new Option(label, entry.id));
    parentSelect.append(new Option(label, entry.id));
  }
  chatSelect.value = entries.some((entry) => entry.id === currentChat) ? currentChat : entries[0].id;
  const fallbackParent = state.baseEntry?.id || entries[0].id;
  parentSelect.value = entries.some((entry) => entry.id === currentParent) ? currentParent : fallbackParent;
}

function restoreActiveModel() {
  const saved = localStorage.getItem(ACTIVE_KEY);
  if (saved && state.models.some((entry) => entry.id === saved)) {
    state.activeModelId = saved;
  } else if (state.baseEntry) {
    state.activeModelId = 'base';
  } else if (state.models[0]) {
    state.activeModelId = state.models[0].id;
  }
  setActiveModel(state.activeModelId, false);
}

async function setActiveModel(id, announce = true) {
  if (!state.models.some((entry) => entry.id === id)) return;
  state.activeModelId = id;
  localStorage.setItem(ACTIVE_KEY, id);
  const select = $('#chat-model');
  if (select) select.value = id;
  const entry = await getModelEntry(id);
  if (!entry) {
    toast('That model could not be loaded.', 'error');
    return;
  }
  setEngineStatus(`${tf.getBackend() || 'cpu'} · ${entry.name}`);
  if (announce) {
    renderChat();
    updateBudgetMeter(0, 0, 0, entry.model.config.ctx);
  }
}

async function getModelEntry(id) {
  if (!id) return null;
  if (state.modelCache.has(id)) return state.modelCache.get(id);
  const record = state.savedModels.find((item) => item.id === id);
  if (!record) return null;
  try {
    const data = typeof record.data === 'string' ? JSON.parse(record.data) : record.data;
    const restored = await importModelData(data);
    const entry = {
      ...record,
      data,
      model: restored.model,
      config: restored.model.config,
      meta: data.meta || record.meta || {},
    };
    state.modelCache.set(id, entry);
    return entry;
  } catch (error) {
    console.error(error);
    toast(`Could not load ${record.name}: ${errorMessage(error)}`, 'error');
    return null;
  }
}

async function getActiveEntry() {
  return getModelEntry(state.activeModelId);
}

function renderModelList() {
  const grid = $('#model-grid');
  if (!grid) return;
  grid.replaceChildren();
  if (!state.models.length) {
    grid.append(el('p', 'hint', 'No models yet. Train one in the Train tab.'));
  }
  for (const entry of state.models) {
    const fragment = $('#tpl-model').content.cloneNode(true);
    const card = fragment.querySelector('.model-card');
    const meta = entry.meta || {};
    const config = entry.config || meta.config || {};
    card.classList.toggle('is-base', entry.kind === 'base');
    fragment.querySelector('.model-name').textContent = entry.name;
    const mode = fragment.querySelector('.model-mode');
    mode.textContent = entry.kind === 'base' ? 'starter' : 'local';
    mode.className = `pill model-mode${entry.kind === 'base' ? ' gold' : ' ok'}`;
    fragment.querySelector('.m-params').textContent = fmtCount(meta.params || entry.model?.parameterCount);
    fragment.querySelector('.m-steps').textContent = meta.steps ? fmtCount(meta.steps) : '—';
    fragment.querySelector('.m-ctx').textContent = config.ctx ? `${config.ctx}` : '—';
    fragment.querySelector('.m-loss').textContent = Number.isFinite(Number(meta.loss)) ? Number(meta.loss).toFixed(3) : '—';
    const lineage = fragment.querySelector('.m-lineage');
    lineage.textContent = entry.kind === 'base'
      ? `Bundled seed corpus · ${fmtCount(meta.chars)} chars · read-only`
      : meta.parentName
        ? `Warm-started from ${meta.parentName} · ${fmtCount(meta.chars)} chars`
        : `Trained from scratch · ${fmtCount(meta.chars)} chars`;
    fragment.querySelector('.m-use').addEventListener('click', async () => {
      await setActiveModel(entry.id);
      toast(`${entry.name} selected for chat.`, 'ok');
      setTab('chat');
    });
    fragment.querySelector('.m-download').addEventListener('click', () => downloadEntry(entry));
    const deleteButton = fragment.querySelector('.m-delete');
    if (entry.kind === 'base') {
      deleteButton.classList.add('hidden');
    } else {
      deleteButton.addEventListener('click', () => removeEntry(entry));
    }
    grid.append(fragment);
  }
  if (state.idbAvailable) {
    $('#storage-status').textContent = `IndexedDB · ${state.savedModels.length} saved model${state.savedModels.length === 1 ? '' : 's'} in this browser. Clearing site data removes them.`;
  } else {
    $('#storage-status').textContent = 'Browser storage is unavailable. Models last only for this tab.';
  }
}

async function downloadEntry(entry) {
  try {
    let json = entry.data;
    if (typeof json !== 'string') json = JSON.stringify(json);
    if (!json) {
      if (!entry.model) throw new Error('The model is not loaded.');
      json = JSON.stringify(await exportModelData(entry.model, entry.meta || {}));
    }
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${safeName(entry.name)}.cgpt.json`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1500);
    toast('Archive download started.', 'ok');
  } catch (error) {
    toast(errorMessage(error), 'error');
  }
}

async function removeEntry(entry) {
  if (!window.confirm(`Delete “${entry.name}” from this browser? Download it first if you want to keep it.`)) return;
  const cached = state.modelCache.get(entry.id);
  if (cached?.model && entry.id !== 'base') cached.model.dispose();
  state.modelCache.delete(entry.id);
  state.savedModels = state.savedModels.filter((item) => item.id !== entry.id);
  await deleteRecord(entry.id);
  if (state.activeModelId === entry.id) {
    state.activeModelId = state.baseEntry ? 'base' : (state.savedModels[0]?.id || '');
    if (state.activeModelId) localStorage.setItem(ACTIVE_KEY, state.activeModelId);
  }
  refreshModelViews();
  renderChat();
  toast('Model deleted.', 'ok');
}

/* ---------------------------------------------------------------- train */

function bindTrain() {
  const drop = $('#drop');
  const input = $('#file-input');
  drop.addEventListener('click', () => input.click());
  drop.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      input.click();
    }
  });
  input.addEventListener('change', () => {
    addFiles(Array.from(input.files || []));
    input.value = '';
  });
  ['dragenter', 'dragover'].forEach((name) => {
    drop.addEventListener(name, (event) => {
      event.preventDefault();
      drop.classList.add('dragover');
    });
  });
  ['dragleave', 'dragend'].forEach((name) => drop.addEventListener(name, () => drop.classList.remove('dragover')));
  drop.addEventListener('drop', (event) => {
    event.preventDefault();
    drop.classList.remove('dragover');
    addFiles(Array.from(event.dataTransfer.files || []));
  });

  $$('.seg[data-mode]').forEach((button) => {
    button.addEventListener('click', () => {
      state.mode = button.dataset.mode;
      $$('.seg[data-mode]').forEach((other) => other.classList.toggle('is-active', other === button));
      $('#finetune-field').classList.toggle('hidden', state.mode !== 'finetune');
      syncTrainButtons();
      refreshArchitecturePreview();
    });
  });

  $('#parent-model').addEventListener('change', refreshArchitecturePreview);
  $$('#step-presets .chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      $('#steps').value = chip.dataset.steps;
      $$('#step-presets .chip').forEach((other) => other.classList.toggle('is-active', other === chip));
    });
  });
  $('#start-train').addEventListener('click', startTraining);
  $('#cancel-train').addEventListener('click', () => {
    if (!state.training) return;
    state.cancel = true;
    $('#cancel-train').disabled = true;
    $('#cancel-train').textContent = 'Cancelling…';
    appendLog('Cancellation requested; finishing the current step.', 'gold');
  });
  $('#reload-base').addEventListener('click', reloadBaseModel);
  $('#steps').addEventListener('input', syncTrainButtons);
}

async function readDocument(file) {
  const extension = (file.name.split('.').pop() || '').toLowerCase();
  if (file.size > 4 * 1024 * 1024) throw new Error('larger than 4 MB');
  if (['docx', 'pdf', 'doc', 'pptx', 'xlsx', 'zip', 'png', 'jpg', 'jpeg', 'mp3', 'mp4'].includes(extension)) {
    throw new Error('this static edition reads text; export it to .txt or .md first');
  }
  let text = await file.text();
  if (extension === 'html' || extension === 'htm' || extension === 'xml') {
    try {
      text = new DOMParser().parseFromString(text, 'text/html').body?.innerText || text;
    } catch (_) {
      /* keep the raw text */
    }
  }
  if (extension === 'ipynb') {
    try {
      const notebook = JSON.parse(text);
      text = (notebook.cells || [])
        .map((cell) => (Array.isArray(cell.source) ? cell.source.join('') : cell.source || ''))
        .join('\n\n');
    } catch (_) {
      /* keep the raw JSON */
    }
  }
  if (extension === 'json' || extension === 'jsonl') {
    try {
      const parsed = JSON.parse(text);
      if (extension === 'json') text = JSON.stringify(parsed, null, 2);
    } catch (_) {
      /* keep raw text */
    }
  }
  text = text.replace(/\r\n?/g, '\n').replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F]/g, ' ').trim();
  if (!text) throw new Error('no readable text found');
  return text.length > MAX_FILE_CHARS ? text.slice(0, MAX_FILE_CHARS) : text;
}

async function addFiles(incoming) {
  if (state.training) {
    toast('Wait for the current run to finish.', 'error');
    return;
  }
  const existing = new Set(state.files.map((file) => `${file.name}:${file.size}`));
  const errors = [];
  let added = 0;
  for (const file of incoming) {
    if (state.files.length >= MAX_FILES) {
      errors.push(`only ${MAX_FILES} files fit`);
      break;
    }
    const key = `${file.name}:${file.size}`;
    if (existing.has(key)) continue;
    try {
      const text = await readDocument(file);
      if (!text) throw new Error('no readable text');
      state.files.push({ name: file.name, size: file.size, text });
      existing.add(key);
      added += 1;
    } catch (error) {
      errors.push(`${file.name}: ${errorMessage(error)}`);
    }
  }
  if (added) {
    $('#upload-state').textContent = 'local only';
    toast(`Added ${added} document${added === 1 ? '' : 's'}.`, 'ok');
  }
  const note = $('#dataset-note');
  if (errors.length) {
    note.textContent = errors.slice(0, 3).join(' · ');
    note.classList.remove('hidden');
  } else {
    note.classList.add('hidden');
  }
  resetDatasetState();
  renderFileList();
  updateDatasetMeta();
  await refreshArchitecturePreview();
}

function resetDatasetState() {
  $('#upload-state').textContent = '';
}

function renderFileList() {
  const list = $('#file-list');
  list.replaceChildren();
  state.files.forEach((file, index) => {
    const row = el('li', 'file-row');
    row.append(el('span', 'file-icon', (file.name.split('.').pop() || 'txt').slice(0, 3)));
    row.append(el('span', 'file-name', file.name));
    row.append(el('span', 'file-size', `${fmtCount(file.text.length)} chars`));
    const remove = el('button', 'file-remove', '×');
    remove.type = 'button';
    remove.title = `Remove ${file.name}`;
    remove.setAttribute('aria-label', `Remove ${file.name}`);
    remove.addEventListener('click', () => {
      state.files.splice(index, 1);
      renderFileList();
      updateDatasetMeta();
      refreshArchitecturePreview();
    });
    row.append(remove);
    list.append(row);
  });
  const total = state.files.reduce((sum, file) => sum + file.size, 0);
  $('#dataset-meta').textContent = state.files.length
    ? `${state.files.length} file${state.files.length === 1 ? '' : 's'} · ${fmtBytes(total)}`
    : 'No files selected';
}

function updateDatasetMeta() {
  const seedChars = state.seedTexts.reduce((sum, text) => sum + text.length, 0);
  const fileChars = state.files.reduce((sum, file) => sum + file.text.length, 0);
  const total = seedChars + fileChars;
  if (state.files.length) {
    $('#dataset-badge').textContent = `${state.files.length} + seed`;
    $('#dataset-badge').className = 'pill ok';
    $('#dataset-meta').textContent = `${fmtCount(total)} chars · seed + ${state.files.length} file${state.files.length === 1 ? '' : 's'}`;
  } else {
    $('#dataset-badge').textContent = 'seed ready';
    $('#dataset-badge').className = 'pill ok';
    $('#dataset-meta').textContent = `${fmtCount(total)} chars · seed corpus`;
  }
  if (total > MAX_TOTAL_CHARS) {
    $('#dataset-meta').textContent += ' · capped';
  }
}

function currentTrainingText() {
  return prepareTrainingText(state.seedTexts, state.files.map((file) => file.text)).slice(0, MAX_TOTAL_CHARS);
}

async function refreshArchitecturePreview() {
  try {
    if (state.mode === 'finetune') {
      const parent = await getModelEntry($('#parent-model').value);
      if (parent) {
        $('#param-preview').textContent = fmtCount(parent.model.parameterCount);
        $('#context-preview').textContent = parent.model.config.ctx;
        $('#vocab-preview').textContent = parent.model.tokenizer.vocabSize;
        $('#parent-hint').textContent = `Keeps ${parent.name}'s tokenizer and architecture.`;
        return;
      }
    }
    const tokenizer = buildTokenizer([currentTrainingText()], 180);
    const params = paramsForConfig(tokenizer.vocabSize, DEFAULT_CONFIG);
    $('#param-preview').textContent = fmtCount(params);
    $('#context-preview').textContent = DEFAULT_CONFIG.ctx;
    $('#vocab-preview').textContent = tokenizer.vocabSize;
    $('#parent-hint').textContent = 'Fresh weights and a tokenizer learned from this dataset.';
  } catch (error) {
    console.warn(error);
  }
}

function syncTrainButtons() {
  const steps = clampInt($('#steps').value, 5, 2000, 80);
  const busy = state.training || !state.ready;
  $('#start-train').disabled = busy;
  $('#cancel-train').classList.toggle('hidden', !state.training);
  $('#reload-base').disabled = state.training;
  if (!state.training) $('#start-train span').textContent = state.mode === 'finetune' ? 'Start warm-start' : 'Start from scratch';
  else $('#start-train span').textContent = 'Training…';
  $('#run-device').textContent = `${tf.getBackend() || 'loading'} · ${steps} steps`;
}

function setTrainBusy(busy) {
  state.training = busy;
  syncTrainButtons();
}

function setRunStatus(text, kind = '') {
  const pill = $('#run-status');
  pill.textContent = text;
  pill.className = `pill ${kind}`.trim();
}

function appendLog(message, kind = '') {
  const log = $('#train-log');
  const placeholder = log.querySelector('.log-line.muted');
  if (placeholder && log.children.length === 1) placeholder.remove();
  const line = el('div', `log-line ${kind}`.trim());
  line.append(el('span', 'ts', new Date().toLocaleTimeString([], { hour12: false })));
  line.append(el('span', '', String(message)));
  log.append(line);
  while (log.children.length > 200) log.firstElementChild.remove();
  log.scrollTop = log.scrollHeight;
}

function resetRun() {
  state.losses = [];
  setText('#stat-step', '0 / 0');
  setText('#stat-loss', '—');
  setText('#stat-tps', '—');
  setText('#stat-ctx', '—');
  setText('#stat-eta', '—');
  $('#train-progress').style.width = '0%';
  $('#train-result').classList.add('hidden');
  $('#train-result').replaceChildren();
  const log = $('#train-log');
  log.replaceChildren(el('div', 'log-line muted', 'Waiting for a run…'));
  drawChart();
}

async function startTraining() {
  if (state.training || !state.ready) return;
  const steps = clampInt($('#steps').value, 5, 2000, 80);
  $('#steps').value = steps;
  const text = currentTrainingText();
  if (text.length < 300) {
    toast('Add a little more text before training.', 'error');
    return;
  }
  if (state.mode === 'finetune' && !$('#parent-model').value) {
    toast('Choose a parent model or switch to From scratch.', 'error');
    return;
  }

  state.training = true;
  state.cancel = false;
  setTrainBusy(true);
  setRunStatus('preparing', 'gold');
  resetRun();
  appendLog(state.mode === 'finetune' ? 'Snapshotting the parent model…' : 'Building a fresh model…');
  let working = null;
  let keepModel = false;
  try {
    let tokenizer;
    let parentName = null;
    if (state.mode === 'finetune') {
      const parent = await getModelEntry($('#parent-model').value);
      if (!parent) throw new Error('The parent model could not be loaded.');
      parentName = parent.name;
      const snapshot = await exportModelData(parent.model, parent.meta);
      const restored = await importModelData(snapshot);
      working = restored.model;
      tokenizer = working.tokenizer;
      appendLog(`Warm-starting from ${parent.name}.`, 'ok');
    } else {
      tokenizer = buildTokenizer([text], 180);
      working = createModel(tokenizer, DEFAULT_CONFIG);
      appendLog(`Fresh model · ${fmtCount(working.parameterCount)} parameters.`, 'ok');
    }

    const ids = encode(text, tokenizer);
    if (ids.length < working.config.ctx + 2) {
      throw new Error(`Need at least ${working.config.ctx + 2} tokens; got ${ids.length}.`);
    }
    const unknown = [...text].reduce((count, ch) => count + (tokenizer.stoi.has(ch) || ch === S.bos ? 0 : 1), 0);
    if (unknown > 0) appendLog(`${fmtCount(unknown)} characters map to the unknown token in this tokenizer.`, 'gold');
    setText('#stat-ctx', `${working.config.ctx}`);
    appendLog(`${fmtCount(ids.length)} tokens · context ${working.config.ctx} · batch ${clampInt($('#batch').value, 1, 16, 4)}`);
    setRunStatus('training', 'gold');
    state.trainStarted = Date.now();
    let lastLoss = null;
    const batch = clampInt($('#batch').value, 1, 16, 4);
    const lr = Math.max(0.0001, Math.min(0.02, Number.parseFloat($('#lr').value) || 0.003));

    await trainModel(working, ids, {
      steps,
      batch,
      learningRate: lr,
      time: working.config.ctx,
      shouldStop: () => state.cancel,
      onStep: async ({ step, loss, history }) => {
        lastLoss = loss;
        state.losses = history.slice();
        const elapsed = (Date.now() - state.trainStarted) / 1000;
        const rate = step / Math.max(0.001, elapsed);
        const tokensPerSecond = rate * batch * working.config.ctx;
        setText('#stat-step', `${step} / ${steps}`);
        setText('#stat-loss', Number(loss).toFixed(3));
        setText('#stat-tps', fmtCount(tokensPerSecond));
        setText('#stat-eta', fmtDuration((steps - step) / Math.max(0.01, rate)));
        $('#train-progress').style.width = `${Math.min(100, (step / steps) * 100)}%`;
        drawChart();
        if (step === 1 || step % 10 === 0 || step === steps) appendLog(`step ${step}/${steps} · loss ${loss.toFixed(3)}`);
      },
    });

    if (state.cancel) {
      setRunStatus('cancelled', 'bad');
      appendLog('Run cancelled. No archive was saved.', 'bad');
      toast('Training cancelled.', 'error');
      return;
    }
    if (lastLoss === null) throw new Error('Training did not report a step.');

    setRunStatus('saving', 'gold');
    appendLog('Packaging weights…');
    const name = $('#model-name').value.trim() || `Forge ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
    const meta = {
      name,
      kind: 'user',
      steps,
      loss: lastLoss,
      chars: text.length,
      vocab: tokenizer.vocabSize,
      params: working.parameterCount,
      parentName,
      config: { ...working.config },
      createdAt: new Date().toISOString(),
      backend: tf.getBackend(),
    };
    const data = await exportModelData(working, meta);
    const id = `m-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
    const json = JSON.stringify(data);
    const record = { id, name, kind: 'user', createdAt: data.createdAt, meta: data.meta, data: json };
    const saved = await saveRecord(record);
    const entry = { ...record, data, model: working, config: working.config, meta: data.meta };
    state.modelCache.set(id, entry);
    state.savedModels = state.savedModels.filter((item) => item.id !== id);
    state.savedModels.push(entry);
    state.activeModelId = id;
    localStorage.setItem(ACTIVE_KEY, id);
    keepModel = true;
    refreshModelViews();
    $('#chat-model').value = id;
    setRunStatus('complete', 'ok');
    appendLog(`Saved ${name} · ${fmtCount(working.parameterCount)} parameters · ${fmtBytes(json.length)}.`, 'ok');
    const result = $('#train-result');
    result.replaceChildren();
    const copy = el('div');
    copy.append(el('strong', '', `${name} is ready.`));
    copy.append(el('span', '', `${fmtCount(working.parameterCount)} params · loss ${lastLoss.toFixed(3)} · saved ${saved ? 'in this browser' : 'for this tab only'}`));
    const use = el('button', 'btn blue small', 'Open chat');
    use.type = 'button';
    use.addEventListener('click', async () => {
      await setActiveModel(id);
      setTab('chat');
    });
    result.append(copy, use);
    result.classList.remove('hidden');
    toast('Training complete. The model is ready in Chat.', 'ok');
    setEngineStatus(`${tf.getBackend()} · ${name}`);
  } catch (error) {
    console.error(error);
    setRunStatus('error', 'bad');
    appendLog(errorMessage(error), 'bad');
    toast(errorMessage(error), 'error');
  } finally {
    if (working && !keepModel) working.dispose();
    setTrainBusy(false);
    state.cancel = false;
    $('#cancel-train').disabled = false;
    $('#cancel-train').textContent = 'Cancel run';
    if (state.activeModelId) {
      const active = state.modelCache.get(state.activeModelId);
      if (active) setEngineStatus(`${tf.getBackend()} · ${active.name}`);
    }
  }
}

/* ----------------------------------------------------------------- chart */

function drawChart() {
  const canvas = $('#loss-canvas');
  if (!canvas) return;
  const bounds = canvas.getBoundingClientRect();
  const width = Math.max(280, bounds.width || 600);
  const height = Math.max(150, bounds.height || 190);
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const empty = $('#chart-empty');
  if (state.losses.length < 2) {
    if (empty) empty.classList.remove('hidden');
    return;
  }
  if (empty) empty.classList.add('hidden');
  const left = 34;
  const right = 10;
  const top = 12;
  const bottom = 20;
  const plotW = width - left - right;
  const plotH = height - top - bottom;
  const values = state.losses;
  const max = Math.max(...values);
  const min = Math.min(...values);
  const span = Math.max(0.02, max - min);
  ctx.font = '9px ui-monospace, monospace';
  ctx.fillStyle = '#647a9c';
  ctx.strokeStyle = 'rgba(27, 51, 85, 0.7)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = top + (plotH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(width - right, y);
    ctx.stroke();
    const value = max - (span / 4) * i;
    ctx.fillText(value.toFixed(2), 2, y + 3);
  }
  const point = (index) => {
    const x = left + (plotW * index) / Math.max(1, values.length - 1);
    const y = top + plotH - ((values[index] - min) / span) * plotH;
    return [x, y];
  };
  const gradient = ctx.createLinearGradient(0, top, 0, top + plotH);
  gradient.addColorStop(0, 'rgba(242, 199, 92, 0.26)');
  gradient.addColorStop(1, 'rgba(47, 123, 255, 0.02)');
  ctx.beginPath();
  values.forEach((_, index) => {
    const [x, y] = point(index);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.lineTo(width - right, top + plotH);
  ctx.lineTo(left, top + plotH);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();
  ctx.beginPath();
  values.forEach((value, index) => {
    const [x, y] = point(index);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = '#f2c75c';
  ctx.lineWidth = 2;
  ctx.stroke();
  const [lastX, lastY] = point(values.length - 1);
  ctx.beginPath();
  ctx.arc(lastX, lastY, 3.2, 0, Math.PI * 2);
  ctx.fillStyle = '#ffe6a6';
  ctx.fill();
  ctx.fillStyle = '#6d87ad';
  ctx.fillText('step 1', left, height - 5);
  const endLabel = `step ${values.length}`;
  ctx.fillText(endLabel, width - right - ctx.measureText(endLabel).width, height - 5);
}

/* ------------------------------------------------------------------ chat */

function bindChat() {
  $('#composer').addEventListener('submit', (event) => {
    event.preventDefault();
    sendChat();
  });
  $('#chat-input').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      sendChat();
    }
  });
  $('#stop-chat').addEventListener('click', () => {
    if (!state.streaming) return;
    state.stopChat = true;
    $('#stop-chat').disabled = true;
    $('#stop-chat').textContent = 'Stopping…';
  });
  $('#clear-chat').addEventListener('click', () => {
    if (state.streaming) return;
    state.chat = [];
    renderChat();
    $('#budget-note').textContent = 'Tokens are characters in the browser edition. Budget details appear after the first reply.';
  });
  $('#chat-model').addEventListener('change', async (event) => {
    await setActiveModel(event.target.value, false);
  });
  $('#chat-reasoning').addEventListener('change', () => {
    $('#think-tokens').disabled = !$('#chat-reasoning').checked;
  });
  $('#think-tokens').addEventListener('input', () => {
    setText('#think-value', $('#think-tokens').value);
  });
  $('#answer-tokens').addEventListener('input', () => {
    setText('#answer-value', $('#answer-tokens').value);
  });
  $('#temperature').addEventListener('input', () => {
    setText('#temp-value', Number($('#temperature').value).toFixed(2));
  });
  $$('#suggestions .chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      $('#chat-input').value = chip.dataset.prompt;
      $('#chat-input').focus();
    });
  });
  $('#think-tokens').disabled = !$('#chat-reasoning').checked;
}

function chatBudgets(ctx) {
  let think = $('#chat-reasoning').checked ? clampInt($('#think-tokens').value, 0, ctx, 16) : 0;
  let answer = clampInt($('#answer-tokens').value, 8, ctx, 48);
  // Keep at least 20 characters for the question and chat markers. Without
  // this reservation, the reasoning and answer budgets can starve the prompt.
  const room = Math.max(8, ctx - 20);
  let clamped = false;
  if (think + answer > room) {
    const scale = room / (think + answer);
    think = Math.floor(think * scale);
    answer = Math.max(8, Math.floor(answer * scale));
    clamped = true;
  }
  const reasoning = think > 0;
  return { think, answer, reasoning, clamped };
}

function formatChatPrompt(history, question, reasoning) {
  let text = S.bos;
  for (const message of history) {
    if (message.role === 'user') text += `${S.user}${message.content}${S.assistant}`;
    else if (message.role === 'assistant') text += `${message.content || ''}${S.end}`;
  }
  text += `${S.user}${question}${S.assistant}`;
  text += reasoning ? S.think : `${S.think}${S.endthink}`;
  return text;
}

function buildChatPrompt(tokenizer, history, question, reasoning, ctx, thinkBudget, answerBudget) {
  const queue = history.slice(-4);
  const reserve = thinkBudget + answerBudget + 2;
  const maxPrompt = Math.max(8, ctx - reserve);
  let q = question;
  let promptText = '';
  let ids = [];
  while (true) {
    promptText = formatChatPrompt(queue, q, reasoning);
    ids = encode(promptText, tokenizer);
    if (ids.length <= maxPrompt || q.length <= 4) break;
    if (queue.length) {
      queue.shift();
    } else {
      q = q.slice(0, Math.max(4, q.length - (ids.length - maxPrompt)));
    }
  }
  if (ids.length > ctx) ids = ids.slice(-ctx);
  const full = encode(formatChatPrompt(history.slice(-4), question, reasoning), tokenizer);
  return { ids, promptText, trimmed: ids.length < full.length };
}

async function sendChat() {
  if (state.streaming) return;
  const question = $('#chat-input').value.trim();
  if (!question) {
    $('#chat-input').focus();
    return;
  }
  const entry = await getActiveEntry();
  if (!entry) {
    toast('Train or select a model first.', 'error');
    return;
  }
  const ctx = entry.model.config.ctx;
  const tokenizer = entry.model.tokenizer;
  const budgets = chatBudgets(ctx);
  const history = state.chat.slice();
  state.chat.push({ role: 'user', content: question, meta: null });
  const assistant = { role: 'assistant', content: '', think: '', meta: null, streaming: true };
  state.chat.push(assistant);
  $('#chat-input').value = '';
  state.streaming = true;
  state.stopChat = false;
  setChatBusy(true);
  renderChat();
  const prompt = buildChatPrompt(tokenizer, history, question, budgets.reasoning, ctx, budgets.think, budgets.answer);
  let promptTokens = prompt.ids.length;
  let thinkTokens = 0;
  let answerTokens = 0;
  updateBudgetMeter(promptTokens, 0, 0, ctx);
  $('#budget-note').textContent = budgets.clamped
    ? `Budgets were scaled to fit the ${ctx}-character context of this model.`
    : `Prompt ${prompt.ids.length} + reasoning ${budgets.think} + answer ${budgets.answer} within ${ctx} characters.`;
  const endId = tokenizer.stoi.get(S.end);
  const endThinkId = tokenizer.stoi.get(S.endthink);
  try {
    let answerPrompt = prompt.ids.slice();
    if (budgets.reasoning && budgets.think > 0) {
      const thinkIds = await generateText(entry.model, tokenizer, answerPrompt, {
        maxTokens: budgets.think,
        temperature: Number($('#temperature').value),
        topK: clampInt($('#top-k').value, 1, 100, 30),
        stopIds: [endThinkId],
        banIds: [endId, tokenizer.stoi.get(S.user), tokenizer.stoi.get(S.assistant)],
        shouldStop: () => state.stopChat,
        onToken: async (id) => {
          if (id === endThinkId) return;
          thinkTokens += 1;
          assistant.think += decode([id], tokenizer);
          scheduleChatRender();
        },
      });
      thinkTokens = thinkIds.filter((id) => id !== endThinkId).length;
      const withEnd = thinkIds.slice();
      if (withEnd.at(-1) !== endThinkId) withEnd.push(endThinkId);
      answerPrompt = [...answerPrompt, ...withEnd];
      assistant.think = decode(withEnd.filter((id) => id !== endThinkId), tokenizer);
      if (state.stopChat) {
        assistant.content = '(stopped while drafting)';
        assistant.meta = {
          promptTokens,
          thinkTokens,
          answerTokens: 0,
          ctx,
          stop: 'stopped',
          trimmed: prompt.trimmed,
        };
        assistant.streaming = false;
        return;
      }
    }

    const answerIds = await generateText(entry.model, tokenizer, answerPrompt, {
      maxTokens: budgets.answer,
      temperature: Number($('#temperature').value),
      topK: clampInt($('#top-k').value, 1, 100, 30),
      stopIds: [endId],
      banIds: [endThinkId, tokenizer.stoi.get(S.think), tokenizer.stoi.get(S.user), tokenizer.stoi.get(S.assistant)],
      shouldStop: () => state.stopChat,
      onToken: async (id) => {
        if (id === endId) return;
        answerTokens += 1;
        assistant.content += decode([id], tokenizer);
        scheduleChatRender();
      },
    });
    answerTokens = answerIds.filter((id) => id !== endId).length;
    assistant.content = decode(answerIds.filter((id) => id !== endId), tokenizer).trim();
    if (!assistant.content) assistant.content = '(the model produced no visible characters)';
    assistant.meta = {
      promptTokens,
      thinkTokens,
      answerTokens,
      ctx,
      stop: state.stopChat ? 'stopped' : (answerIds.at(-1) === endId ? 'end' : 'budget'),
      trimmed: prompt.trimmed,
    };
    updateBudgetMeter(promptTokens, thinkTokens, answerTokens, ctx);
    $('#budget-note').textContent = `${assistant.meta.stop === 'end' ? 'The model emitted its end token.' : assistant.meta.stop === 'stopped' ? 'Stopped by you.' : 'Answer budget reached.'}${prompt.trimmed ? ' Older history was trimmed.' : ''}`;
  } catch (error) {
    console.error(error);
    assistant.error = true;
    assistant.content = errorMessage(error);
    assistant.meta = { promptTokens, thinkTokens, answerTokens, ctx, stop: 'error' };
    toast(errorMessage(error), 'error');
  } finally {
    assistant.streaming = false;
    state.streaming = false;
    state.stopChat = false;
    setChatBusy(false);
    renderChat();
  }
}

function setChatBusy(busy) {
  $('#send-chat').disabled = busy;
  $('#clear-chat').disabled = busy;
  $('#stop-chat').classList.toggle('hidden', !busy);
  $('#stop-chat').disabled = false;
  $('#stop-chat').textContent = 'Stop';
  $('#chat-model').disabled = busy;
}

function scheduleChatRender() {
  if (state.renderQueued) return;
  state.renderQueued = true;
  window.requestAnimationFrame(() => {
    state.renderQueued = false;
    renderChat(false);
    const transcript = $('#transcript');
    if (transcript) transcript.scrollTop = transcript.scrollHeight;
  });
}

function renderChat(scroll = true) {
  const transcript = $('#transcript');
  if (!transcript) return;
  transcript.replaceChildren();
  if (!state.chat.length) {
    const empty = el('div', 'empty-state');
    const strong = el('p');
    strong.append(el('strong', '', 'No messages yet.'));
    empty.append(
      el('span', 'empty-glyph', '◇'),
      strong,
      el('p', 'hint', 'The bundled starter is rough. Fine-tune it on your notes, then ask about what it learned.'),
    );
    transcript.append(empty);
    return;
  }
  for (const message of state.chat) {
    const wrap = el('div', `msg ${message.role}${message.error ? ' error' : ''}`);
    const bubble = el('div', 'bubble');
    if (message.role === 'assistant' && (message.think || message.streaming)) {
      const details = el('details', 'trace');
      details.open = Boolean(message.streaming);
      const summary = el('summary');
      summary.append(document.createTextNode('Reasoning draft'));
      const traceMeta = el('span', 'trace-meta', message.think ? `${message.think.length} chars` : 'writing');
      summary.append(traceMeta);
      details.append(summary, el('pre', '', message.think || '…'));
      bubble.append(details);
    }
    const body = el('div', 'body');
    body.textContent = message.content || (message.streaming ? '' : '');
    if (message.error) body.textContent = message.content;
    bubble.append(body);
    if (message.meta) {
      const foot = el('div', 'msg-foot');
      foot.append(el('span', 'tk blue', `prompt ${message.meta.promptTokens}`));
      foot.append(el('span', 'tk gold', `think ${message.meta.thinkTokens}`));
      foot.append(el('span', 'tk', `answer ${message.meta.answerTokens}`));
      foot.append(el('span', 'tk', `ctx ${message.meta.ctx}`));
      if (message.meta.stop) foot.append(el('span', `tk ${message.meta.stop === 'error' ? 'gold' : ''}`, message.meta.stop));
      bubble.append(foot);
    }
    wrap.append(bubble);
    transcript.append(wrap);
  }
  if (scroll) transcript.scrollTop = transcript.scrollHeight;
}

function updateBudgetMeter(prompt, think, answer, ctx) {
  const total = Math.max(1, ctx);
  const clamp = (n) => Math.min(100, Math.max(0, (n / total) * 100));
  $('#meter-prompt').style.width = `${clamp(prompt)}%`;
  $('#meter-think').style.width = `${clamp(think)}%`;
  $('#meter-answer').style.width = `${clamp(answer)}%`;
  setText('#budget-prompt', `${prompt}`);
  setText('#budget-think', `${think}`);
  setText('#budget-answer', `${answer}`);
  setText('#budget-context', `${ctx}`);
}

/* ------------------------------------------------------------------ save */

function bindSave() {
  const drop = $('#import-drop');
  const input = $('#import-input');
  drop.addEventListener('click', () => input.click());
  drop.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      input.click();
    }
  });
  input.addEventListener('change', () => {
    importArchive(Array.from(input.files || [])[0]);
    input.value = '';
  });
  ['dragenter', 'dragover'].forEach((name) => drop.addEventListener(name, (event) => {
    event.preventDefault();
    drop.classList.add('dragover');
  }));
  ['dragleave', 'dragend'].forEach((name) => drop.addEventListener(name, () => drop.classList.remove('dragover')));
  drop.addEventListener('drop', (event) => {
    event.preventDefault();
    drop.classList.remove('dragover');
    importArchive(Array.from(event.dataTransfer.files || [])[0]);
  });
  $('#refresh-models').addEventListener('click', async () => {
    await loadVault();
    refreshModelViews();
    toast('Vault refreshed.', 'ok');
  });
}

async function importArchive(file) {
  if (!file) return;
  const status = $('#import-status');
  status.className = 'hint';
  status.textContent = 'Reading archive…';
  try {
    if (file.size > 20 * 1024 * 1024) throw new Error('Archive is larger than 20 MB.');
    const parsed = JSON.parse(await file.text());
    if (parsed.format !== 'chainforge-browser-transformer') {
      throw new Error('This is not a ChainForge browser archive. Use the JSON export from this edition.');
    }
    const restored = await importModelData(parsed);
    const name = parsed.meta?.name || file.name.replace(/\.cgpt\.json$|\.json$/i, '') || 'Imported model';
    const id = `m-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
    const record = {
      id,
      name,
      kind: 'user',
      createdAt: new Date().toISOString(),
      meta: { ...(parsed.meta || {}), name, params: restored.model.parameterCount, config: { ...restored.model.config } },
      data: JSON.stringify(parsed),
    };
    const saved = await saveRecord(record);
    const entry = { ...record, data: parsed, model: restored.model, config: restored.model.config, meta: record.meta };
    state.modelCache.set(id, entry);
    state.savedModels.push(entry);
    state.activeModelId = id;
    localStorage.setItem(ACTIVE_KEY, id);
    refreshModelViews();
    $('#chat-model').value = id;
    status.className = 'hint import-status-ok';
    status.textContent = `Imported ${name}${saved ? '' : ' for this tab only'}. It is selected for chat.`;
    toast('Model imported.', 'ok');
  } catch (error) {
    console.error(error);
    status.className = 'hint import-status-bad';
    status.textContent = errorMessage(error);
    toast(errorMessage(error), 'error');
  }
}

/* --------------------------------------------------------------- keyboard */
