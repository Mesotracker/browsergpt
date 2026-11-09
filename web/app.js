'use strict';

(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const MAX_FILES = 24;

  const state = {
    files: [],
    mode: 'major',
    dataset: null,
    models: [],
    activeModel: null,
    trainingJobId: null,
    trainBusy: false,
    chatMessages: [],
    chatAbort: null,
    streaming: false,
    losses: [],
    valPoints: [],
    totalSteps: 0,
  };

  document.addEventListener('DOMContentLoaded', init);

  async function init() {
    bindTabs();
    bindTrain();
    bindChat();
    bindSave();
    loadToken();
    drawChart();
    const hash = location.hash.replace('#', '');
    if (['train', 'chat', 'save'].includes(hash)) setTab(hash, false);
    await Promise.allSettled([refreshModels(), refreshHealth()]);
    window.setInterval(refreshHealth, 30000);
    window.addEventListener('resize', drawChart);
  }

  /* ------------------------------------------------------------------ utils */
  const fmtCount = (value) => {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
    const n = Number(value);
    if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
    return String(Math.round(n));
  };

  const fmtDuration = (seconds) => {
    if (!Number.isFinite(seconds)) return '—';
    const total = Math.max(0, Math.round(seconds));
    if (total < 60) return `${total}s`;
    const minutes = Math.floor(total / 60);
    if (minutes < 60) return `${minutes}m ${total % 60}s`;
    return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
  };

  const fmtBytes = (bytes) => {
    if (!Number.isFinite(bytes)) return '—';
    if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
    if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
    if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(1)} kB`;
    return `${bytes} B`;
  };

  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  const clampInt = (value, min, max, fallback) => {
    const parsed = Number.parseInt(value, 10);
    if (!Number.isFinite(parsed)) return fallback;
    return Math.min(max, Math.max(min, parsed));
  };

  function authHeaders(extra = {}) {
    const token = localStorage.getItem('cf_token') || '';
    return token ? { ...extra, 'X-Chainforge-Token': token } : { ...extra };
  }

  function errorMessage(error) {
    if (!error) return 'Unknown error';
    if (typeof error === 'string') return error;
    return error.message || 'Request failed';
  }

  async function api(path, options = {}) {
    const response = await fetch(path, {
      ...options,
      headers: authHeaders(options.headers || {}),
    });
    if (!response.ok) {
      let detail = null;
      try {
        const body = await response.json();
        detail = body.detail;
      } catch (_) {
        /* non-JSON error body */
      }
      const message =
        typeof detail === 'string'
          ? detail
          : (detail && detail.message) || `Request failed (${response.status})`;
      const error = new Error(message);
      error.detail = detail;
      error.status = response.status;
      throw error;
    }
    return response.json();
  }

  async function streamSSE(url, options, onEvent) {
    const response = await fetch(url, {
      ...options,
      headers: authHeaders(options.headers || {}),
    });
    if (!response.ok) {
      let detail = null;
      try {
        const body = await response.json();
        detail = body.detail;
      } catch (_) {
        /* ignore */
      }
      const message =
        typeof detail === 'string'
          ? detail
          : (detail && detail.message) || `Request failed (${response.status})`;
      const error = new Error(message);
      error.detail = detail;
      throw error;
    }
    if (!response.body) throw new Error('This browser does not support streaming responses');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let boundary = buffer.indexOf('\n\n');
      while (boundary !== -1) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        for (const line of block.split('\n')) {
          if (!line.startsWith('data:')) continue;
          const payload = line.slice(5).trim();
          if (!payload) continue;
          try {
            onEvent(JSON.parse(payload));
          } catch (error) {
            if (error instanceof SyntaxError) {
              console.warn('Skipped malformed stream event', payload);
            } else {
              throw error;
            }
          }
        }
        boundary = buffer.indexOf('\n\n');
      }
    }
  }

  let toastTimer = null;
  function toast(message, kind = 'info') {
    const node = $('#toast');
    node.textContent = message;
    node.className = `toast show ${kind}`;
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => {
      node.className = 'toast';
    }, 4200);
  }

  /* ------------------------------------------------------------------- tabs */
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
    $$('.panel').forEach((panel) => {
      panel.classList.toggle('hidden', panel.id !== `panel-${name}`);
    });
    if (updateHash) history.replaceState(null, '', `#${name}`);
    if (name === 'save' || name === 'chat') refreshModels();
  }

  /* ----------------------------------------------------------------- health */
  async function refreshHealth() {
    const dot = $('#health-dot');
    const label = $('#health-label');
    try {
      const health = await api('/api/health');
      dot.className = 'status-dot online';
      label.textContent = `${health.device} · ${health.models} model${health.models === 1 ? '' : 's'}`;
    } catch (_) {
      dot.className = 'status-dot offline';
      label.textContent = 'server offline';
    }
  }

  /* ------------------------------------------------------------------ train */
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
    ['dragleave', 'dragend'].forEach((name) => {
      drop.addEventListener(name, () => drop.classList.remove('dragover'));
    });
    drop.addEventListener('drop', (event) => {
      event.preventDefault();
      drop.classList.remove('dragover');
      addFiles(Array.from(event.dataTransfer.files || []));
    });

    $$('.seg[data-mode]').forEach((button) => {
      button.addEventListener('click', () => {
        state.mode = button.dataset.mode;
        $$('.seg[data-mode]').forEach((other) => {
          other.classList.toggle('is-active', other === button);
        });
        $('#finetune-field').classList.toggle('hidden', state.mode !== 'finetune');
      });
    });

    $$('#step-presets .chip').forEach((chip) => {
      chip.addEventListener('click', () => {
        $('#steps').value = chip.dataset.steps;
        $$('#step-presets .chip').forEach((other) => {
          other.classList.toggle('is-active', other === chip);
        });
      });
    });

    $('#start-train').addEventListener('click', startTraining);
    $('#cancel-train').addEventListener('click', cancelTraining);
    $('#build-base').addEventListener('click', buildBaseModel);
  }

  function addFiles(incoming) {
    const existing = new Set(state.files.map((file) => `${file.name}:${file.size}:${file.lastModified}`));
    let added = 0;
    for (const file of incoming) {
      const key = `${file.name}:${file.size}:${file.lastModified}`;
      if (existing.has(key)) continue;
      if (state.files.length >= MAX_FILES) {
        toast(`Only ${MAX_FILES} files fit in one dataset`, 'error');
        break;
      }
      state.files.push(file);
      existing.add(key);
      added += 1;
    }
    if (added) resetDataset();
    renderFileList();
  }

  function renderFileList() {
    const list = $('#file-list');
    list.replaceChildren();
    state.files.forEach((file, index) => {
      const row = el('li', 'file-row');
      const extension = (file.name.split('.').pop() || 'txt').slice(0, 3);
      row.append(el('span', 'file-icon', extension));
      row.append(el('span', 'file-name', file.name));
      row.append(el('span', 'file-size', fmtBytes(file.size)));
      const remove = el('button', 'file-remove', '×');
      remove.type = 'button';
      remove.title = `Remove ${file.name}`;
      remove.setAttribute('aria-label', `Remove ${file.name}`);
      remove.addEventListener('click', () => {
        state.files.splice(index, 1);
        resetDataset();
        renderFileList();
      });
      row.append(remove);
      list.append(row);
    });
    const meta = $('#dataset-meta');
    if (!state.files.length) {
      meta.textContent = 'No files selected';
    } else {
      const bytes = state.files.reduce((sum, file) => sum + file.size, 0);
      meta.textContent = `${state.files.length} file${state.files.length === 1 ? '' : 's'} · ${fmtBytes(bytes)}`;
    }
    $('#dataset-badge').textContent = state.files.length ? `${state.files.length} queued` : 'no dataset';
    $('#dataset-badge').className = 'pill';
  }

  function resetDataset() {
    state.dataset = null;
    $('#upload-state').textContent = '';
    $('#dataset-note').classList.add('hidden');
    $('#base-status').textContent = '';
  }

  function resetRun() {
    state.losses = [];
    state.valPoints = [];
    state.totalSteps = 0;
    $('#stat-step').textContent = '0 / 0';
    $('#stat-loss').textContent = '—';
    $('#stat-val').textContent = '—';
    $('#stat-tps').textContent = '—';
    $('#stat-eta').textContent = '—';
    $('#run-device').textContent = '—';
    $('#train-progress').style.width = '0%';
    $('#train-result').classList.add('hidden');
    $('#train-result').replaceChildren();
    const log = $('#train-log');
    log.replaceChildren(el('div', 'log-line muted', 'Waiting for a run…'));
    drawChart();
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
    while (log.children.length > 220) log.firstElementChild.remove();
    log.scrollTop = log.scrollHeight;
  }

  function setTrainBusy(busy) {
    state.trainBusy = busy;
    const start = $('#start-train');
    start.disabled = busy;
    const cancel = $('#cancel-train');
    cancel.classList.toggle('hidden', !busy);
    cancel.disabled = false;
    cancel.textContent = 'Cancel run';
    $('#build-base').disabled = busy;
  }

  function syncStepFields() {
    $('#run-device').textContent = $('#steps').value ? `${$('#steps').value} steps` : '—';
  }

  async function uploadDataset() {
    $('#upload-state').textContent = 'uploading…';
    setRunStatus('extracting', 'gold');
    appendLog('Uploading and extracting documents…');
    const form = new FormData();
    state.files.forEach((file) => form.append('files', file, file.name));
    const data = await api('/api/datasets', { method: 'POST', body: form });
    state.dataset = data;
    $('#upload-state').textContent = 'ready';
    $('#dataset-badge').textContent = `${data.files.length} file${data.files.length === 1 ? '' : 's'}`;
    $('#dataset-badge').className = 'pill ok';
    $('#dataset-meta').textContent = `${data.total_chars.toLocaleString()} chars · ${fmtBytes(data.total_bytes)}`;
    const note = $('#dataset-note');
    if (data.rejected && data.rejected.length) {
      note.textContent = `Skipped ${data.rejected.length}: ` + data.rejected
        .slice(0, 3)
        .map((item) => `${item.name} — ${item.reason}`)
        .join(' · ');
      note.classList.remove('hidden');
      appendLog(`Skipped ${data.rejected.length} file(s)`, 'gold');
    } else {
      note.classList.add('hidden');
    }
    appendLog(`Extracted ${data.total_chars.toLocaleString()} characters from ${data.files.length} file(s)`);
    return data;
  }

  async function startTraining() {
    if (state.trainBusy) return;
    if (!state.files.length) {
      toast('Add at least one document first', 'error');
      $('#drop').focus();
      return;
    }
    if (state.mode === 'finetune' && !$('#parent-model').value) {
      toast('Choose a parent model to fine-tune', 'error');
      return;
    }
    const steps = clampInt($('#steps').value, 1, 20000, 300);
    $('#steps').value = steps;
    setTrainBusy(true);
    resetRun();
    try {
      let dataset = state.dataset;
      if (!dataset) dataset = await uploadDataset();

      const body = {
        dataset_id: dataset.id,
        mode: state.mode,
        name: $('#model-name').value.trim() || null,
        steps,
        batch_size: clampInt($('#batch').value, 1, 64, 16),
        lr: Number.parseFloat($('#lr').value) || null,
        parent: state.mode === 'finetune' ? $('#parent-model').value : null,
        seed: clampInt($('#seed').value, 0, 2147483647, 1337),
      };
      setRunStatus('queued', 'gold');
      appendLog(`Queueing ${state.mode === 'finetune' ? 'fine-tune' : 'major'} run · ${steps} steps`);
      const job = await api('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      state.trainingJobId = job.job_id;
      await followJob(job.job_id);
    } catch (error) {
      setRunStatus('error', 'bad');
      appendLog(errorMessage(error), 'bad');
      toast(errorMessage(error), 'error');
    } finally {
      state.trainingJobId = null;
      setTrainBusy(false);
    }
  }

  async function cancelTraining() {
    if (!state.trainingJobId) return;
    const button = $('#cancel-train');
    button.disabled = true;
    button.textContent = 'Cancelling…';
    try {
      await api(`/api/train/${state.trainingJobId}/cancel`, { method: 'POST' });
      appendLog('Cancellation requested; the run stops after the current step', 'gold');
    } catch (error) {
      button.disabled = false;
      button.textContent = 'Cancel run';
      toast(errorMessage(error), 'error');
    }
  }

  async function followJob(jobId) {
    await streamSSE(`/api/train/${jobId}/events`, {}, (event) => {
      handleTrainEvent(event);
    });
  }

  function handleTrainEvent(event) {
    switch (event.type) {
      case 'queued':
        setRunStatus('queued', 'gold');
        appendLog(event.label ? `${event.label} queued` : 'Job queued');
        break;
      case 'start':
        setRunStatus('running', 'gold');
        appendLog('Run started');
        break;
      case 'log':
        appendLog(event.message);
        break;
      case 'step': {
        state.losses.push({ step: event.step, loss: event.loss });
        state.totalSteps = event.total || state.totalSteps;
        $('#stat-step').textContent = `${event.step} / ${event.total}`;
        $('#stat-loss').textContent = Number(event.loss).toFixed(3);
        $('#stat-tps').textContent = fmtCount(event.tok_s);
        $('#stat-eta').textContent = fmtDuration(event.eta_s);
        $('#train-progress').style.width = `${Math.min(100, (event.step / Math.max(1, event.total)) * 100)}%`;
        if ($('#run-device').textContent === '—') $('#run-device').textContent = 'cpu · training';
        drawChart();
        break;
      }
      case 'eval':
        state.valPoints.push({ step: event.step, loss: event.val_loss });
        $('#stat-val').textContent = Number(event.val_loss).toFixed(3);
        appendLog(`validation @ ${event.step} · loss ${Number(event.val_loss).toFixed(3)}`);
        drawChart();
        break;
      case 'status':
        if (event.status === 'done') {
          setRunStatus('complete', 'ok');
          $('#train-progress').style.width = '100%';
          appendLog('Run complete. Model archived.', 'ok');
          if (event.result) {
            showTrainResult(event.result);
            if (event.result.model && event.result.model.name) {
              setActiveModel(event.result.model.name, false);
              toast(`Model ${event.result.model.name} is ready`, 'ok');
            }
          }
          refreshModels();
          refreshHealth();
        } else if (event.status === 'cancelled') {
          setRunStatus('cancelled', '');
          appendLog('Run cancelled', 'gold');
          toast('Training cancelled', 'info');
          refreshModels();
        } else {
          setRunStatus('error', 'bad');
          appendLog(event.error || 'Run failed', 'bad');
          toast(event.error || 'Training failed', 'error');
        }
        break;
      default:
        break;
    }
  }

  function showTrainResult(result) {
    const box = $('#train-result');
    box.replaceChildren();
    const model = result.model || {};
    const report = result.report || {};
    const copy = el('div');
    copy.append(el('strong', '', model.name ? `Model “${model.name}” archived` : 'Model archived'));
    const details = [
      report.params ? `${fmtCount(report.params)} params` : null,
      report.steps ? `${report.steps} steps` : null,
      report.final_loss !== undefined && report.final_loss !== null
        ? `train loss ${Number(report.final_loss).toFixed(3)}`
        : null,
      report.best_val_loss !== undefined && report.best_val_loss !== null
        ? `val loss ${Number(report.best_val_loss).toFixed(3)}`
        : null,
      report.seconds ? fmtDuration(report.seconds) : null,
    ].filter(Boolean);
    copy.append(el('span', '', details.join(' · ')));
    box.append(copy);
    if (model.name) {
      const button = el('button', 'btn gold small', 'Chat with it');
      button.type = 'button';
      button.addEventListener('click', () => {
        setActiveModel(model.name, true);
        setTab('chat');
      });
      box.append(button);
    }
    box.classList.remove('hidden');
  }

  async function buildBaseModel() {
    if (state.trainBusy) return;
    setTrainBusy(true);
    resetRun();
    $('#base-status').textContent = 'checking…';
    try {
      const response = await api('/api/models/base/build', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ steps: 300, preset: 'base' }),
      });
      if (response.status === 'exists') {
        $('#base-status').textContent = 'already available';
        toast('A base model is already in the vault', 'ok');
        await refreshModels();
        setActiveModel('base', false);
        return;
      }
      $('#base-status').textContent = 'building…';
      appendLog('Building the base model from corpus/base (300 steps)…');
      state.trainingJobId = response.job_id;
      await followJob(response.job_id);
      $('#base-status').textContent = 'ready';
    } catch (error) {
      $('#base-status').textContent = 'failed';
      setRunStatus('error', 'bad');
      appendLog(errorMessage(error), 'bad');
      toast(errorMessage(error), 'error');
    } finally {
      state.trainingJobId = null;
      setTrainBusy(false);
    }
  }

  /* ------------------------------------------------------------------ chart */
  function drawChart() {
    const canvas = $('#loss-canvas');
    const empty = $('#chart-empty');
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const points = state.losses;
    if (!points.length) {
      empty.classList.remove('hidden');
      return;
    }
    empty.classList.add('hidden');

    const pad = { left: 34, right: 12, top: 14, bottom: 18 };
    const width = rect.width - pad.left - pad.right;
    const height = rect.height - pad.top - pad.bottom;
    const losses = points.map((point) => point.loss).concat(state.valPoints.map((point) => point.loss));
    let min = Math.min(...losses);
    let max = Math.max(...losses);
    if (!Number.isFinite(min) || !Number.isFinite(max)) return;
    if (max - min < 0.05) {
      max += 0.025;
      min -= 0.025;
    } else {
      const margin = (max - min) * 0.08;
      max += margin;
      min -= margin;
    }
    const maxStep = Math.max(state.totalSteps || 0, points[points.length - 1].step, 1);
    const x = (step) => pad.left + (Math.max(0, step) / maxStep) * width;
    const y = (loss) => pad.top + height - ((loss - min) / (max - min)) * height;

    ctx.font = '9px ui-monospace, Menlo, monospace';
    ctx.textBaseline = 'middle';
    ctx.strokeStyle = 'rgba(73, 104, 151, 0.22)';
    ctx.fillStyle = '#63799c';
    ctx.lineWidth = 1;
    for (let index = 0; index <= 3; index += 1) {
      const value = min + ((max - min) * index) / 3;
      const lineY = y(value);
      ctx.beginPath();
      ctx.moveTo(pad.left, lineY);
      ctx.lineTo(pad.left + width, lineY);
      ctx.stroke();
      ctx.fillText(value.toFixed(2), 4, lineY);
    }

    const fill = ctx.createLinearGradient(0, pad.top, 0, pad.top + height);
    fill.addColorStop(0, 'rgba(242, 199, 92, 0.22)');
    fill.addColorStop(1, 'rgba(242, 199, 92, 0)');

    ctx.beginPath();
    points.forEach((point, index) => {
      const px = x(point.step);
      const py = y(point.loss);
      if (index === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    const last = points[points.length - 1];
    ctx.lineTo(x(last.step), pad.top + height);
    ctx.lineTo(x(points[0].step), pad.top + height);
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();

    ctx.beginPath();
    points.forEach((point, index) => {
      const px = x(point.step);
      const py = y(point.loss);
      if (index === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.strokeStyle = '#f2c75c';
    ctx.lineWidth = 2;
    ctx.shadowColor = 'rgba(242, 199, 92, 0.45)';
    ctx.shadowBlur = 8;
    ctx.stroke();
    ctx.shadowBlur = 0;

    ctx.fillStyle = '#6ba4ff';
    state.valPoints.forEach((point) => {
      ctx.beginPath();
      ctx.arc(x(point.step), y(point.loss), 2.6, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  /* ------------------------------------------------------------------- chat */
  function bindChat() {
    $('#chat-model').addEventListener('change', (event) => {
      state.activeModel = event.target.value || null;
    });
    const think = $('#think-tokens');
    think.addEventListener('input', () => {
      $('#think-value').textContent = think.value;
    });
    const answer = $('#answer-tokens');
    answer.addEventListener('input', () => {
      $('#answer-value').textContent = answer.value;
    });
    const temperature = $('#temperature');
    temperature.addEventListener('input', () => {
      $('#temp-value').textContent = Number(temperature.value).toFixed(2);
    });
    $('#composer').addEventListener('submit', (event) => {
      event.preventDefault();
      sendChat();
    });
    $('#chat-input').addEventListener('keydown', (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
        event.preventDefault();
        sendChat();
      }
    });
    $('#stop-chat').addEventListener('click', () => {
      if (state.chatAbort) state.chatAbort.abort();
    });
    $('#clear-chat').addEventListener('click', clearChat);
    renderChatEmpty();
  }

  function renderChatEmpty() {
    const transcript = $('#transcript');
    transcript.replaceChildren();
    const empty = el('div', 'empty-state');
    empty.id = 'chat-empty';
    empty.append(el('span', 'empty-glyph', '◇'));
    const title = el('p');
    title.append(el('strong', '', 'No messages yet.'));
    empty.append(title);
    empty.append(el('p', 'hint', 'Train a model first, then ask it about the documents it learned from.'));
    transcript.append(empty);
  }

  function clearChat() {
    if (state.streaming) return;
    state.chatMessages = [];
    renderChatEmpty();
    $('#budget-prompt').textContent = '—';
    $('#budget-think').textContent = '—';
    $('#budget-answer').textContent = '—';
    $('#budget-context').textContent = '—';
    $('#budget-note').textContent = 'Budget details appear after the first reply.';
    ['#meter-prompt', '#meter-think', '#meter-answer'].forEach((selector) => {
      $(selector).style.width = '0%';
    });
  }

  function scrollTranscript() {
    const transcript = $('#transcript');
    const distance = transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight;
    if (distance < 160) transcript.scrollTop = transcript.scrollHeight;
  }

  function appendUserMessage(text) {
    const wrap = el('div', 'msg user');
    const bubble = el('div', 'bubble');
    bubble.append(el('div', 'body', text));
    wrap.append(bubble);
    $('#transcript').append(wrap);
    const empty = $('#chat-empty');
    if (empty) empty.remove();
    scrollTranscript();
  }

  function appendAssistantShell() {
    const empty = $('#chat-empty');
    if (empty) empty.remove();
    const wrap = el('div', 'msg assistant');
    const bubble = el('div', 'bubble');
    const trace = document.createElement('details');
    trace.className = 'trace';
    trace.hidden = true;
    trace.open = true;
    const summary = document.createElement('summary');
    summary.append(el('span', 'trace-label', 'Reasoning draft'));
    const meta = el('span', 'trace-meta', '0 tk');
    summary.append(meta);
    const pre = el('pre');
    trace.append(summary, pre);
    const body = el('div', 'body');
    const foot = el('div', 'msg-foot');
    bubble.append(trace, body, foot);
    wrap.append(bubble);
    $('#transcript').append(wrap);
    scrollTranscript();
    return { wrap, bubble, trace, pre, meta, body, foot, traceText: '' };
  }

  function addFootChips(foot, budget, stopped = false) {
    foot.replaceChildren();
    if (!budget) return;
    const chips = [
      [`prompt ${budget.prompt_tokens} tk`, ''],
      [`reasoning ${budget.think_used}/${budget.think_budget}`, 'gold'],
      [`answer ${budget.answer_used}/${budget.answer_budget}`, 'blue'],
      [`ctx ${budget.context}`, ''],
    ];
    if (budget.truncated_tokens) chips.push([`trimmed ${budget.truncated_tokens} tk`, 'gold']);
    if (budget.reasoning_stop) chips.push([`reasoning: ${budget.reasoning_stop}`, '']);
    if (budget.answer_stop) chips.push([`answer: ${budget.answer_stop}`, '']);
    if (stopped) chips.push(['stopped', '']);
    chips.forEach(([label, kind]) => foot.append(el('span', `tk ${kind}`.trim(), label)));
  }

  function updateBudgetMeter(budget) {
    if (!budget) return;
    const context = Math.max(1, budget.context);
    const parts = [
      ['#meter-prompt', budget.prompt_tokens],
      ['#meter-think', budget.think_used],
      ['#meter-answer', budget.answer_used],
    ];
    parts.forEach(([selector, value]) => {
      $(selector).style.width = `${Math.min(100, (value / context) * 100)}%`;
    });
    $('#budget-prompt').textContent = budget.prompt_tokens;
    $('#budget-think').textContent = `${budget.think_used} / ${budget.think_budget}`;
    $('#budget-answer').textContent = `${budget.answer_used} / ${budget.answer_budget}`;
    $('#budget-context').textContent = budget.context;
    const notes = [];
    if (budget.truncated_tokens) notes.push(`${budget.truncated_tokens} prompt token(s) trimmed from the oldest history`);
    if (budget.reasoning_stop === 'budget') notes.push('reasoning hit its token budget');
    if (budget.answer_stop === 'budget') notes.push('answer hit its token budget');
    if (budget.reasoning_stop === 'endthink') notes.push('reasoning closed itself with <|endthink|>');
    if (budget.answer_stop === 'end') notes.push('answer ended with <|end|>');
    $('#budget-note').textContent = notes.join(' · ') || 'Budget details from the last reply.';
  }

  async function sendChat() {
    if (state.streaming) return;
    const input = $('#chat-input');
    const text = input.value.trim();
    if (!text) return;
    const model = $('#chat-model').value;
    if (!model) {
      toast('Train or build a model before chatting', 'error');
      setTab('train');
      return;
    }
    input.value = '';
    state.activeModel = model;
    appendUserMessage(text);
    state.chatMessages.push({ role: 'user', content: text });
    const shell = appendAssistantShell();
    const reasoning = $('#chat-reasoning').checked;
    const payload = {
      model,
      messages: state.chatMessages.slice(-12),
      reasoning,
      think_tokens: Number.parseInt($('#think-tokens').value, 10) || 0,
      answer_tokens: Number.parseInt($('#answer-tokens').value, 10) || 96,
      temperature: Number.parseFloat($('#temperature').value) || 0,
      top_k: Number.parseInt($('#top-k').value, 10) || 0,
    };

    state.streaming = true;
    $('#send-chat').disabled = true;
    $('#stop-chat').classList.remove('hidden');
    const controller = new AbortController();
    state.chatAbort = controller;
    let answerText = '';
    let budget = null;
    let streamError = null;

    try {
      await streamSSE(
        '/api/chat',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          signal: controller.signal,
        },
        (event) => {
          if (event.type === 'start') {
            budget = event.budget;
            if (reasoning && budget.think_budget > 0) {
              shell.trace.hidden = false;
              shell.meta.textContent = `0 / ${budget.think_budget} tk`;
            }
            updateBudgetMeter(budget);
          } else if (event.type === 'reasoning') {
            shell.trace.hidden = false;
            shell.traceText += event.text;
            shell.pre.textContent = shell.traceText;
            scrollTranscript();
          } else if (event.type === 'reasoning_end') {
            if (budget) shell.meta.textContent = `${event.tokens} / ${budget.think_budget} tk`;
            if (!reasoning) shell.trace.hidden = true;
          } else if (event.type === 'answer') {
            answerText += event.text;
            shell.body.textContent = answerText;
            scrollTranscript();
          } else if (event.type === 'done') {
            budget = event.budget;
            addFootChips(shell.foot, budget);
            updateBudgetMeter(budget);
          } else if (event.type === 'error') {
            streamError = new Error(event.message || 'Generation failed');
            throw streamError;
          }
        },
      );
      if (streamError) throw streamError;
      if (!answerText.trim()) {
        shell.body.textContent = '(the model produced no answer tokens — try more steps or a larger answer budget)';
      } else {
        state.chatMessages.push({ role: 'assistant', content: answerText.trim() });
      }
      if (budget) addFootChips(shell.foot, budget);
    } catch (error) {
      if (error.name === 'AbortError') {
        if (answerText.trim()) {
          state.chatMessages.push({ role: 'assistant', content: answerText.trim() });
          addFootChips(shell.foot, budget, true);
        } else {
          shell.wrap.remove();
        }
        toast('Generation stopped', 'info');
      } else {
        shell.wrap.classList.add('error');
        shell.body.textContent = errorMessage(error);
        shell.trace.hidden = true;
        toast(errorMessage(error), 'error');
      }
    } finally {
      state.streaming = false;
      state.chatAbort = null;
      $('#send-chat').disabled = false;
      $('#stop-chat').classList.add('hidden');
      $('#chat-input').focus();
      scrollTranscript();
    }
  }

  /* ------------------------------------------------------------------- save */
  function bindSave() {
    $('#refresh-models').addEventListener('click', refreshModels);
    const input = $('#import-input');
    const drop = $('#import-drop');
    drop.addEventListener('click', () => input.click());
    drop.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        input.click();
      }
    });
    input.addEventListener('change', () => {
      if (input.files && input.files[0]) importModel(input.files[0]);
      input.value = '';
    });
    ['dragenter', 'dragover'].forEach((name) => {
      drop.addEventListener(name, (event) => {
        event.preventDefault();
        drop.classList.add('dragover');
      });
    });
    ['dragleave', 'dragend'].forEach((name) => {
      drop.addEventListener(name, () => drop.classList.remove('dragover'));
    });
    drop.addEventListener('drop', (event) => {
      event.preventDefault();
      drop.classList.remove('dragover');
      const file = event.dataTransfer.files && event.dataTransfer.files[0];
      if (file) importModel(file);
    });
    $('#save-token').addEventListener('click', async () => {
      const value = $('#token-input').value.trim();
      if (value) localStorage.setItem('cf_token', value);
      else localStorage.removeItem('cf_token');
      toast(value ? 'Access token saved in this browser' : 'Access token cleared', 'ok');
      await Promise.allSettled([refreshModels(), refreshHealth()]);
    });
  }

  function loadToken() {
    $('#token-input').value = localStorage.getItem('cf_token') || '';
  }

  async function refreshModels() {
    try {
      const data = await api('/api/models');
      state.models = data.models || [];
      renderModels();
      populateModelSelects();
      refreshHealth();
    } catch (error) {
      state.models = [];
      const grid = $('#model-grid');
      grid.replaceChildren(el('p', 'hint', errorMessage(error)));
      populateModelSelects();
    }
  }

  function populateModelSelects() {
    const chat = $('#chat-model');
    const parent = $('#parent-model');
    const previousChat = state.activeModel || chat.value;
    const previousParent = parent.value;

    chat.replaceChildren();
    parent.replaceChildren();

    if (!state.models.length) {
      const option = new Option('No models — train one first', '');
      option.disabled = true;
      chat.append(option);
      chat.disabled = true;
      const parentOption = new Option('No models available', '');
      parentOption.disabled = true;
      parent.append(parentOption);
      parent.disabled = true;
      state.activeModel = null;
      return;
    }

    state.models.forEach((model) => {
      const label = `${model.name}${model.bundled ? ' · base' : ''} — ${fmtCount(model.params)} params`;
      chat.append(new Option(label, model.name));
      parent.append(new Option(label, model.name));
    });
    chat.disabled = false;
    parent.disabled = false;

    const chosen =
      state.models.find((model) => model.name === previousChat) ||
      state.models.find((model) => model.name === 'base') ||
      state.models[0];
    chat.value = chosen.name;
    state.activeModel = chosen.name;
    if (state.models.some((model) => model.name === previousParent)) {
      parent.value = previousParent;
    }
  }

  function setActiveModel(name, switchTab) {
    state.activeModel = name;
    if (state.models.some((model) => model.name === name)) {
      $('#chat-model').value = name;
      if (!$('#parent-model').disabled) $('#parent-model').value = name;
    }
    if (switchTab) setTab('chat');
  }

  function renderModels() {
    const grid = $('#model-grid');
    grid.replaceChildren();
    if (!state.models.length) {
      const empty = el('div', 'empty-state');
      empty.append(el('span', 'empty-glyph', '◇'));
      empty.append(el('p', 'hint', 'No models yet. Train one, build the base model, or import a .cgpt file.'));
      grid.append(empty);
      return;
    }
    const template = $('#tpl-model');
    state.models.forEach((model) => {
      const node = template.content.firstElementChild.cloneNode(true);
      node.querySelector('.model-name').textContent = model.name;
      const mode = node.querySelector('.model-mode');
      const isFineTune = model.lineage && model.lineage.mode === 'finetune';
      mode.textContent = model.bundled ? 'bundled' : isFineTune ? 'fine-tune' : 'major';
      mode.classList.add(model.bundled ? 'subtle' : 'gold');
      node.querySelector('.m-params').textContent = fmtCount(model.params);
      node.querySelector('.m-steps').textContent = model.steps ? fmtCount(model.steps) : '—';
      node.querySelector('.m-ctx').textContent = model.arch ? fmtCount(model.arch.ctx) : '—';
      node.querySelector('.m-loss').textContent =
        model.final_loss === null || model.final_loss === undefined
          ? '—'
          : Number(model.final_loss).toFixed(2);
      const lineage = [];
      if (model.created) {
        const date = new Date(model.created);
        if (!Number.isNaN(date.getTime())) lineage.push(date.toLocaleDateString());
      }
      if (model.lineage && model.lineage.parent) lineage.push(`from ${model.lineage.parent}`);
      if (model.size) lineage.push(fmtBytes(model.size));
      node.querySelector('.m-lineage').textContent = lineage.join(' · ') || 'no metadata';
      node.querySelector('.m-use').addEventListener('click', () => {
        setActiveModel(model.name, true);
        toast(`Chatting with ${model.name}`, 'ok');
      });
      node.querySelector('.m-download').addEventListener('click', () => downloadModel(model.name));
      const deleteButton = node.querySelector('.m-delete');
      if (model.bundled) {
        deleteButton.disabled = true;
        deleteButton.title = 'Bundled models are read-only';
      } else {
        deleteButton.addEventListener('click', () => deleteModel(model.name));
      }
      grid.append(node);
    });
  }

  async function downloadModel(name) {
    try {
      const response = await fetch(`/api/models/${encodeURIComponent(name)}/file`, {
        headers: authHeaders(),
      });
      if (!response.ok) throw new Error(`Download failed (${response.status})`);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${name}.cgpt`;
      document.body.append(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1500);
      toast(`Downloading ${name}.cgpt`, 'ok');
    } catch (error) {
      toast(errorMessage(error), 'error');
    }
  }

  async function deleteModel(name) {
    if (!window.confirm(`Delete “${name}”? This removes the archive from the server.`)) return;
    try {
      await api(`/api/models/${encodeURIComponent(name)}`, { method: 'DELETE' });
      toast(`Deleted ${name}`, 'ok');
      if (state.activeModel === name) state.activeModel = null;
      await refreshModels();
    } catch (error) {
      toast(errorMessage(error), 'error');
    }
  }

  async function importModel(file) {
    const button = $('#import-input');
    button.disabled = true;
    try {
      const form = new FormData();
      form.append('file', file, file.name);
      const result = await api('/api/models/import', { method: 'POST', body: form });
      toast(`Imported ${result.model.name}`, 'ok');
      state.activeModel = result.model.name;
      await refreshModels();
      setTab('save');
    } catch (error) {
      toast(errorMessage(error), 'error');
    } finally {
      button.disabled = false;
    }
  }
})();
