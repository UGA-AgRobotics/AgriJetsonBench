const fmt = {
  num(v, digits = 2) {
    return (v === null || v === undefined || Number.isNaN(Number(v))) ? '-' : Number(v).toFixed(digits);
  },
  secs(v) {
    if (v === null || v === undefined || Number.isNaN(Number(v))) return '-';
    const s = Math.max(0, Math.round(Number(v)));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    if (h > 0) return `${h}h ${m}m ${sec}s`;
    if (m > 0) return `${m}m ${sec}s`;
    return `${sec}s`;
  },
  pct(v) {
    return (v === null || v === undefined || Number.isNaN(Number(v))) ? '-' : `${Number(v).toFixed(0)}%`;
  },
};

let selectedDeviceKey = window.localStorage.getItem('selectedDeviceKey') || '';

function humanStage(value) {
  if (!value) return 'idle';
  return String(value).replace(/_/g, ' ');
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[ch]));
}

function text(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function drawLines(canvasId, seriesDefs) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#08101d';
  ctx.fillRect(0, 0, w, h);

  const padding = { left: 40, right: 12, top: 12, bottom: 22 };
  const plotW = w - padding.left - padding.right;
  const plotH = h - padding.top - padding.bottom;
  const points = [];
  seriesDefs.forEach(def => {
    (def.values || []).forEach(v => {
      const n = Number(v);
      if (Number.isFinite(n)) points.push(n);
    });
  });
  if (points.length === 0) {
    ctx.fillStyle = '#64748b';
    ctx.font = '14px sans-serif';
    ctx.fillText('No live data yet', 20, h / 2);
    return;
  }
  const min = Math.min(...points);
  const max = Math.max(...points);
  const span = max - min || 1;

  ctx.strokeStyle = '#1f2a3d';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padding.top + (plotH * i / 4);
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(w - padding.right, y);
    ctx.stroke();
  }
  ctx.strokeStyle = '#334155';
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, h - padding.bottom);
  ctx.lineTo(w - padding.right, h - padding.bottom);
  ctx.stroke();

  ctx.fillStyle = '#94a3b8';
  ctx.font = '12px sans-serif';
  ctx.fillText(max.toFixed(2), 6, padding.top + 6);
  ctx.fillText(min.toFixed(2), 6, h - padding.bottom);

  seriesDefs.forEach(def => {
    const vals = (def.values || []).map(v => Number(v));
    const usable = vals.filter(v => Number.isFinite(v));
    if (!usable.length) return;
    ctx.strokeStyle = def.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    vals.forEach((v, idx) => {
      if (!Number.isFinite(v)) return;
      const x = padding.left + (plotW * idx / Math.max(vals.length - 1, 1));
      const y = padding.top + plotH - ((v - min) / span) * plotH;
      if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });

  let legendX = padding.left;
  const legendY = 12;
  seriesDefs.forEach(def => {
    if (!(def.values || []).some(v => Number.isFinite(Number(v)))) return;
    ctx.fillStyle = def.color;
    ctx.fillRect(legendX, legendY, 12, 3);
    legendX += 16;
    ctx.fillStyle = '#cbd5e1';
    ctx.font = '12px sans-serif';
    ctx.fillText(def.label, legendX, legendY + 5);
    legendX += ctx.measureText(def.label).width + 16;
  });
}

function summarizeRails(hwmon) {
  if (!hwmon || !Array.isArray(hwmon.rails) || !hwmon.rails.length) return '-';
  return hwmon.rails.slice(0, 10).map(r => `${r.label}: ${fmt.num(r.power_w, 2)} W`).join('\n');
}

function summarizeThermals(thermal) {
  if (!thermal || !Array.isArray(thermal.zones) || !thermal.zones.length) return '-';
  return thermal.zones.slice(0, 10).map(z => `${z.type}: ${fmt.num(z.temp_c, 1)} C`).join('\n');
}

function setBadge(id, value, variant = 'muted') {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = value;
  el.className = `badge ${variant}`;
}

function isDeviceActive(device) {
  const state = String(device.state || '').toLowerCase();
  const stage = String(device.stage || '').toLowerCase();
  if (!device.last_seen_utc) return false;
  if (device.stale_s !== null && device.stale_s !== undefined && Number(device.stale_s) > 8) return false;
  return state === 'running' || (stage && !['idle', 'finished', 'repeat complete'].includes(stage));
}

function renderDeviceTabs(devices, selectedKey) {
  const wrap = document.getElementById('device-tabs');
  if (!wrap) return;
  const list = Array.isArray(devices) && devices.length ? devices : [
    { key: 'jetson-nano', label: 'Jetson Nano' },
    { key: 'jetson-agx', label: 'Jetson AGX' },
  ];
  wrap.innerHTML = list.map(device => {
    const key = escapeHtml(device.key);
    const label = escapeHtml(device.label || device.key);
    const selected = device.key === selectedKey ? ' active' : '';
    const active = isDeviceActive(device) ? ' live' : '';
    const meta = device.source_ip ? `<span>${escapeHtml(device.source_ip)}</span>` : '';
    return `
      <button class="device-tab${selected}${active}" type="button" data-device-key="${key}">
        <span class="device-name">${label}</span>
        ${meta}
      </button>
    `;
  }).join('');
  wrap.querySelectorAll('button[data-device-key]').forEach(button => {
    button.addEventListener('click', () => {
      selectedDeviceKey = button.dataset.deviceKey || '';
      window.localStorage.setItem('selectedDeviceKey', selectedDeviceKey);
      poll();
    });
  });
}

function renderAlerts(alerts) {
  const wrap = document.getElementById('alerts-list');
  if (!wrap) return;
  if (!alerts || !alerts.length) {
    wrap.innerHTML = 'No alerts yet.';
    return;
  }
  wrap.innerHTML = alerts.slice().reverse().map(a => {
    const cls = a.event_type === 'voltage_drop' ? 'danger' : 'success';
    return `
      <div class="alert-item ${cls}">
        <div class="title">${a.event_type}</div>
        <div>${a.message || ''}</div>
        <div class="meta">${a.timestamp_utc || ''} | baseline=${fmt.num(a.baseline, 3)} | value=${fmt.num(a.value, 3)}</div>
      </div>
    `;
  }).join('');
}

function summarizePowerSnapshot(event) {
  const parts = [];
  const maxTemp = event.max_temp_c ?? event.thermal_max_c;
  if (maxTemp !== undefined && maxTemp !== null) {
    parts.push(`max temp ${fmt.num(maxTemp, 1)} C`);
  }
  const thermal = event.thermal || event.thermal_zones || {};
  const thermalPairs = Object.entries(thermal)
    .filter(([, v]) => v !== null && v !== undefined)
    .slice(0, 8)
    .map(([k, v]) => `${k}=${fmt.num(v, 1)}C`);
  if (thermalPairs.length) parts.push(`temps: ${thermalPairs.join(', ')}`);

  const rails = event.rails || event.hwmon_rails || {};
  const railPairs = Object.entries(rails).slice(0, 8).map(([name, r]) => {
    if (!r || typeof r !== 'object') return `${name}`;
    const bits = [];
    if (r.voltage_mv !== undefined && r.voltage_mv !== null) bits.push(`${fmt.num(Number(r.voltage_mv) / 1000, 3)}V`);
    if (r.current_ma !== undefined && r.current_ma !== null) bits.push(`${fmt.num(Number(r.current_ma) / 1000, 3)}A`);
    if (r.power_mw !== undefined && r.power_mw !== null) bits.push(`${fmt.num(Number(r.power_mw) / 1000, 2)}W`);
    return `${name}: ${bits.join(' / ')}`;
  });
  if (railPairs.length) parts.push(`rails: ${railPairs.join('; ')}`);

  const oc = event.oc_counters || {};
  const ocPairs = Object.entries(oc).slice(0, 8).map(([k, v]) => `${k}=${v}`);
  if (ocPairs.length) parts.push(`oc: ${ocPairs.join(', ')}`);
  return parts.join('\n') || 'snapshot saved';
}

function renderPowerEvents(events) {
  const wrap = document.getElementById('power-events-list');
  if (!wrap) return;
  const count = Array.isArray(events) ? events.length : 0;
  setBadge('badge-power-events', count ? `${count} events` : '0 events', count ? 'danger' : 'muted');
  if (!count) {
    wrap.innerHTML = 'No throttle or over-current events yet.';
    return;
  }
  wrap.innerHTML = events.slice().reverse().map(e => {
    const title = e.kind || e.source || 'power_event';
    const key = e.key ? ` | ${e.key}` : '';
    const delta = e.delta !== undefined && e.delta !== null ? ` | delta=${e.delta}` : '';
    const path = e.snapshot_path || e.logger_event_log_path || '';
    return `
      <div class="alert-item danger power-event">
        <div class="title">${escapeHtml(title)}</div>
        <div>${escapeHtml(e.message || '')}</div>
        <div class="meta">${escapeHtml(e.timestamp_utc || e.server_received_utc || '')}${escapeHtml(key)}${escapeHtml(delta)}</div>
        <pre class="event-snapshot">${escapeHtml(summarizePowerSnapshot(e))}</pre>
        ${path ? `<div class="meta">${escapeHtml(path)}</div>` : ''}
      </div>
    `;
  }).join('');
}

function updateUI(data) {
  const devices = data.devices || [];
  if (!selectedDeviceKey || !devices.some(d => d.key === selectedDeviceKey)) {
    selectedDeviceKey = data.selected_device_key || (devices[0] && devices[0].key) || 'jetson-nano';
    window.localStorage.setItem('selectedDeviceKey', selectedDeviceKey);
  }
  renderDeviceTabs(devices, data.selected_device_key || selectedDeviceKey);

  const latest = data.latest_external || {};
  const currentRun = data.current_run || {};
  const jet = data.jetson_status || {};
  const internalLive = jet.internal_live || {};
  const quiet = jet.noise_guard || {};
  const hwmon = internalLive.hwmon || {};
  const thermal = internalLive.thermal || {};
  const tstats = internalLive.tegrastats || {};

  const stage = jet.stage || currentRun.status || 'idle';
  const stageLabel = humanStage(stage);
  text('stage-value', stageLabel);
  text('stage-message', jet.message || 'Waiting');
  text('repeat-value', jet.repeat_name || (jet.repeat_index ? `${jet.repeat_index}/${jet.total_repeats || '?'}` : '-'));
  text('run-id', `run_id: ${jet.run_id || currentRun.run_id || '-'}`);
  text('elapsed-value', fmt.secs(jet.overall_elapsed_s ?? jet.stage_elapsed_s ?? currentRun.elapsed_s));
  text('logger-elapsed', `logger: ${fmt.secs(currentRun.elapsed_s || 0)}`);
  text('eta-value', fmt.secs(jet.stage_eta_s ?? jet.overall_eta_s));
  text('progress-text', `${jet.processed_images || jet.stage_images_done || 0} / ${(jet.min_images_target || jet.stage_images_target || 0)}`);

  const progressPct = Math.max(0, Math.min(100, Number(jet.progress_pct || 0)));
  document.getElementById('progress-bar').style.width = `${progressPct}%`;
  text('progress-meta-left', `images: ${jet.processed_images || 0} / ${jet.min_images_target || '-'}`);
  text('progress-meta-right', `duration target: ${fmt.secs(jet.min_duration_s_target)}`);

  text('ext-voltage', fmt.num(latest.bus_voltage_v, 3));
  text('ext-current', fmt.num(latest.current_a, 3));
  text('ext-power', fmt.num(latest.power_w, 2));
  text('ext-energy', fmt.num(latest.energy_joules, 2));
  text('ext-temp', fmt.num(latest.external_temp_c ?? latest.tmp117_temp_c, 2));
  text('ext-temp-type', latest.temperature_sensor_type || 'sensor');
  text('ext-samples', String(currentRun.samples_written ?? 0));

  text('int-power', fmt.num(hwmon.total_power_w, 2));
  text('int-temp', fmt.num(thermal.max_temp_c ?? tstats.max_temp_c, 1));
  text('int-ram', fmt.num(tstats.ram_used_mb, 0));
  text('int-gr3d', fmt.pct(tstats.gr3d_pct));
  text('int-emc', fmt.pct(tstats.emc_pct));
  text('board-mode', `${jet.board_name || '-'} / ${(jet.power_mode || {}).power_mode_label || '-'}`);
  text('image-now', `image: ${jet.current_image_name || '-'}`);
  text('rail-list', summarizeRails(hwmon));
  text('thermal-list', summarizeThermals(thermal));

  const selectedDevice = devices.find(d => d.key === (data.selected_device_key || selectedDeviceKey)) || {};
  const deviceLabel = selectedDevice.label || jet.device_label || 'Jetson';
  text('device-summary', `${deviceLabel}${selectedDevice.source_ip ? ` | ${selectedDevice.source_ip}` : ''}`);
  setBadge('badge-device', deviceLabel, isDeviceActive(selectedDevice) ? '' : 'muted');
  setBadge('badge-stage', stageLabel, jet.state === 'error' ? 'danger' : '');
  setBadge('badge-logger', currentRun.active ? `logger ${currentRun.status}` : 'logger idle', currentRun.active ? '' : 'muted');
  const quietLabel = quiet.active ? `quiet ${quiet.profile || 'safe'}` : 'quiet restored';
  const quietVariant = (quiet.errors && quiet.errors.length) ? 'danger' : (quiet.active ? '' : 'muted');
  setBadge('badge-quiet', quiet.enabled === false ? 'quiet off' : quietLabel, quietVariant);
  setBadge('badge-updated', new Date().toLocaleTimeString(), '');

  const externalRecent = data.external_recent || [];
  const internalRecent = data.internal_recent || [];
  drawLines('chart-voltage', [{ label: 'Voltage', color: '#38bdf8', values: externalRecent.map(r => r.bus_voltage_v) }]);
  drawLines('chart-power', [{ label: 'Power', color: '#22c55e', values: externalRecent.map(r => r.power_w) }]);
  drawLines('chart-current', [{ label: 'Current', color: '#f59e0b', values: externalRecent.map(r => r.current_a) }]);
  drawLines('chart-temp', [
    { label: 'External temp', color: '#e879f9', values: externalRecent.map(r => r.external_temp_c ?? r.tmp117_temp_c) },
    { label: 'Internal max', color: '#ef4444', values: internalRecent.map(r => r.internal_max_temp_c) },
  ]);
  drawLines('chart-int-power', [{ label: 'Internal power', color: '#14b8a6', values: internalRecent.map(r => r.internal_total_power_w) }]);
  drawLines('chart-ram', [
    { label: 'RAM MB', color: '#a78bfa', values: internalRecent.map(r => r.ram_used_mb) },
    { label: 'GR3D %', color: '#f97316', values: internalRecent.map(r => r.gr3d_pct) },
  ]);

  renderAlerts(data.alerts_recent || []);
  renderPowerEvents(data.jetson_power_events || []);
}

async function poll() {
  try {
    const query = selectedDeviceKey ? `?device=${encodeURIComponent(selectedDeviceKey)}` : '';
    const resp = await fetch(`/api/live${query}`, { cache: 'no-store' });
    const data = await resp.json();
    updateUI(data);
  } catch (err) {
    text('stage-message', `Dashboard error: ${err}`);
  }
}

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').catch(() => {});
  });
}

poll();
setInterval(poll, 1000);
