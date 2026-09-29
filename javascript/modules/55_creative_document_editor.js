(function () {
  'use strict';

  const API = '/creative_document_api';
  const MAX_ASSET_FETCHES = 3;
  const OWNER_KEY_STORAGE = 'nexfocus.creative-document.owner.v1';

  function loadOrCreateOwnerKey() {
    try {
      let value = window.localStorage.getItem(OWNER_KEY_STORAGE);
      if (!value || !/^[A-Za-z0-9_-]{32,128}$/.test(value)) {
        const bytes = new Uint8Array(32);
        window.crypto.getRandomValues(bytes);
        value = btoa(String.fromCharCode(...bytes)).replaceAll('+', '-').replaceAll('/', '_').replace(/=+$/g, '');
        window.localStorage.setItem(OWNER_KEY_STORAGE, value);
      }
      return value;
    } catch (_) {
      return null;
    }
  }
  const SVG_NS = 'http://www.w3.org/2000/svg';

  class EditorFetchScheduler {
    constructor(limit, headers = () => ({})) {
      this.limit = Math.min(3, Math.max(1, limit || 3));
      this.headers = headers;
      this.active = 0;
      this.queue = [];
      this.peakActive = 0;
    }

    fetch(url, signal) {
      return new Promise((resolve, reject) => {
        const task = { url, signal, resolve, reject };
        if (signal && signal.aborted) {
          reject(new DOMException('Request cancelled', 'AbortError'));
          return;
        }
        this.queue.push(task);
        this._pump();
      });
    }

    _pump() {
      while (this.active < this.limit && this.queue.length) {
        const task = this.queue.shift();
        if (task.signal && task.signal.aborted) {
          task.reject(new DOMException('Request cancelled', 'AbortError'));
          continue;
        }
        this.active += 1;
        this.peakActive = Math.max(this.peakActive, this.active);
        fetch(task.url, { credentials: 'same-origin', headers: this.headers(), signal: task.signal, cache: 'force-cache' })
          .then((response) => {
            if (!response.ok) return response.json().catch(() => ({})).then((detail) => {
              const payload = detail && detail.detail ? detail.detail : detail;
              const error = new Error(payload && payload.message || payload && payload.code || `Asset request failed (${response.status})`);
              error.code = payload && payload.code;
              error.status = response.status;
              throw error;
            });
            return response.blob();
          })
          .then(task.resolve, task.reject)
          .finally(() => {
            this.active -= 1;
            this._pump();
          });
      }
    }

    cancelQueued() {
      const pending = this.queue.splice(0);
      for (const task of pending) task.reject(new DOMException('Request cancelled', 'AbortError'));
    }
  }

  function makeButton(label, action, title) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = label;
    button.dataset.action = action;
    if (title) button.title = title;
    return button;
  }

  function makeInput(type, value, action, min, max, step) {
    const input = document.createElement('input');
    input.type = type;
    input.value = value;
    input.dataset.action = action;
    if (min !== undefined) input.min = String(min);
    if (max !== undefined) input.max = String(max);
    if (step !== undefined) input.step = String(step);
    return input;
  }

  function matrixToAttrs(Konva, matrix) {
    // W02 stores row-major homogeneous matrices. Konva uses the canvas order
    // [a, b, c, d, e, f] where x'=a*x+c*y+e and y'=b*x+d*y+f.
    const values = [matrix[0], matrix[3], matrix[1], matrix[4], matrix[2], matrix[5]];
    const transform = new Konva.Transform(values);
    const attrs = transform.decompose();
    return {
      x: attrs.x,
      y: attrs.y,
      rotation: attrs.rotation,
      scaleX: attrs.scaleX,
      scaleY: attrs.scaleY,
      skewX: attrs.skewX || 0,
      skewY: attrs.skewY || 0,
      offsetX: 0,
      offsetY: 0,
    };
  }

  function attrsToMatrix(Konva, node) {
    const attrs = node.getAttrs();
    const transform = new Konva.Transform();
    transform.translate(attrs.x || 0, attrs.y || 0);
    transform.rotate(((attrs.rotation || 0) * Math.PI) / 180);
    transform.skew(attrs.skewX || 0, attrs.skewY || 0);
    transform.scale(attrs.scaleX === undefined ? 1 : attrs.scaleX, attrs.scaleY === undefined ? 1 : attrs.scaleY);
    const matrix = transform.getMatrix();
    return [matrix[0], matrix[2], matrix[4], matrix[1], matrix[3], matrix[5], 0, 0, 1];
  }

  class CreativeDocumentEditor extends HTMLElement {
    constructor() {
      super();
      this._connected = false;
      this._controller = null;
      this._resizeObserver = null;
      this._sessionObserver = null;
      this._scheduler = null;
      this._renderController = null;
      this._stage = null;
      this._mainLayer = null;
      this._overlayLayer = null;
      this._mainGroup = null;
      this._overlayGroup = null;
      this._transformer = null;
      this._activeNodeRecords = new Map();
      this._assetCache = new Map();
      this._viewSnapshots = new Map();
      this._objectUrls = new Set();
      this._selectedObjectIds = new Set();
      this._contextReferenceIds = new Set();
      this._polygonPoints = [];
      this._maskSeedPoints = [];
      this._selectionSeedPurpose = null;
      this._drawing = null;
      this._contextMaskId = null;
      this._editMaskId = null;
      this._lastPointer = null;
      this._currentObjectUrls = new Set();
      this._devicePixelRatio = window.devicePixelRatio || 1;
      this._tool = 'select';
      this._selectionMode = null;
      this._doc = null;
      this._capability = null;
      this._ownerKey = loadOrCreateOwnerKey();
      this._agentGrantId = null;
      this._agentGrantDocumentId = null;
      this._selectedLayerId = null;
      this._activeSelectionId = null;
      this._dirty = false;
      this._view = { zoom: 1, panX: 0, panY: 0, fitScale: 1, offsetX: 0, offsetY: 0 };
      this._report = { mountMs: 0, rendererLayers: 0, rendererCanvases: 0, interactionMs: [] };
      this._fitBounds = null;
      this._gestureFrame = 0;
      this._gestureObjectIds = new Set();
      this._actionInFlight = false;
      this._spacePanning = false;
      this._panPointerActive = false;
      this._fidelityBlocked = false;
      this._rendererIssues = [];
      this._gestureStartPositions = null;
    }

    connectedCallback() {
      if (this._connected) return;
      this._connected = true;
      const startedAt = performance.now();
      this._controller = new AbortController();
      this._scheduler = new EditorFetchScheduler(MAX_ASSET_FETCHES, () => this._capability && this._capability.enabled ? {
        Authorization: `Bearer ${this._capability.token}`,
        'X-Gradio-Session': this._capability.session,
        'X-Editor-Owner-Key': this._ownerKey || '',
      } : {});
      this._buildShell();
      this._wireDom();
      this._capabilityRaw = null;
      this._watchCapability();
      if (!window.Konva || window.Konva.version !== '10.6.0') {
        this._setDisabled(window.__nexCreativeEditorRendererError || 'Konva 10.6.0 could not be loaded. The editor is disabled.');
      } else {
        window.Konva.pixelRatio = window.devicePixelRatio || 1;
        this._setStatus(`Konva ${window.Konva.version} ready. Create or open a scene.`);
        this._enableButtons(!!(this._capability && this._capability.enabled));
      }
      this._resizeObserver = new ResizeObserver(() => this._resizeStage());
      const stageWrap = this.querySelector('.ncd-stage-wrap');
      if (stageWrap) this._resizeObserver.observe(stageWrap);
      if (this._doc) this._acceptView(this._doc).catch((error) => this._showError(error));
      this._report.mountMs = performance.now() - startedAt;
      this._setStatusMetric();
    }

    disconnectedCallback() {
      if (!this._connected) return;
      this._connected = false;
      if (this._controller) this._controller.abort();
      if (this._renderController) this._renderController.abort();
      if (this._sessionObserver) this._sessionObserver.disconnect();
      if (this._resizeObserver) this._resizeObserver.disconnect();
      if (this._scheduler) this._scheduler.cancelQueued();
      if (this._stage) {
        this._stage.off();
        this._stage.destroy();
        this._stage = null;
      }
      this._mainLayer = null;
      this._overlayLayer = null;
      this._mainGroup = null;
      this._overlayGroup = null;
      this._transformer = null;
      this._activeNodeRecords.clear();
      this._revokeObjectUrls();
      this._assetCache.clear();
      this._drawing = null;
      this._polygonPoints = [];
    }

    _buildShell() {
      this.innerHTML = '';
      const shell = document.createElement('div');
      shell.className = 'ncd-shell';
      shell.innerHTML = `
        <div class="ncd-topbar">
          <button type="button" data-action="new">New scene</button>
          <input type="text" data-action="document-id" placeholder="Document ID to open" aria-label="Document ID">
          <button type="button" data-action="open">Open</button>
          <button type="button" data-action="save">Save</button>
          <button type="button" data-action="preview">Preview composite</button>
          <button type="button" data-action="export">Export composite</button>
          <button type="button" data-action="import">Import image</button>
          <input type="file" data-action="file-input" accept="image/png,image/jpeg,image/webp" hidden>
          <span class="ncd-small" data-role="document-id-label">No scene open</span>
        </div>
        <div class="ncd-toolbar ncd-driver-tools" aria-label="Local Agent driver">
          <strong class="ncd-small">Local Agent driver</strong>
          <label class="ncd-small"><input type="checkbox" data-agent-scope="inspect" checked> Inspect</label>
          <label class="ncd-small"><input type="checkbox" data-agent-scope="propose"> Propose</label>
          <label class="ncd-small"><input type="checkbox" data-agent-scope="mutate"> Mutate</label>
          <button type="button" data-action="agent-enable">Enable &amp; download grant</button>
          <button type="button" data-action="agent-status">Grant status</button>
          <button type="button" data-action="agent-revoke">Revoke grant</button>
          <span class="ncd-small" data-role="agent-driver-status">Disabled</span>
        </div>
        <div class="ncd-toolbar">
          <button type="button" data-tool="select" aria-pressed="true">Select</button>
          <button type="button" data-tool="pan">Pan</button>
          <button type="button" data-tool="line">Line</button>
          <button type="button" data-tool="rectangle">Rectangle</button>
          <button type="button" data-tool="ellipse">Ellipse</button>
          <button type="button" data-tool="polygon">Polygon</button>
          <button type="button" data-tool="path">Path</button>
          <button type="button" data-tool="paint">Paint</button>
          <button type="button" data-tool="erase">Erase</button>
          <button type="button" data-tool="context-box">Context box</button>
          <button type="button" data-tool="context-polygon">Context polygon</button>
          <button type="button" data-action="finish-path">Finish path</button>
          <span class="ncd-divider"></span>
          <button type="button" data-action="fit">Fit</button>
          <button type="button" data-action="actual">100%</button>
          <button type="button" data-action="zoom-out">−</button>
          <button type="button" data-action="zoom-in">+</button>
          <input type="color" data-action="color" value="#e7b45a" title="Drawing color">
          <button type="button" class="ncd-color-swatch" data-color-swatch="#e7b45a" style="--swatch:#e7b45a" aria-label="Gold color"></button>
          <button type="button" class="ncd-color-swatch" data-color-swatch="#ffffff" style="--swatch:#ffffff" aria-label="White color"></button>
          <button type="button" class="ncd-color-swatch" data-color-swatch="#4d91e8" style="--swatch:#4d91e8" aria-label="Blue color"></button>
          <button type="button" class="ncd-color-swatch" data-color-swatch="#e65353" style="--swatch:#e65353" aria-label="Red color"></button>
          <label class="ncd-small">Size <input type="range" data-action="brush-size" min="1" max="256" value="14"></label>
          <label class="ncd-small">Opacity <input type="range" data-action="brush-opacity" min="0.05" max="1" step="0.05" value="1"></label>
          <label class="ncd-small">Hardness <input type="range" data-action="brush-hardness" min="0" max="1" step="0.05" value="0.8"></label>
          <button type="button" data-action="eyedropper">Pick color</button>
          <button type="button" data-action="rotate">Rotate 90°</button>
          <button type="button" data-action="flip">Flip</button>
          <button type="button" data-action="undo">Undo</button>
          <button type="button" data-action="redo">Redo</button>
        </div>
        <div class="ncd-toolbar ncd-selection-tools">
          <button type="button" data-selection-mode="box">Box selection</button>
          <button type="button" data-selection-mode="polygon">Polygon selection</button>
          <button type="button" data-selection-mode="paint">Paint seed</button>
          <button type="button" data-selection-mode="point">Point assist</button>
          <button type="button" data-action="selection-alpha">Use source alpha</button>
          <select data-action="selection-mask" aria-label="Allowlisted mask seed"><option value="">Choose selection, layer alpha, or visibility mask</option></select>
          <button type="button" data-action="selection-mask-create">Use mask seed</button>
          <label class="ncd-small">Semantic hint <input type="text" data-action="selection-hint" maxlength="160" placeholder="Optional subject or region"></label>
          <button type="button" data-action="finish-selection">Finish selection</button>
          <select data-action="mask-seed-target" aria-label="Mask seed use"><option value="refine">Refinement seed</option><option value="rebase">Rebase seed</option></select>
          <button type="button" data-action="draw-mask-seed">Draw refinement/rebase polygon</button>
          <button type="button" data-action="draw-paint-refinement">Paint refinement seed</button>
          <button type="button" data-action="duplicate-selection">Duplicate selection to layer</button>
          <select data-action="refine-operation" aria-label="Selection refinement">
            <option value="add">Add</option><option value="subtract">Subtract</option>
            <option value="intersect">Intersect</option><option value="invert">Invert</option>
            <option value="grow">Grow</option><option value="shrink">Shrink</option>
            <option value="feather">Feather</option><option value="clean">Edge cleanup</option>
          </select>
          <input type="number" data-action="mask-radius" value="4" min="1" max="128" aria-label="Mask radius in document pixels">
          <button type="button" data-action="refine-selection">Apply refinement</button>
          <button type="button" data-action="rebase-selection">Rebase selection</button>
        </div>
        <div class="ncd-body">
          <div class="ncd-stage-wrap"><div class="ncd-stage"></div></div>
          <aside class="ncd-side">
            <div class="ncd-side-head">Layers</div>
            <div class="ncd-toolbar">
              <button type="button" data-action="add-vector">+ Vector</button>
              <button type="button" data-action="add-paint">+ Paint</button>
              <button type="button" data-action="add-group">+ Group</button>
              <button type="button" data-action="duplicate-layer">Duplicate</button>
              <button type="button" data-action="move-up">↑</button>
              <button type="button" data-action="move-down">↓</button>
              <button type="button" data-action="delete-layer">Delete</button>
              <select data-action="parent-group" aria-label="Parent group"><option value="">Root level</option></select>
              <button type="button" data-action="reparent-layer">Move to parent</button>
            </div>
            <div class="ncd-layers" data-role="layers"></div>
            <div class="ncd-properties">
              <div class="ncd-property-row"><label for="ncd-layer-opacity">Layer opacity</label><input id="ncd-layer-opacity" type="range" data-action="layer-opacity" min="0" max="1" step="0.01" value="1"></div>
              <div class="ncd-property-row"><label for="ncd-layer-name">Selected layer</label><input id="ncd-layer-name" type="text" data-action="layer-name" maxlength="120"></div>
              <div class="ncd-property-row"><label>Zoom</label><span data-role="zoom-label">100%</span></div>
              <div class="ncd-property-row"><label>Context mask</label><select data-action="context-mask"></select></div>
              <div class="ncd-property-row"><label>Dilation (doc px)</label><input type="number" data-action="context-dilation" min="0" max="256" value="0"></div>
              <div class="ncd-property-row"><label>Edit mask</label><select data-action="edit-mask"></select></div>
              <div class="ncd-property-row"><button type="button" data-action="save-context">Save context references</button></div>
            <div class="ncd-small">Context references describe visible scene context. The edit mask remains a separate permission mask.</div>
            </div>
            <div class="ncd-relations"><strong>Scene relations</strong><div data-role="relations"></div></div>
            <div class="ncd-guides"><strong>Semantic guides</strong><div data-role="guides"></div><button type="button" data-action="create-guide">New proposed guide</button><button type="button" data-action="replace-guide">Mark selected guide replacement</button></div>
            <div class="ncd-selections"><strong>Persistent selections</strong><div data-role="selections"></div></div>
          </aside>
        </div>
        <div class="ncd-statusbar"><span data-role="status">Waiting for session capability.</span><span data-role="metrics"></span><span data-role="coordinates">x: —, y: —</span></div>
      `;
      this.appendChild(shell);
      this._renderStatus = shell.querySelector('[data-role="status"]');
      this._stageWrap = shell.querySelector('.ncd-stage-wrap');
      this._stageContainer = shell.querySelector('.ncd-stage');
      this._refreshToolButtons();
    }

    _wireDom() {
      const signal = this._controller.signal;
      this.addEventListener('click', (event) => this._onClick(event), { signal });
      this.addEventListener('change', (event) => this._onChange(event), { signal });
      const stageWrap = this.querySelector('.ncd-stage-wrap');
      if (stageWrap) stageWrap.tabIndex = 0;
      window.addEventListener('keyup', (event) => {
        if (event.code !== 'Space') return;
        this._spacePanning = false;
        if (this._stage && !this._panPointerActive) this._stage.draggable(false);
      }, { signal });
      window.addEventListener('blur', () => {
        this._spacePanning = false;
        if (this._stage) {
          if (this._stage.isDragging()) this._stage.stopDrag();
          this._stage.draggable(false);
        }
        this._panPointerActive = false;
      }, { signal });
      this.addEventListener('dblclick', (event) => {
        if (event.target.closest('.ncd-stage-wrap')) {
          if (this._polygonPoints.length > 1) this._finishPoints();
        }
      }, { signal });
      window.addEventListener('keydown', (event) => {
        if (event.code === 'Space' && !event.repeat && !/^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName)) {
          event.preventDefault();
          this._spacePanning = true;
          if (this._stage) this._stage.draggable(true);
        }
        if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
          event.preventDefault();
          this._sendAction(event.shiftKey ? 'redo' : 'undo');
        }
      }, { signal });
    }

    _watchCapability() {
      const fieldId = this.dataset.sessionField;
      if (!fieldId) return;
      const read = () => {
        const field = document.getElementById(fieldId);
        if (!field) return;
        const input = field.matches('input,textarea') ? field : field.querySelector('textarea,input');
        const raw = input ? input.value : field.textContent;
        if (!raw || raw === this._capabilityRaw) return;
        this._capabilityRaw = raw;
        try {
          const capability = JSON.parse(raw);
          this._capability = capability;
          if (!capability.enabled) this._setDisabled(capability.reason || 'This editor session is unavailable.');
          else {
            this._setStatus('Session ready. Create a scene or open a document ID.');
            this._enableButtons(true);
          }
        } catch (_) {
          this._setDisabled('The Gradio session capability could not be read. Reload the workspace.');
        }
      };
      read();
      this._sessionObserver = new MutationObserver(read);
      this._sessionObserver.observe(document.body, { childList: true, subtree: true, attributes: true, characterData: true });
      const field = document.getElementById(fieldId);
      if (field) {
        field.addEventListener('input', read, { signal: this._controller.signal });
        field.addEventListener('change', read, { signal: this._controller.signal });
      }
    }

    _setDisabled(message) {
      this._enableButtons(false);
      let banner = this.querySelector('.ncd-disabled');
      if (!banner) {
        banner = document.createElement('div');
        banner.className = 'ncd-disabled';
        this.querySelector('.ncd-shell')?.prepend(banner);
      }
      banner.textContent = message;
      this._setStatus(message, 'warning');
    }

    _enableButtons(enabled) {
      const blockedForFidelity = this._fidelityBlocked;
      for (const button of this.querySelectorAll('button')) {
        if (button.dataset.action === 'new' || button.dataset.action === 'open') button.disabled = !enabled || this._actionInFlight;
        else {
          const documentUnavailable = !this._doc && !['fit', 'actual', 'zoom-in', 'zoom-out'].includes(button.dataset.action);
          const keepsVisibilityAvailable = button.dataset.action === 'layer-eye';
          button.disabled = !enabled || documentUnavailable || (blockedForFidelity && !keepsVisibilityAvailable) || this._actionInFlight;
        }
      }
      for (const input of this.querySelectorAll('input,select')) input.disabled = !enabled || !this._doc || this._actionInFlight || blockedForFidelity;
      const fileInput = this.querySelector('[data-action="file-input"]');
      if (fileInput) fileInput.disabled = !enabled || this._actionInFlight || blockedForFidelity;
    }

    _setStatus(message, state) {
      if (!this._renderStatus) return;
      this._renderStatus.textContent = message;
      this._renderStatus.className = `ncd-status ${state || ''}`;
    }

    _setStatusMetric() {
      const value = this.querySelector('[data-role="metrics"]');
      if (!value) return;
      const surfaces = this._stage ? this._stage.getLayers().length : 0;
      const canvases = this._stage ? this._stageContainer.querySelectorAll('canvas').length : 0;
      const sortedLatency = this._report.interactionMs.slice().sort((a, b) => a - b);
      const p95 = sortedLatency.length ? sortedLatency[Math.min(sortedLatency.length - 1, Math.ceil(sortedLatency.length * 0.95) - 1)] : 0;
      const memory = performance.memory && Number.isFinite(performance.memory.usedJSHeapSize)
        ? ` · heap ${(performance.memory.usedJSHeapSize / (1024 * 1024)).toFixed(0)} MiB` : '';
      this._report.rendererLayers = surfaces;
      this._report.rendererCanvases = canvases;
      value.textContent = `mount ${this._report.mountMs.toFixed(1)} ms · surfaces ${surfaces} · canvases ${canvases} · fetch peak ${this._scheduler ? this._scheduler.peakActive : 0} · p95 edit ${p95.toFixed(1)} ms · DPR ${this._devicePixelRatio.toFixed(2)}${memory}`;
    }

    _refreshToolButtons() {
      for (const button of this.querySelectorAll('[data-tool]')) {
        button.setAttribute('aria-pressed', String(button.dataset.tool === this._tool));
      }
      for (const button of this.querySelectorAll('[data-selection-mode]')) {
        button.setAttribute('aria-pressed', String(button.dataset.selectionMode === this._selectionMode));
      }
      for (const record of this._activeNodeRecords.values()) {
        const interactive = !record.locked && !this._actionInFlight && !this._fidelityBlocked;
        record.node.listening(interactive);
        record.node.draggable(this._tool === 'select' && interactive);
        for (const hit of record.node.find('.ncd-raster-hit')) hit.listening(interactive);
      }
    }

    _onClick(event) {
      const toolButton = event.target.closest('[data-tool]');
      if (toolButton) {
        this._tool = toolButton.dataset.tool;
        this._selectionMode = null;
        this._polygonPoints = [];
        this._refreshToolButtons();
        return;
      }
      const selectionButton = event.target.closest('[data-selection-mode]');
      if (selectionButton) {
        this._selectionMode = selectionButton.dataset.selectionMode;
        this._tool = 'select';
        this._polygonPoints = [];
        this._selectionSeedPurpose = null;
        this._refreshToolButtons();
        this._setStatus(`Persistent ${this._selectionMode} selection: draw or click in document space.`);
        return;
      }
      const swatch = event.target.closest('[data-color-swatch]');
      if (swatch) {
        const color = this.querySelector('[data-action="color"]');
        if (color) color.value = swatch.dataset.colorSwatch;
        return;
      }
      const layerRow = event.target.closest('[data-layer-id]');
      if (layerRow && !event.target.closest('button,input')) {
        this._selectedLayerId = layerRow.dataset.layerId;
        if (!event.shiftKey) this._selectedObjectIds.clear();
        this._renderLayerList();
        this._renderProperties();
        return;
      }
      const objectRow = event.target.closest('[data-object-id]');
      if (objectRow && !event.target.closest('button')) {
        const objectId = objectRow.dataset.objectId;
        if (event.shiftKey) {
          if (this._selectedObjectIds.has(objectId)) this._selectedObjectIds.delete(objectId);
          else this._selectedObjectIds.add(objectId);
        } else {
          this._selectedObjectIds = new Set([objectId]);
        }
        this._updateTransformer();
        this._renderLayerList();
        this._renderProperties();
        return;
      }
      const button = event.target.closest('[data-action]');
      if (!button) return;
      const action = button.dataset.action;
      if (action === 'new') this._createDocument();
      else if (action === 'open') this._openDocument();
      else if (action === 'save') this._saveDocument();
      else if (action === 'agent-enable') this._enableAgentDriver();
      else if (action === 'agent-status') this._refreshAgentDriverStatus();
      else if (action === 'agent-revoke') this._revokeAgentDriver();
      else if (action === 'preview') this._previewComposite();
      else if (action === 'export') this._exportComposite();
      else if (action === 'import') this.querySelector('[data-action="file-input"]').click();
      else if (action === 'fit') this._fitToView();
      else if (action === 'actual') this._actualSize();
      else if (action === 'zoom-in') this._setZoom(this._view.zoom * 1.2);
      else if (action === 'zoom-out') this._setZoom(this._view.zoom / 1.2);
      else if (action === 'finish-path') this._finishPoints();
      else if (action === 'undo' || action === 'redo') this._sendAction(action);
      else if (action === 'add-vector') this._sendAction('add_layer', { kind: 'vector', name: 'Vector layer', parentId: this._parentGroupId() });
      else if (action === 'add-paint') this._sendAction('add_layer', { kind: 'paint', name: 'Paint layer', parentId: this._parentGroupId() });
      else if (action === 'add-group') this._sendAction('add_layer', { kind: 'group', name: 'Group', parentId: this._parentGroupId() });
      else if (action === 'duplicate-layer') this._runOnSelectedLayer('duplicate_layer', { layerId: this._selectedLayerId });
      else if (action === 'move-up') this._moveSelectedLayer('up');
      else if (action === 'move-down') this._moveSelectedLayer('down');
      else if (action === 'reparent-layer') this._runOnSelectedLayer('reparent_layer', { layerId: this._selectedLayerId, parentId: this._parentGroupId() });
      else if (action === 'delete-layer') this._deleteSelectedLayer();
      else if (action === 'rotate') this._transformSelected(Math.PI / 2, false);
      else if (action === 'flip') this._transformSelected(0, true);
      else if (action === 'eyedropper') this._tool = 'eyedropper';
      else if (action === 'duplicate-selection') this._duplicateSelection();
      else if (action === 'refine-selection') this._refineSelection();
      else if (action === 'finish-selection') this._finishSelectionPolygon();
      else if (action === 'selection-alpha') this._createSelection([{ kind: 'alpha' }]);
      else if (action === 'selection-mask-create') {
        const select = this.querySelector('[data-action="selection-mask"]');
        if (select && select.value) this._createSelection([{ kind: 'mask', maskId: select.value }]);
      }
      else if (action === 'draw-paint-refinement') {
        if (!this._activeSelection()) return this._setStatus('Create or choose a persistent selection first.', 'warning');
        this._selectionMode = 'refine-paint';
        this._tool = 'select';
        this._refreshToolButtons();
        this._setStatus('Paint a bounded add, subtract, or intersect seed in document space.');
      }
      else if (action === 'draw-mask-seed') {
        this._selectionSeedPurpose = this.querySelector('[data-action="mask-seed-target"]').value;
        this._maskSeedPoints = [];
        this._selectionMode = 'polygon';
        this._tool = 'select';
        this._polygonPoints = [];
        this._refreshToolButtons();
        this._setStatus(`Draw a polygon seed for ${this._selectionSeedPurpose}, then choose Finish selection.`);
      }
      else if (action === 'rebase-selection') this._rebaseSelection();
      else if (action === 'create-guide') this._createGuide();
      else if (action === 'replace-guide') this._markGuideReplacement();
      else if (action === 'save-context') this._saveRelationalContext();
      else if (action === 'file-input') return;
      else if (action.startsWith('guide-')) this._transitionGuide(button.dataset.guideId, action.slice(6));
      else if (action === 'selection-activate') {
        this._activeSelectionId = button.dataset.selectionId;
        this._renderSelectionList();
        this._renderDocument();
      } else if (action === 'layer-eye') this._toggleLayer(button.dataset.layerId, 'visible');
      else if (action === 'layer-lock') this._toggleLayer(button.dataset.layerId, 'lock');
    }

    _onChange(event) {
      const target = event.target;
      const action = target.dataset.action;
      if (action === 'file-input' && target.files && target.files[0]) this._importImage(target.files[0]);
      else if (action === 'layer-opacity') this._runOnSelectedLayer('set_layer_opacity', { layerId: this._selectedLayerId, opacity: Number(target.value) });
      else if (action === 'layer-name') this._runOnSelectedLayer('rename_layer', { layerId: this._selectedLayerId, name: target.value });
      else if (action === 'context-mask') { this._contextMaskId = target.value || null; this._renderDocument(); }
      else if (action === 'edit-mask') this._editMaskId = target.value || null;
      else if (action === 'brush-hardness') this._updateHardnessWarning(Number(target.value));
      else if (target.classList.contains('ncd-context-ref')) {
        const id = target.dataset.layerId;
        if (target.checked) this._contextReferenceIds.add(id);
        else this._contextReferenceIds.delete(id);
      }
    }

    _watchStage() {
      if (!this._stage) return;
      this._stage.on('pointerdown.creative', (event) => this._onPointerDown(event));
      this._stage.on('pointermove.creative', (event) => this._onPointerMove(event));
      this._stage.on('pointerup.creative', (event) => this._onPointerUp(event));
      this._stage.on('dragend.creative', () => {
        const p = this._stage.position();
        this._view.panX += p.x;
        this._view.panY += p.y;
        this._stage.position({ x: 0, y: 0 });
        this._stage.draggable(false);
        this._panPointerActive = false;
        this._applyViewport();
      });
    }

    _createStage() {
      if (!window.Konva || !this._stageContainer) return;
      if (this._stage) {
        this._stage.off();
        this._stage.destroy();
        this._stage = null;
      }
      this._stage = new window.Konva.Stage({ container: this._stageContainer, width: 800, height: 500 });
      this._mainLayer = new window.Konva.Layer({ listening: true, clearBeforeDraw: true });
      this._overlayLayer = new window.Konva.Layer({ listening: true, clearBeforeDraw: true });
      this._mainGroup = new window.Konva.Group({ name: 'ncd-main-content' });
      this._overlayGroup = new window.Konva.Group({ name: 'ncd-interaction-overlay' });
      this._mainLayer.add(this._mainGroup);
      this._overlayLayer.add(this._overlayGroup);
      this._transformer = new window.Konva.Transformer({
        rotateEnabled: true, resizeEnabled: true, keepRatio: false,
        ignoreStroke: true, flipEnabled: true,
        boundBoxFunc: (oldBox, newBox) => {
          if (newBox.width < 1 || newBox.height < 1 || Math.abs(newBox.width) > this._doc.width * 4 || Math.abs(newBox.height) > this._doc.height * 4) return oldBox;
          return newBox;
        },
      });
      this._overlayLayer.add(this._transformer);
      this._stage.add(this._mainLayer);
      this._stage.add(this._overlayLayer);
      this._stage.draggable(false);
      this._watchStage();
      this._resizeStage();
    }

    _resizeStage() {
      if (!this._stage || !this._stageWrap) return;
      const bounds = this._stageWrap.getBoundingClientRect();
      const width = Math.max(200, Math.floor(bounds.width));
      const height = Math.max(300, Math.floor(bounds.height));
      if (this._stage.width() !== width) this._stage.width(width);
      if (this._stage.height() !== height) this._stage.height(height);
      this._devicePixelRatio = window.devicePixelRatio || 1;
      if (window.Konva) window.Konva.pixelRatio = this._devicePixelRatio;
      if (this._doc && !this._fitBounds) this._fitToView();
      else this._applyViewport();
      this._setStatusMetric();
    }

    _applyViewport() {
      if (!this._stage || !this._mainGroup || !this._overlayGroup || !this._doc) return;
      if (!this._stage.isDragging()) {
        const stagePosition = this._stage.position();
        if (stagePosition.x || stagePosition.y) {
          this._view.panX += stagePosition.x;
          this._view.panY += stagePosition.y;
          this._stage.position({ x: 0, y: 0 });
        }
      }
      const width = this._stage.width();
      const height = this._stage.height();
      const fit = Math.min(width / this._doc.width, height / this._doc.height);
      this._view.fitScale = fit;
      this._view.offsetX = (width - this._doc.width * fit) / 2;
      this._view.offsetY = (height - this._doc.height * fit) / 2;
      const scale = fit * this._view.zoom;
      const x = this._view.offsetX + this._view.panX;
      const y = this._view.offsetY + this._view.panY;
      for (const group of [this._mainGroup, this._overlayGroup]) {
        group.position({ x, y });
        group.scale({ x: scale, y: scale });
      }
      this._stage.batchDraw();
      const zoom = this.querySelector('[data-role="zoom-label"]');
      if (zoom) zoom.textContent = `${Math.round(scale * 100)}%`;
    }

    _fitToView() {
      this._view.zoom = 1;
      this._view.panX = 0;
      this._view.panY = 0;
      this._fitBounds = true;
      this._applyViewport();
    }

    _actualSize() {
      if (!this._doc || !this._stage) return;
      const fit = Math.min(this._stage.width() / this._doc.width, this._stage.height() / this._doc.height);
      this._view.zoom = Math.min(8, Math.max(0.1, 1 / fit));
      this._view.panX = 0;
      this._view.panY = 0;
      this._applyViewport();
    }

    _setZoom(value) {
      const prior = this._view.zoom;
      this._view.zoom = Math.max(0.1, Math.min(8, value));
      if (this._stage) {
        const center = { x: this._stage.width() / 2, y: this._stage.height() / 2 };
        const docPoint = this._previewToDocument(center.x, center.y, prior);
        const scale = this._view.fitScale * this._view.zoom;
        this._view.panX = center.x - this._view.offsetX - docPoint.x * scale;
        this._view.panY = center.y - this._view.offsetY - docPoint.y * scale;
      }
      this._fitBounds = true;
      this._applyViewport();
    }

    _previewToDocument(x, y, zoom) {
      const scale = this._view.fitScale * (zoom || this._view.zoom);
      return { x: (x - this._view.offsetX - this._view.panX) / scale, y: (y - this._view.offsetY - this._view.panY) / scale };
    }

    _pointerDocumentPoint() {
      const point = this._stage && this._stage.getPointerPosition();
      return point ? this._previewToDocument(point.x, point.y) : null;
    }

    _onPointerDown(event) {
      if (!this._doc || !this._stage || this._actionInFlight) return;
      if (this._tool === 'eyedropper') {
        this._sampleColor();
        this._tool = 'select';
        this._refreshToolButtons();
        return;
      }
      if (this._tool === 'pan' || event.evt.button === 1 || this._spacePanning) {
        this._panPointerActive = true;
        this._stage.draggable(true);
        if (!this._stage.isDragging()) this._stage.startDrag();
        return;
      }
      if (this._fidelityBlocked) return;
      const target = event.target;
      if (target !== this._stage && target !== this._mainLayer && target !== this._mainGroup && !target.hasName('ncd-background')) {
        if (this._tool === 'select' && this._selectionMode === null) return;
      }
      const point = this._pointerDocumentPoint();
      if (!point) return;
      this._lastPointer = point;
      const active = this._selectedLayer();
      if (this._tool === 'context-polygon') {
        this._polygonPoints.push([point.x, point.y]);
        this._drawPointPreview();
        return;
      }
      if (this._tool === 'path' || this._tool === 'polygon') {
        this._polygonPoints.push([point.x, point.y]);
        this._drawPointPreview();
        return;
      }
      if (this._selectionMode === 'point') {
        if (!active) return this._setStatus('Select a raster layer before point assist.', 'warning');
        this._createSelection([{ kind: 'point', x: point.x, y: point.y, tolerance: 24 }]);
        return;
      }
      if (this._selectionMode === 'polygon') {
        this._polygonPoints.push([point.x, point.y]);
        this._drawPointPreview();
        return;
      }
      if (this._selectionMode === 'paint' || this._selectionMode === 'refine-paint') {
        this._drawing = { start: point, current: point, points: [[point.x, point.y]], tool: 'selection-paint', mode: this._selectionMode };
        this._drawTemporary();
        return;
      }
      if (this._selectionMode === 'box' || ['line', 'rectangle', 'ellipse', 'paint', 'erase', 'context-box'].includes(this._tool)) {
        if (this._tool === 'paint' || this._tool === 'erase') {
          if (!active) return this._setStatus('Select an unlocked paint layer first.', 'warning');
          if (active.kind !== 'paint' && active.kind !== 'vector') return this._setStatus('Paint strokes require a paint or vector layer.', 'warning');
        }
        this._drawing = { start: point, current: point, points: [[point.x, point.y]], tool: this._tool,
          mode: this._selectionMode || (this._tool === 'context-box' ? 'context-box' : null) };
        this._drawTemporary();
      }
    }

    _onPointerMove(event) {
      if (!this._doc || !this._stage) return;
      const point = this._pointerDocumentPoint();
      if (point) {
        const readout = this.querySelector('[data-role="coordinates"]');
        if (readout) readout.textContent = `x: ${point.x.toFixed(1)}, y: ${point.y.toFixed(1)} · DPR ${this._devicePixelRatio.toFixed(2)}`;
      }
      if (!this._drawing || !point) return;
      this._drawing.current = point;
      if (['paint', 'erase', 'selection-paint'].includes(this._drawing.tool)) this._drawing.points.push([point.x, point.y]);
      this._drawTemporary();
    }

    _onPointerUp() {
      if (!this._stage) return;
      this._panPointerActive = false;
      if (!this._spacePanning && !this._stage.isDragging()) this._stage.draggable(false);
      if (!this._drawing || !this._doc) return;
      const drawing = this._drawing;
      this._drawing = null;
      this._clearTemporary();
      if (drawing.mode === 'paint' || drawing.mode === 'refine-paint') {
        const seed = { kind: 'paint', geometry: { points: drawing.points, size: Number(this.querySelector('[data-action="brush-size"]').value) } };
        if (drawing.mode === 'paint') this._createSelection([seed]);
        else {
          const selection = this._activeSelection();
          const operation = this.querySelector('[data-action="refine-operation"]').value;
          if (!selection) return this._setStatus('Create or choose a persistent selection first.', 'warning');
          if (!['add', 'subtract', 'intersect'].includes(operation)) return this._setStatus('Paint refinement requires Add, Subtract, or Intersect.', 'warning');
          this._sendAction('refine_selection', { selectionId: selection.id, expectedSelectionRevision: selection.selectionRevision,
            operation, seed });
          this._selectionMode = null;
          this._refreshToolButtons();
        }
        return;
      }
      if (drawing.mode === 'box') {
        const left = Math.min(drawing.start.x, drawing.current.x);
        const top = Math.min(drawing.start.y, drawing.current.y);
        const width = Math.abs(drawing.current.x - drawing.start.x);
        const height = Math.abs(drawing.current.y - drawing.start.y);
        if (width < 1 || height < 1) return;
        this._createSelection([{ kind: 'box', geometry: { x: left, y: top, width, height } }]);
      } else if (drawing.mode === 'context-box') {
        const left = Math.min(drawing.start.x, drawing.current.x);
        const top = Math.min(drawing.start.y, drawing.current.y);
        const width = Math.abs(drawing.current.x - drawing.start.x);
        const height = Math.abs(drawing.current.y - drawing.start.y);
        if (width < 1 || height < 1) return;
        this._sendAction('create_context_mask', { seeds: [{ kind: 'box', geometry: { x: left, y: top, width, height } }] });
      } else if (drawing.tool === 'paint' || drawing.tool === 'erase') {
        if (drawing.points.length === 1) drawing.points.push(drawing.points[0]);
        const style = {
          color: this.querySelector('[data-action="color"]').value,
          width: Number(this.querySelector('[data-action="brush-size"]').value),
          opacity: Number(this.querySelector('[data-action="brush-opacity"]').value),
          hardness: Number(this.querySelector('[data-action="brush-hardness"]').value),
          mode: drawing.tool === 'erase' ? 'erase' : 'paint',
        };
        this._sendAction('create_object', { layerId: this._selectedLayerId, kind: 'paint-stroke', geometry: { points: drawing.points, closed: false }, style });
      } else {
        const x = drawing.start.x, y = drawing.start.y;
        const w = drawing.current.x - x, h = drawing.current.y - y;
        if (drawing.tool === 'line') this._createShape('line', { x, y, width: w, height: h });
        else if (drawing.tool === 'rectangle') this._createShape('rectangle', { x: Math.min(x, x + w), y: Math.min(y, y + h), width: Math.abs(w), height: Math.abs(h) });
        else if (drawing.tool === 'ellipse') this._createShape('ellipse', { x: Math.min(x, x + w), y: Math.min(y, y + h), width: Math.abs(w), height: Math.abs(h) });
      }
    }

    _drawTemporary() {
      if (!this._overlayGroup || !this._drawing) return;
      this._clearTemporary();
      const draw = this._drawing;
      const color = this.querySelector('[data-action="color"]').value;
      const common = { stroke: color, strokeWidth: Math.max(1, Number(this.querySelector('[data-action="brush-size"]').value)), listening: false, name: 'ncd-temp' };
      const hardness = Number(this.querySelector('[data-action="brush-hardness"]').value);
      if (hardness < 1) Object.assign(common, { shadowColor: color, shadowBlur: common.strokeWidth * (1 - hardness) * 0.25, shadowOpacity: 0.35 });
      if (draw.tool === 'line') {
        this._overlayGroup.add(new window.Konva.Line({ ...common, points: [draw.start.x, draw.start.y, draw.current.x, draw.current.y] }));
      } else if (draw.tool === 'rectangle') {
        this._overlayGroup.add(new window.Konva.Rect({ ...common, x: Math.min(draw.start.x, draw.current.x), y: Math.min(draw.start.y, draw.current.y), width: Math.abs(draw.current.x - draw.start.x), height: Math.abs(draw.current.y - draw.start.y), fill: 'rgba(255,255,255,0.05)' }));
      } else if (draw.tool === 'ellipse') {
        this._overlayGroup.add(new window.Konva.Ellipse({ ...common, x: (draw.start.x + draw.current.x) / 2, y: (draw.start.y + draw.current.y) / 2, radiusX: Math.abs(draw.current.x - draw.start.x) / 2, radiusY: Math.abs(draw.current.y - draw.start.y) / 2, fill: 'rgba(255,255,255,0.05)' }));
      } else if (draw.tool === 'paint' || draw.tool === 'erase' || draw.tool === 'selection-paint') {
        this._overlayGroup.add(new window.Konva.Line({ ...common, points: draw.points.flat(), lineCap: 'round', lineJoin: 'round', opacity: 0.8 }));
      } else if (draw.tool === 'context-box') {
        this._overlayGroup.add(new window.Konva.Rect({ ...common, x: Math.min(draw.start.x, draw.current.x), y: Math.min(draw.start.y, draw.current.y),
          width: Math.abs(draw.current.x - draw.start.x), height: Math.abs(draw.current.y - draw.start.y), fill: 'rgba(50,180,130,0.16)' }));
      }
      this._overlayLayer.batchDraw();
    }

    _drawPointPreview() {
      this._clearTemporary();
      if (this._polygonPoints.length < 2) return;
      const contextDraft = this._tool === 'context-polygon';
      const line = new window.Konva.Line({ points: this._polygonPoints.flat(), stroke: contextDraft ? '#43d6a4' : '#48b6ff', strokeWidth: 2,
        closed: contextDraft, fill: contextDraft ? 'rgba(50,180,130,0.16)' : undefined, listening: false, name: 'ncd-temp' });
      this._overlayGroup.add(line);
      this._overlayLayer.batchDraw();
    }

    _clearTemporary() {
      if (!this._overlayGroup) return;
      this._overlayGroup.find('.ncd-temp').forEach((node) => node.destroy());
      this._overlayLayer.batchDraw();
    }

    _finishPoints() {
      if (this._polygonPoints.length < 2) return;
      const points = this._polygonPoints.slice();
      this._polygonPoints = [];
      this._clearTemporary();
      if (this._tool === 'context-polygon') {
        if (points.length < 3) return this._setStatus('A context polygon needs at least three points.', 'warning');
        this._sendAction('create_context_mask', { seeds: [{ kind: 'polygon', geometry: { points, closed: true } }] });
        this._tool = 'select';
        this._refreshToolButtons();
        return;
      }
      if (this._selectionMode === 'polygon') {
        if (points.length < 3) return this._setStatus('A selection polygon needs at least three points.', 'warning');
        if (this._selectionSeedPurpose) {
          this._maskSeedPoints = points;
          this._selectionMode = null;
          this._selectionSeedPurpose = null;
          this._refreshToolButtons();
          this._setStatus('Polygon seed is ready for the selected refinement or rebase operation.');
          return;
        }
        this._createSelection([{ kind: 'polygon', geometry: { points } }]);
        this._selectionMode = null;
        this._refreshToolButtons();
        return;
      }
      if (this._tool === 'polygon') {
        if (points.length < 3) return this._setStatus('A polygon needs at least three points.', 'warning');
        this._createShape('polygon', { points });
      }
      else this._sendAction('create_object', {
        layerId: this._selectedLayerId,
        kind: 'path',
        geometry: { points, closed: false },
        style: { stroke: this.querySelector('[data-action="color"]').value, width: 3, opacity: 1 },
      });
    }

    _finishSelectionPolygon() {
      if (this._selectionMode === 'polygon') this._finishPoints();
    }

    _createShape(shape, geometry) {
      const style = { fill: this.querySelector('[data-action="color"]').value, stroke: this.querySelector('[data-action="color"]').value,
        strokeWidth: Math.max(1, Number(this.querySelector('[data-action="brush-size"]').value)), opacity: Number(this.querySelector('[data-action="brush-opacity"]').value) };
      const kind = shape === 'polygon' ? 'shape' : 'shape';
      this._sendAction('create_object', { layerId: this._selectedLayerId, kind, geometry: { shape, ...geometry }, style });
    }

    _isEditableLayer(layerId) {
      const layer = this._doc && this._doc.layers.find((item) => item.id === layerId);
      return !!layer && !layer.locked && !layer.deleted;
    }

    _selectedLayer() {
      return this._doc && this._doc.layers.find((item) => item.id === this._selectedLayerId);
    }

    _parentGroupId() {
      const select = this.querySelector('[data-action="parent-group"]');
      return select && select.value ? select.value : null;
    }

    _createSelection(seeds) {
      const layer = this._selectedLayer();
      if (!layer) return this._setStatus('Select a source raster layer before creating a selection.', 'warning');
      if (layer.kind !== 'raster') return this._setStatus('Persistent selections need a raster source layer.', 'warning');
      const semanticHint = this.querySelector('[data-action="selection-hint"]')?.value.trim() || '';
      this._sendAction('create_selection', { sourceLayerId: layer.id, seeds, semanticHint });
    }

    _activeSelection() {
      const selectionId = this._activeSelectionId || (this._doc && this._doc.selections.length ? this._doc.selections[this._doc.selections.length - 1].id : null);
      return this._doc && this._doc.selections.find((item) => item.id === selectionId);
    }

    _duplicateSelection() {
      const selection = this._activeSelection();
      if (!selection) return this._setStatus('Create or choose a persistent selection first.', 'warning');
      this._sendAction('duplicate_selection_to_layer', { selectionId: selection.id, expectedSelectionRevision: selection.selectionRevision,
        name: 'Selection copy' });
    }

    _refineSelection() {
      const selection = this._activeSelection();
      if (!selection) return this._setStatus('Create or choose a persistent selection first.', 'warning');
      const operation = this.querySelector('[data-action="refine-operation"]').value;
      const data = { selectionId: selection.id, expectedSelectionRevision: selection.selectionRevision, operation,
        radius: Number(this.querySelector('[data-action="mask-radius"]').value) };
      if (['add', 'subtract', 'intersect'].includes(operation)) {
        const points = this._maskSeedPoints.slice();
        if (points.length < 3) return this._setStatus(`Draw a polygon seed, then choose Finish selection before ${operation}.`, 'warning');
        data.seed = { kind: 'polygon', geometry: { points } };
        this._maskSeedPoints = [];
      } else if (operation === 'paint') {
        return this._setStatus('Use Paint refinement seed to draw an add, subtract, or intersect brush.', 'warning');
      }
      this._sendAction('refine_selection', data);
    }

    _rebaseSelection() {
      const selection = this._activeSelection();
      if (!selection || selection.state === 'current') return this._setStatus('Select a stale persistent selection to rebase.', 'warning');
      if (selection.staleReason === 'SOURCE_TRANSFORM_CHANGED') {
        this._sendAction('rebase_selection', { selectionId: selection.id, expectedSelectionRevision: selection.selectionRevision, geometryOnly: true });
        return;
      }
      const points = this._maskSeedPoints.slice();
      if (points.length < 3) return this._setStatus('Draw a new polygon mask and choose Finish selection before rebasing content changes.', 'warning');
      this._maskSeedPoints = [];
      this._sendAction('rebase_selection', { selectionId: selection.id, expectedSelectionRevision: selection.selectionRevision,
        geometryOnly: false, seeds: [{ kind: 'polygon', geometry: { points } }] });
    }

    _createGuide() {
      const points = this._polygonPoints.slice();
      const selected = this._selectedObjectIds.size === 1
        ? this._doc.objects.find((object) => object.id === Array.from(this._selectedObjectIds)[0]) : null;
      let kind = selected ? selected.kind : 'path';
      let geometry = selected ? selected.geometry : { points, closed: false };
      let style = selected ? selected.style : { stroke: this.querySelector('[data-action="color"]').value, width: 5, opacity: 0.8 };
      if (kind === 'paint-stroke') kind = 'path';
      if (!selected && points.length < 2) return this._setStatus('Draw a path or select a shape before creating a semantic guide.', 'warning');
      const guideName = window.prompt('Guide name:', 'Proposed guide') || 'Proposed guide';
      const semanticRole = window.prompt('Guide role (for example: focal subject, pose, depth anchor):', 'composition-anchor') || 'composition-anchor';
      this._polygonPoints = [];
      this._clearTemporary();
      this._sendAction('create_guide', { name: guideName, semanticRole, kind, geometry, style });
    }

    _markGuideReplacement() {
      const replacement = this._doc && this._doc.guides.find((guide) => guide.guideId === this._selectedGuideId && guide.lifecycle === 'proposed');
      if (!replacement) return this._setStatus('Select a proposed replacement guide in the guide list first.', 'warning');
      const oldId = window.prompt('ID of the existing guide being replaced:', '');
      const old = this._doc.guides.find((guide) => guide.guideId === oldId && guide.guideId !== replacement.guideId);
      if (!old) return this._setStatus('Enter an existing guide ID different from the replacement.', 'warning');
      this._sendAction('transition_guide', { guideId: old.guideId, lifecycle: 'replacement-pending', replacementId: replacement.guideId });
    }

    _transitionGuide(guideId, lifecycle) {
      if (!guideId) return;
      const map = { activate: 'active', consume: 'consumed', pending: 'replacement-pending', supersede: 'superseded', retire: 'safe-to-remove' };
      const next = map[lifecycle];
      if (!next) return;
      const guide = this._doc.guides.find((item) => item.guideId === guideId);
      let replacementId;
      if (next === 'replacement-pending') {
        replacementId = window.prompt('Replacement guide ID:');
        if (!replacementId) return;
      }
      if (next === 'superseded') replacementId = guide && guide.replacementId;
      this._sendAction('transition_guide', { guideId, lifecycle: next, replacementId });
    }

    _saveRelationalContext() {
      const maskSelect = this.querySelector('[data-action="context-mask"]');
      const editSelect = this.querySelector('[data-action="edit-mask"]');
      const contextMaskId = maskSelect && maskSelect.value;
      if (!contextMaskId) return this._setStatus('Create or choose a context mask first.', 'warning');
      const references = Array.from(this._contextReferenceIds);
      this._sendAction('set_relational_context', {
        contextMaskId,
        referenceIds: references,
        dilationPx: Number(this.querySelector('[data-action="context-dilation"]').value),
        editMaskId: editSelect && editSelect.value ? editSelect.value : null,
        editTargetIds: this._selectedObjectIds.size ? Array.from(this._selectedObjectIds) : (this._selectedLayerId ? [this._selectedLayerId] : []),
      });
    }

    _sampleColor() {
      try {
        const pointer = this._stage.getPointerPosition();
        const canvas = this._stage.toCanvas({ pixelRatio: 1 });
        const context = canvas.getContext('2d', { willReadFrequently: true });
        const sample = context.getImageData(Math.floor(pointer.x), Math.floor(pointer.y), 1, 1).data;
        const hex = `#${[sample[0], sample[1], sample[2]].map((value) => value.toString(16).padStart(2, '0')).join('')}`;
        this.querySelector('[data-action="color"]').value = hex;
        canvas.width = 0;
        canvas.height = 0;
      } catch (_) {
        this._setStatus('Could not sample the rendered pixel at this view.', 'warning');
      }
    }

    _transformSelected(rotationRadians, flip) {
      const layer = this._selectedLayer();
      const targetIds = this._selectedObjectIds.size ? Array.from(this._selectedObjectIds) : (layer ? [layer.id] : []);
      if (!targetIds.length) return;
      const actions = [];
      for (const targetId of targetIds) {
        const object = this._doc.objects.find((item) => item.id === targetId);
        const targetLayer = targetId === layer?.id ? layer : this._doc.layers.find((item) => item.id === object?.layerId);
        if (!targetLayer || targetLayer.effectiveLocked || targetLayer.deleted || this._actionInFlight || this._fidelityBlocked) continue;
        const current = targetId === layer?.id ? layer.transform : object?.transform;
        if (!current) continue;
        const transform = new window.Konva.Transform([current[0], current[3], current[1], current[4], current[2], current[5]]);
        const attrs = transform.decompose();
        const cx = this._doc.width / 2;
        const cy = this._doc.height / 2;
        const delta = new window.Konva.Transform();
        delta.translate(cx, cy);
        if (flip) delta.scale(-1, 1);
        else delta.rotate(rotationRadians);
        delta.translate(-cx, -cy);
        const composed = delta.copy().multiply(transform).getMatrix();
        const matrix = [composed[0], composed[2], composed[4], composed[1], composed[3], composed[5], 0, 0, 1];
        actions.push({ actionType: 'transform', targetIds: [targetId], data: { transform: matrix } });
      }
      if (actions.length) this._sendAction('batch', null, actions);
    }

    _toggleLayer(layerId, kind) {
      const layer = this._doc.layers.find((item) => item.id === layerId);
      if (!layer) return;
      if (kind === 'visible') this._sendAction('set_layer_visibility', { layerId, visible: !layer.visible });
      else this._sendAction('set_layer_lock', { layerId, locked: !layer.locked });
    }

    _moveSelectedLayer(direction) {
      if (!this._selectedLayerId) return;
      this._sendAction('reorder_layer', { layerId: this._selectedLayerId, direction });
    }

    _deleteSelectedLayer() {
      if (!this._selectedLayerId || !window.confirm('Delete this layer? Its history and assets remain recoverable.')) return;
      this._sendAction('delete_layer', { layerId: this._selectedLayerId, confirmed: true });
    }

    _runOnSelectedLayer(action, data) {
      if (!data.layerId) return this._setStatus('Select a layer first.', 'warning');
      this._sendAction(action, data);
    }

    async _createDocument() {
      if (!this._capability || !this._capability.enabled) return;
      const width = Number(window.prompt('Document width in pixels:', '2400'));
      const height = Number(window.prompt('Document height in pixels:', '1792'));
      if (!Number.isInteger(width) || !Number.isInteger(height)) return;
      try {
        const result = await this._request(`${API}/documents`, { method: 'POST', body: { width, height, name: 'Creative Document' } });
        this._doc = result.view;
        this._selectedLayerId = this._doc.rootLayerIds[0] || null;
        this._dirty = false;
        await this._acceptView(this._doc);
        this._setStatus(`Created ${this._doc.documentId} at revision ${this._doc.revision}.`);
      } catch (error) { this._showError(error); }
    }

    async _openDocument() {
      const id = this.querySelector('[data-action="document-id"]').value.trim();
      if (!id) return this._setStatus('Enter an opaque document ID to open.', 'warning');
      if (this._dirty && !window.confirm('Discard unsaved changes and reopen the committed revision?')) return;
      try {
        const result = await this._request(`${API}/documents/${encodeURIComponent(id)}/open`, { method: 'POST', body: { discardUnsaved: true } });
        this._doc = result.view;
        this._selectedLayerId = this._doc.rootLayerIds[0] || null;
        this._dirty = false;
        await this._acceptView(this._doc);
        this._setStatus(`Opened revision ${this._doc.revision}.`);
      } catch (error) { this._showError(error); }
    }

    async _enableAgentDriver() {
      if (!this._doc) return;
      const scopes = Array.from(this.querySelectorAll('[data-agent-scope]:checked'))
        .map((input) => input.dataset.agentScope);
      if (!scopes.length) return this._setAgentDriverStatus('Choose at least one scope.', true);
      try {
        const grant = await this._request(`${API}/documents/${encodeURIComponent(this._doc.documentId)}/agent-grants`, {
          method: 'POST', body: { scopes, lifetimeSeconds: 3600 },
        });
        const capability = grant && grant.capability;
        if (!capability || capability.schemaVersion !== 1 || capability.documentId !== this._doc.documentId
            || typeof capability.token !== 'string' || typeof capability.baseUrl !== 'string') {
          throw new Error('The local driver capability response is malformed.');
        }
        const contents = JSON.stringify(capability, null, 2);
        const url = URL.createObjectURL(new Blob([contents], { type: 'application/json' }));
        this._objectUrls.add(url);
        const link = document.createElement('a');
        link.href = url;
        link.download = `nexfocus-driver-${this._doc.documentId}.json`;
        link.click();
        this._agentGrantId = grant.grantId;
        this._agentGrantDocumentId = this._doc.documentId;
        this._setAgentDriverStatus(`Enabled for ${scopes.join(', ')}. Capability file downloaded.`);
      } catch (error) {
        this._setAgentDriverStatus(error.message || 'Could not enable the local driver.', true);
      }
    }

    async _refreshAgentDriverStatus() {
      if (!this._doc) return;
      try {
        const status = await this._request(`${API}/documents/${encodeURIComponent(this._doc.documentId)}/agent-grants/status`);
        this._agentGrantId = status.grantId || null;
        this._agentGrantDocumentId = this._doc.documentId;
        const message = status.status === 'enabled'
          ? `Enabled for ${(status.scopes || []).join(', ')} until ${new Date(status.expiresAt * 1000).toLocaleTimeString()}.`
          : 'Disabled';
        this._setAgentDriverStatus(message);
      } catch (error) {
        this._setAgentDriverStatus(error.message || 'Could not read driver status.', true);
      }
    }

    async _revokeAgentDriver() {
      if (!this._doc) return;
      try {
        if (this._agentGrantDocumentId !== this._doc.documentId || !this._agentGrantId) {
          const status = await this._request(`${API}/documents/${encodeURIComponent(this._doc.documentId)}/agent-grants/status`);
          this._agentGrantId = status.grantId || null;
          this._agentGrantDocumentId = this._doc.documentId;
        }
        if (!this._agentGrantId) return this._setAgentDriverStatus('Disabled');
        await this._request(`${API}/documents/${encodeURIComponent(this._doc.documentId)}/agent-grants/${encodeURIComponent(this._agentGrantId)}`, {
          method: 'DELETE',
        });
        this._agentGrantId = null;
        this._setAgentDriverStatus('Disabled; the capability has been revoked.');
      } catch (error) {
        this._setAgentDriverStatus(error.message || 'Could not revoke the local driver.', true);
      }
    }

    _setAgentDriverStatus(message, isError = false) {
      const target = this.querySelector('[data-role="agent-driver-status"]');
      if (target) {
        target.textContent = message;
        target.classList.toggle('ncd-warning', !!isError);
      }
    }

    async _saveDocument() {
      if (!this._doc) return;
      const commandId = `cmd-${crypto.randomUUID()}`;
      try {
        const result = await this._request(`${API}/documents/${encodeURIComponent(this._doc.documentId)}/save`, {
          method: 'POST', body: { expectedRevision: this._doc.revision, commandId },
        });
        this._dirty = false;
        this._doc.dirty = false;
        this._doc.committedRevision = result.committedRevision;
        this._doc.dirty = false;
        this._doc.recoveryNotice = 'Explicit save checkpoint is available.';
        this._viewSnapshots.delete(`${this._doc.documentId}:${this._doc.revision}`);
        const label = this.querySelector('[data-role="document-id-label"]');
        if (label) label.textContent = `${this._doc.documentId} · revision ${this._doc.revision} · saved`;
        this._setStatus(`Saved revision ${result.committedRevision}; recovery checkpoint is available.`);
        this._enableButtons(!!(this._capability && this._capability.enabled));
      } catch (error) { this._showError(error); }
    }

    async _exportComposite() {
      if (!this._doc) return;
      try {
        const blob = await this._apiBlob(`${API}/documents/${encodeURIComponent(this._doc.documentId)}/export`);
        const url = URL.createObjectURL(blob);
        this._objectUrls.add(url);
        const link = document.createElement('a');
        link.href = url;
        link.download = `creative-document-${this._doc.documentId}-r${this._doc.revision}.png`;
        link.click();
        this._setStatus(`Exported revision ${this._doc.revision}; the scene was not changed.`);
      } catch (error) { this._showError(error); }
    }

    async _previewComposite() {
      if (!this._doc) return;
      try {
        const blob = await this._apiBlob(`${API}/documents/${encodeURIComponent(this._doc.documentId)}/preview`);
        const url = URL.createObjectURL(blob);
        this._objectUrls.add(url);
        window.open(url, '_blank', 'noopener');
        this._setStatus(`Previewed revision ${this._doc.revision}; the scene was not changed.`);
      } catch (error) { this._showError(error); }
    }

    async _importImage(file) {
      if (!this._doc || !file) return;
      const form = new FormData();
      form.append('file', file, file.name);
      form.append('expected_revision', String(this._doc.revision));
      form.append('actor_kind', 'director');
      form.append('command_id', `cmd-${crypto.randomUUID()}`);
      form.append('transaction_id', `txn-${crypto.randomUUID()}`);
      form.append('group_id', `grp-${crypto.randomUUID()}`);
      const asGuide = window.confirm('Import this image as a proposed semantic guide?');
      if (asGuide) {
        form.append('as_guide', 'true');
        form.append('semantic_role', window.prompt('Guide role:', 'visual-reference') || 'visual-reference');
      }
      try {
        const receipt = await this._request(`${API}/documents/${encodeURIComponent(this._doc.documentId)}/assets/import`, { method: 'POST', body: form });
        this._dirty = !!receipt.dirty;
        await this._refreshView(receipt);
      } catch (error) { this._showError(error); }
      finally { this.querySelector('[data-action="file-input"]').value = ''; }
    }

    async _sendAction(actionType, data, actions) {
      if (!this._doc || this._actionInFlight || (this._fidelityBlocked && actionType !== 'set_layer_visibility')) return;
      this._actionInFlight = true;
      this._enableButtons(!!(this._capability && this._capability.enabled));
      this._refreshToolButtons();
      const startedAt = performance.now();
      const payload = {
        documentId: this._doc.documentId,
        expectedRevision: this._doc.revision,
        actorKind: 'director',
        commandId: `cmd-${crypto.randomUUID()}`,
        transactionId: `txn-${crypto.randomUUID()}`,
        groupId: `grp-${crypto.randomUUID()}`,
        actionType,
        targetIds: data && data.targetIds ? data.targetIds : [],
        ...(actionType === 'batch' ? { actions: actions || (data && data.actions) || [] } : { data: data || {} }),
      };
      try {
        const receipt = await this._request(`${API}/documents/${encodeURIComponent(this._doc.documentId)}/actions`, { method: 'POST', body: payload });
        this._dirty = !!receipt.dirty;
        this._report.interactionMs.push(performance.now() - startedAt);
        if (this._report.interactionMs.length > 60) this._report.interactionMs.shift();
        await this._refreshView(receipt);
        const newContextMask = (receipt.createdIds || []).find((id) => id.startsWith('mask-') && this._doc.masks.some((mask) => mask.id === id && mask.purpose === 'context'));
        if (newContextMask) { this._contextMaskId = newContextMask; this._renderProperties(); this._renderDocument(); }
      } catch (error) {
        try { await this._refreshAuthoritativeView(); } catch (_) { /* Preserve the refusal below. */ }
        if (error.code === 'STALE_DOCUMENT_REVISION') {
          this._setStatus(`STALE_DOCUMENT_REVISION: ${error.message}. The authoritative scene was restored; repeat the edit if needed.`, 'warning');
        } else this._showError(error);
      } finally {
        this._actionInFlight = false;
        this._enableButtons(!!(this._capability && this._capability.enabled));
        this._refreshToolButtons();
      }
    }

    async _refreshAuthoritativeView() {
      if (!this._doc) return;
      const documentId = this._doc.documentId;
      for (const key of Array.from(this._viewSnapshots.keys())) if (key.startsWith(`${documentId}:`)) this._viewSnapshots.delete(key);
      const view = await this._request(`${API}/documents/${encodeURIComponent(documentId)}/view`);
      this._doc = view;
      await this._acceptView(view);
    }

    async _refreshView(receipt) {
      const revision = receipt && receipt.currentRevision;
      if (revision !== undefined && this._doc) this._doc.revision = revision;
      const view = await this._snapshot(this._doc.documentId, revision);
      this._doc = view;
      this._dirty = !!view.dirty;
      const createdIds = receipt && Array.isArray(receipt.createdIds) ? receipt.createdIds : [];
      const createdLayer = createdIds.slice().reverse().find((id) => view.layers.some((layer) => layer.id === id));
      if (createdLayer) { this._selectedLayerId = createdLayer; this._selectedObjectIds.clear(); }
      const createdObject = createdIds.slice().reverse().find((id) => view.objects.some((object) => object.id === id));
      if (createdObject) {
        const object = view.objects.find((item) => item.id === createdObject);
        this._selectedLayerId = object.layerId;
        this._selectedObjectIds = new Set([object.id]);
      }
      const createdSelection = createdIds.slice().reverse().find((id) => view.selections.some((selection) => selection.id === id));
      if (createdSelection) this._activeSelectionId = createdSelection;
      if (this._selectedLayerId && !view.layers.some((item) => item.id === this._selectedLayerId)) this._selectedLayerId = view.rootLayerIds[0] || null;
      await this._acceptView(view);
    }

    async _snapshot(documentId, revision) {
      const key = `${documentId}:${revision === undefined ? 'latest' : revision}`;
      if (this._viewSnapshots.has(key)) return this._viewSnapshots.get(key);
      const promise = this._request(`${API}/documents/${encodeURIComponent(documentId)}/view`).then((view) => {
        const snapshotKey = `${documentId}:${view.revision}`;
        this._viewSnapshots.set(snapshotKey, Promise.resolve(view));
        return view;
      });
      this._viewSnapshots.set(key, promise);
      try { return await promise; }
      catch (error) { this._viewSnapshots.delete(key); throw error; }
    }

    async _acceptView(view) {
      if (!view || !view.documentId) throw new Error('Document view is malformed.');
      this._doc = view;
      this._dirty = !!view.dirty;
      this.querySelector('[data-role="document-id-label"]').textContent = `${view.documentId} · revision ${view.revision}${this._dirty ? ' · unsaved' : ' · saved'}`;
      const input = this.querySelector('[data-action="document-id"]');
      if (input) input.value = view.documentId;
      this._enableButtons(!!(this._capability && this._capability.enabled));
      if (!this._stage) this._createStage();
      this._renderLayerList();
      this._renderProperties();
      this._renderGuideList();
      this._renderSelectionList();
      this._renderDocument();
      if (view.recoveryNotice) this._setStatus(view.recoveryNotice, 'warning');
      else this._setStatus(`Revision ${view.revision}${this._dirty ? ' has unsaved edits.' : ' is committed.'}`);
    }

    async _request(url, options = {}) {
      if (!this._capability || !this._capability.enabled) throw new Error('Editor session is unavailable.');
      const headers = new Headers(options.headers || {});
      headers.set('Authorization', `Bearer ${this._capability.token}`);
      headers.set('X-Gradio-Session', this._capability.session);
      if (this._ownerKey) headers.set('X-Editor-Owner-Key', this._ownerKey);
      let body = options.body;
      if (body && !(body instanceof FormData) && typeof body !== 'string') {
        headers.set('Content-Type', 'application/json');
        body = JSON.stringify(body);
      }
      const response = await fetch(url, { method: options.method || 'GET', headers, body,
        credentials: 'same-origin', signal: options.signal || this._controller.signal });
      if (!response.ok) {
        const result = await response.json().catch(() => ({}));
        const detail = result.detail || result;
        if (response.status === 409 && detail.currentRevision !== undefined && this._doc) {
          this._viewSnapshots.delete(`${this._doc.documentId}:${detail.currentRevision}`);
        }
        const error = new Error(detail.message || detail.code || `Request failed (${response.status})`);
        error.code = detail.code;
        error.currentRevision = detail.currentRevision;
        error.refreshHint = detail.refreshHint;
        throw error;
      }
      const contentType = response.headers.get('content-type') || '';
      return contentType.includes('application/json') ? response.json() : response.blob();
    }

    async _apiBlob(url) {
      if (!this._capability || !this._capability.enabled) throw new Error('Editor session is unavailable.');
      const response = await fetch(url, {
        credentials: 'same-origin', signal: this._controller.signal,
        headers: { Authorization: 'Bearer ' + this._capability.token, 'X-Gradio-Session': this._capability.session,
          'X-Editor-Owner-Key': this._ownerKey || '' },
      });
      if (!response.ok) {
        const result = await response.json().catch(() => ({}));
        const detail = result.detail || result;
        const error = new Error(detail.message || detail.code || `Export failed (${response.status})`);
        error.code = detail.code;
        throw error;
      }
      return response.blob();
    }

    _showError(error) {
      const message = error && error.message ? error.message : String(error);
      this._setStatus(message, 'error');
      if (message.includes('expired') || message.includes('session')) this._setDisabled(message);
    }

    _assetRecord(assetId) {
      return this._doc && this._doc.assets.find((asset) => asset.id === assetId);
    }

    _revokeObjectUrls() {
      for (const url of this._objectUrls) URL.revokeObjectURL(url);
      this._objectUrls.clear();
    }

    async _loadImageAsset(asset, signal, onThumbnail) {
      let cache = this._assetCache.get(asset.contentHash);
      if (!cache) {
        cache = { thumbnailPromise: null, fullPromise: null, thumbnailUrl: null, fullUrl: null };
        this._assetCache.set(asset.contentHash, cache);
      }
      const loadUrl = async (url, expectedHash, kind) => {
        const blob = await this._scheduler.fetch(url, signal);
        if (signal && signal.aborted) throw new DOMException('Request cancelled', 'AbortError');
        if (kind === 'full') {
          const bytes = await blob.arrayBuffer();
          const digest = await crypto.subtle.digest('SHA-256', bytes);
          const actual = Array.from(new Uint8Array(digest)).map((value) => value.toString(16).padStart(2, '0')).join('').toUpperCase();
          if (actual !== expectedHash.toUpperCase()) throw new Error('Asset hash verification failed.');
        }
        const objectUrl = URL.createObjectURL(blob);
        this._objectUrls.add(objectUrl);
        const image = new Image();
        image.decoding = 'async';
        image.src = objectUrl;
        try { await image.decode(); }
        catch (error) { URL.revokeObjectURL(objectUrl); this._objectUrls.delete(objectUrl); throw error; }
        if (signal && signal.aborted) {
          URL.revokeObjectURL(objectUrl);
          this._objectUrls.delete(objectUrl);
          throw new DOMException('Request cancelled', 'AbortError');
        }
        return { image, url: objectUrl };
      };
      if (!cache.thumbnailPromise) {
        const pending = loadUrl(asset.thumbnailUrl, asset.contentHash, 'thumbnail').then((result) => {
          cache.thumbnailUrl = result.url;
          return result.image;
        });
        cache.thumbnailPromise = pending;
        pending.then(undefined, () => { if (cache.thumbnailPromise === pending) cache.thumbnailPromise = null; });
      }
      let thumbnail;
      try { thumbnail = await cache.thumbnailPromise; }
      catch (error) {
        if (error.name === 'AbortError' && !(signal && signal.aborted)) {
          return this._loadImageAsset(asset, signal, onThumbnail);
        }
        throw error;
      }
      if (onThumbnail) onThumbnail(thumbnail);
      if (!cache.fullPromise) {
        const pending = loadUrl(asset.url, asset.contentHash, 'full').then((result) => {
          cache.fullUrl = result.url;
          return result.image;
        });
        cache.fullPromise = pending;
        pending.then(undefined, () => { if (cache.fullPromise === pending) cache.fullPromise = null; });
      }
      try { return await cache.fullPromise; }
      catch (error) {
        if (error.name === 'AbortError' && !(signal && signal.aborted)) {
          return this._loadImageAsset(asset, signal, onThumbnail);
        }
        throw error;
      }
    }

    _renderDocument() {
      if (!this._doc || !window.Konva) return;
      this._rendererIssues = Array.isArray(this._doc.rendererIssues) ? this._doc.rendererIssues.slice() : [];
      const hardness = Number(this.querySelector('[data-action="brush-hardness"]')?.value || 1);
      if (hardness < 1 && !this._rendererIssues.some((issue) => issue.code === 'PAINT_HARDNESS_PREVIEW_APPROXIMATION')) {
        this._rendererIssues.push({ code: 'PAINT_HARDNESS_PREVIEW_APPROXIMATION', severity: 'warning',
          message: 'Soft brush edges use an approximate canvas halo; preview and export edges may differ.' });
      }
      this._fidelityBlocked = this._rendererIssues.some((issue) => issue.severity === 'error');
      this._renderFidelityIssues();
      this._enableButtons(!!(this._capability && this._capability.enabled));
      this._refreshToolButtons();
      if (this._renderController) this._renderController.abort();
      this._renderController = new AbortController();
      if (!this._stage) this._createStage();
      if (!this._mainGroup || !this._overlayGroup) return;
      this._mainGroup.destroyChildren();
      this._overlayGroup.destroyChildren();
      this._activeNodeRecords.clear();
      this._transformer.nodes([]);
      const layerMap = new Map(this._doc.layers.map((layer) => [layer.id, layer]));
      const objectMap = new Map(this._doc.objects.map((object) => [object.id, object]));
      const assetMap = new Map(this._doc.assets.map((asset) => [asset.id, asset]));
      const addLayer = (layerId, parent) => {
        const layer = layerMap.get(layerId);
        if (!layer) return;
        const group = new window.Konva.Group({
          name: `ncd-layer-${layer.id}`,
          id: `layer-${layer.id}`,
          visible: layer.visible && !layer.deleted,
          opacity: layer.opacity,
          listening: true,
        });
        group.setAttrs(matrixToAttrs(window.Konva, layer.transform));
        group.on('click.creative tap.creative', (event) => {
          event.cancelBubble = true;
          this._selectedLayerId = layer.id;
          this._renderLayerList();
          this._renderProperties();
        });
        parent.add(group);
        if (layer.kind === 'group') {
          for (const child of layer.childIds) addLayer(child, group);
        }
        for (const objectId of layer.objectIds) {
          const object = objectMap.get(objectId);
          if (!object) continue;
          const wrapper = new window.Konva.Group({
            name: `ncd-object-${object.id}`,
            id: `object-${object.id}`,
            draggable: this._tool === 'select' && !layer.effectiveLocked && !layer.deleted && !this._actionInFlight && !this._fidelityBlocked,
            listening: !layer.effectiveLocked && !layer.deleted && !this._actionInFlight && !this._fidelityBlocked,
          });
          wrapper.setAttrs(matrixToAttrs(window.Konva, object.transform));
          wrapper.on('click.creative tap.creative', (event) => {
            event.cancelBubble = true;
            if (event.evt && event.evt.shiftKey) {
              if (this._selectedObjectIds.has(object.id)) this._selectedObjectIds.delete(object.id);
              else this._selectedObjectIds.add(object.id);
            } else this._selectedObjectIds = new Set([object.id]);
            this._selectedLayerId = layer.id;
            this._updateTransformer();
            this._renderLayerList();
            this._renderProperties();
          });
          wrapper.on('dragstart.creative', () => {
            const ids = new Set([...this._selectedObjectIds, object.id]);
            this._gestureStartPositions = new Map(Array.from(ids).map((id) => {
              const record = this._activeNodeRecords.get(id);
              return record ? [id, record.node.absolutePosition()] : null;
            }).filter(Boolean));
          });
          wrapper.on('dragmove.creative', () => {
            const origin = this._gestureStartPositions && this._gestureStartPositions.get(object.id);
            if (!origin) return;
            const current = wrapper.absolutePosition();
            const delta = { x: current.x - origin.x, y: current.y - origin.y };
            for (const [id, start] of this._gestureStartPositions) {
              if (id === object.id) continue;
              const record = this._activeNodeRecords.get(id);
              if (record && !record.locked) record.node.absolutePosition({ x: start.x + delta.x, y: start.y + delta.y });
            }
          });
          wrapper.on('dragend.creative', () => {
            this._gestureStartPositions = null;
            this._scheduleTransformCommit(object.id);
          });
          group.add(wrapper);
          this._activeNodeRecords.set(object.id, { node: wrapper, layerId: layer.id, objectId: object.id,
            locked: layer.effectiveLocked || layer.deleted || this._fidelityBlocked });
          if (object.kind === 'raster-placement' && object.assetId) {
            const asset = assetMap.get(object.assetId);
            const box = object.geometry || {};
            const raster = new window.Konva.Image({ x: Number(box.x || 0), y: Number(box.y || 0),
              width: Number(box.width || asset?.width || this._doc.width), height: Number(box.height || asset?.height || this._doc.height),
              listening: false, name: 'ncd-raster-image' });
            wrapper.add(raster);
            const hit = new window.Konva.Rect({ x: raster.x(), y: raster.y(), width: raster.width(), height: raster.height(),
              fill: 'rgba(0,0,0,0.001)', listening: !layer.effectiveLocked && !layer.deleted && !this._actionInFlight && !this._fidelityBlocked, name: 'ncd-raster-hit' });
            wrapper.add(hit);
            if (asset && !asset.external) {
              this._loadImageAsset(asset, this._renderController.signal, (thumb) => {
                if (this._connected && wrapper.getStage()) { raster.image(thumb); this._mainLayer.batchDraw(); }
              }).then((image) => {
                if (this._connected && wrapper.getStage()) { raster.image(image); this._mainLayer.batchDraw(); }
              }).catch((error) => {
                if (error.name !== 'AbortError') this._setRendererIssue({ code: error.code || 'ASSET_UNAVAILABLE', severity: 'error',
                  layerId: layer.id, objectId: object.id, message: 'Raster image could not be loaded and verified.' });
              });
            }
          } else {
            this._appendVectorNode(wrapper, object);
          }
        }
      };
      const background = new window.Konva.Rect({ x: 0, y: 0, width: this._doc.width, height: this._doc.height,
        fill: '#2c3036', listening: false, name: 'ncd-background' });
      this._mainGroup.add(background);
      for (const layerId of this._doc.rootLayerIds) addLayer(layerId, this._mainGroup);
      this._renderSelectionOverlay();
      this._renderContextOverlay();
      this._applyViewport();
      this._mainLayer.batchDraw();
      this._overlayLayer.batchDraw();
      this._updateTransformer();
      this._setStatusMetric();
    }

    _appendVectorNode(wrapper, object) {
      const geometry = object.geometry || {};
      const style = object.style || {};
      const color = style.color || style.stroke || '#e7b45a';
      const width = Math.max(1, Number(style.width || style.strokeWidth || 2));
      const hardness = Math.max(0, Math.min(1, Number(style.hardness === undefined ? 1 : style.hardness)));
      const attrs = { stroke: style.stroke || color, strokeWidth: width, opacity: Number(style.opacity === undefined ? 1 : style.opacity),
        lineCap: 'round', lineJoin: 'round', shadowColor: color,
        shadowBlur: hardness < 1 ? Math.max(1, width * (1 - hardness) * 0.25) : 0,
        shadowOpacity: hardness < 1 ? 0.35 : 0, shadowForStrokeEnabled: true };
      if (object.kind === 'paint-stroke' || object.kind === 'path' || object.kind === 'guide') {
        const points = Array.isArray(geometry.points) ? geometry.points.flat().map(Number) : [];
        if (points.length < 2) return;
        const node = new window.Konva.Line({ ...attrs, points, closed: !!geometry.closed, fill: style.fill || undefined,
          globalCompositeOperation: style.mode === 'erase' ? 'destination-out' : 'source-over',
          listening: true, hitStrokeWidth: Math.max(width, 8) });
        wrapper.add(node);
      } else if (object.kind === 'shape') {
        const shape = geometry.shape;
        if (shape === 'rectangle') wrapper.add(new window.Konva.Rect({ ...attrs, x: Number(geometry.x), y: Number(geometry.y), width: Number(geometry.width), height: Number(geometry.height), fill: style.fill || undefined }));
        else if (shape === 'ellipse') wrapper.add(new window.Konva.Ellipse({ ...attrs, x: Number(geometry.x) + Number(geometry.width) / 2,
          y: Number(geometry.y) + Number(geometry.height) / 2, radiusX: Number(geometry.width) / 2, radiusY: Number(geometry.height) / 2, fill: style.fill || undefined }));
        else if (shape === 'line') wrapper.add(new window.Konva.Line({ ...attrs, points: [Number(geometry.x), Number(geometry.y), Number(geometry.x) + Number(geometry.width), Number(geometry.y) + Number(geometry.height)] }));
        else if (shape === 'polygon') wrapper.add(new window.Konva.Line({ ...attrs, points: (geometry.points || []).flat(), closed: true, fill: style.fill || undefined }));
        else this._setRendererIssue({ code: 'UNSUPPORTED_OBJECT_SHAPE', severity: 'error', layerId: object.layerId,
          objectId: object.id, message: `Shape '${shape || 'unspecified'}' is not rendered in the editor.` });
      } else {
        this._setRendererIssue({ code: 'UNSUPPORTED_OBJECT_KIND', severity: 'error', layerId: object.layerId, objectId: object.id,
          message: `Object kind '${object.kind}' is not rendered in the editor.` });
      }
    }

    _renderFidelityIssues() {
      const shell = this.querySelector('.ncd-shell');
      if (!shell) return;
      let banner = shell.querySelector('[data-role="fidelity-issues"]');
      if (!this._rendererIssues.length) {
        if (banner) banner.remove();
        return;
      }
      if (!banner) {
        banner = document.createElement('section');
        banner.className = 'ncd-fidelity-issues';
        banner.dataset.role = 'fidelity-issues';
        shell.insertBefore(banner, shell.querySelector('.ncd-body'));
      }
      const blocked = this._rendererIssues.some((issue) => issue.severity === 'error');
      banner.innerHTML = '';
      const heading = document.createElement('strong');
      heading.textContent = blocked ? 'Renderer fidelity blocked editing' : 'Renderer fidelity warning';
      banner.appendChild(heading);
      const list = document.createElement('ul');
      for (const issue of this._rendererIssues) {
        const item = document.createElement('li');
        item.dataset.code = issue.code;
        item.dataset.layerId = issue.layerId || '';
        item.dataset.objectId = issue.objectId || '';
        item.textContent = `${issue.code}: ${issue.message}`;
        list.appendChild(item);
      }
      banner.appendChild(list);
      this._enableButtons(!!(this._capability && this._capability.enabled));
      this._refreshToolButtons();
    }

    _setRendererIssue(issue) {
      if (this._rendererIssues.some((current) => current.code === issue.code && current.objectId === issue.objectId && current.layerId === issue.layerId)) return;
      this._rendererIssues.push(issue);
      this._fidelityBlocked = this._rendererIssues.some((current) => current.severity === 'error');
      this._renderFidelityIssues();
    }

    _updateHardnessWarning(hardness) {
      this._rendererIssues = this._rendererIssues.filter((issue) => !(issue.code === 'PAINT_HARDNESS_PREVIEW_APPROXIMATION' && !issue.objectId));
      if (hardness < 1 && !this._rendererIssues.some((issue) => issue.code === 'PAINT_HARDNESS_PREVIEW_APPROXIMATION')) {
        this._rendererIssues.push({ code: 'PAINT_HARDNESS_PREVIEW_APPROXIMATION', severity: 'warning',
          message: 'Soft brush edges use an approximate canvas halo; preview and export edges may differ.' });
      }
      this._renderFidelityIssues();
    }

    async _tintMask(image, rgb) {
      const canvas = document.createElement('canvas');
      canvas.width = image.naturalWidth || image.width;
      canvas.height = image.naturalHeight || image.height;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(image, 0, 0);
      const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height);
      for (let i = 0; i < pixels.data.length; i += 4) {
        const luminance = Math.round((pixels.data[i] * 0.2126) + (pixels.data[i + 1] * 0.7152) + (pixels.data[i + 2] * 0.0722));
        const alpha = Math.round(pixels.data[i + 3] * luminance / 255);
        pixels.data[i] = rgb[0];
        pixels.data[i + 1] = rgb[1];
        pixels.data[i + 2] = rgb[2];
        pixels.data[i + 3] = alpha;
      }
      ctx.putImageData(pixels, 0, 0);
      const result = new Image();
      result.src = canvas.toDataURL('image/png');
      await result.decode();
      canvas.width = 0;
      canvas.height = 0;
      return result;
    }

    _renderContextOverlay() {
      const maskId = this._contextMaskId || (this._doc.masks.slice().reverse().find((mask) => mask.purpose === 'context') || {}).id;
      const mask = this._doc.masks.find((item) => item.id === maskId && item.purpose === 'context');
      const asset = mask && this._assetRecord(mask.assetId);
      if (!asset || asset.external || !this._overlayGroup) return;
      this._loadImageAsset(asset, this._renderController.signal).then((source) => this._tintMask(source, [54, 196, 142])).then((tinted) => {
        if (!this._connected || !this._overlayGroup) return;
        this._overlayGroup.add(new window.Konva.Image({ image: tinted, x: 0, y: 0, width: this._doc.width, height: this._doc.height,
          opacity: 0.15, listening: false, name: 'ncd-context-mask' }));
        this._overlayLayer.batchDraw();
      }).catch((error) => { if (error.name !== 'AbortError') this._setStatus('Context mask preview is unavailable.', 'warning'); });
    }

    _scheduleTransformCommit(objectId = null) {
      if (this._actionInFlight || !this._doc) return;
      if (objectId) this._gestureObjectIds.add(objectId);
      if (this._gestureFrame) return;
      this._gestureFrame = window.requestAnimationFrame(() => {
        this._gestureFrame = 0;
        const targetIds = new Set([...this._selectedObjectIds, ...this._gestureObjectIds]);
        this._gestureObjectIds.clear();
        const actions = Array.from(targetIds).map((targetId) => {
          const record = this._activeNodeRecords.get(targetId);
          if (!record || record.locked || !record.node.getStage()) return null;
          return {
            actionType: 'transform',
            targetIds: [targetId],
            data: { targetIds: [targetId], transform: attrsToMatrix(window.Konva, record.node) },
          };
        }).filter(Boolean);
        if (actions.length === 1) this._sendAction('transform', actions[0].data);
        else if (actions.length > 1) this._sendAction('batch', null, actions);
      });
    }

    _updateTransformer() {
      if (!this._transformer) return;
      const nodes = Array.from(this._selectedObjectIds).map((id) => this._activeNodeRecords.get(id))
        .filter((record) => record && !record.locked && !this._actionInFlight && !this._fidelityBlocked)
        .map((record) => record.node);
      this._transformer.nodes(nodes);
      this._transformer.off('transformend.creative');
      this._transformer.on('transformend.creative', () => this._scheduleTransformCommit());
      this._overlayLayer.batchDraw();
    }

    _flattenLayers() {
      const map = new Map(this._doc.layers.map((layer) => [layer.id, layer]));
      const output = [];
      const visit = (id, depth) => {
        const layer = map.get(id);
        if (!layer) return;
        output.push({ layer, depth });
        for (const child of layer.childIds) visit(child, depth + 1);
      };
      for (const id of this._doc.rootLayerIds) visit(id, 0);
      return output;
    }

    _renderLayerList() {
      if (!this._doc) return;
      const list = this.querySelector('[data-role="layers"]');
      if (!list) return;
      list.innerHTML = '';
      for (const { layer, depth } of this._flattenLayers()) {
        const row = document.createElement('div');
        const layerIssues = layer.rendererIssues || [];
        row.className = `ncd-layer-row${layer.id === this._selectedLayerId ? ' selected' : ''}${layer.stale ? ' stale' : ''}${layer.deleted ? ' deleted' : ''}${layerIssues.some((issue) => issue.severity === 'error') ? ' error' : ''}`;
        row.dataset.layerId = layer.id;
        row.style.paddingLeft = `${4 + depth * 14}px`;
        const visibility = makeButton(layer.visible ? '●' : '○', 'layer-eye', 'Show or hide layer');
        visibility.dataset.layerId = layer.id;
        visibility.disabled = layer.effectiveLocked || layer.deleted || this._actionInFlight;
        const summary = document.createElement('div');
        summary.className = 'ncd-layer-summary';
        const name = document.createElement('span');
        name.className = 'ncd-layer-name';
        name.textContent = layer.name;
        const state = document.createElement('span');
        state.className = 'ncd-layer-state';
        const status = [layer.kind, layer.role, `r${layer.revision}`, layer.workStatus];
        if (layer.depthElement) status.push('depth element');
        if (layer.completePlateId) status.push(`complete plate: ${this._layerName(layer.completePlateId)}`);
        if (layer.stale) status.push('stale selection source');
        if (layer.deleted) status.push('deleted');
        if (layer.effectiveLocked) status.push(layer.lockedByAncestor ? 'locked by parent' : 'locked');
        state.textContent = status.filter(Boolean).join(' · ');
        summary.append(name, state);
        for (const issue of layerIssues) {
          const error = document.createElement('span');
          error.className = `ncd-layer-issue ${issue.severity}`;
          error.textContent = `${issue.code}: ${issue.message}`;
          summary.appendChild(error);
        }
        const lock = makeButton(layer.locked ? '🔒' : '🔓', 'layer-lock', 'Lock or unlock layer');
        lock.dataset.layerId = layer.id;
        lock.disabled = layer.lockedByAncestor || this._actionInFlight;
        const ref = document.createElement('input');
        ref.type = 'checkbox';
        ref.className = 'ncd-context-ref';
        ref.dataset.layerId = layer.id;
        ref.title = 'Use as a relational context reference';
        ref.checked = this._contextReferenceIds.has(layer.id);
        row.append(visibility, summary, lock, ref);
        row.addEventListener('dblclick', () => {
          const updated = window.prompt('Rename layer:', layer.name);
          if (updated !== null) this._sendAction('rename_layer', { layerId: layer.id, name: updated });
        }, { signal: this._controller.signal });
        list.appendChild(row);
      }
    }

    _layerName(identity) {
      const layer = this._doc.layers.find((item) => item.id === identity);
      return layer ? layer.name : String(identity || '').slice(0, 16) || 'unknown layer';
    }

    _renderProperties() {
      if (!this._doc) return;
      const layer = this._selectedLayer();
      const opacity = this.querySelector('[data-action="layer-opacity"]');
      const name = this.querySelector('[data-action="layer-name"]');
      if (opacity) { opacity.value = layer ? String(layer.opacity) : '1'; opacity.disabled = !layer || layer.effectiveLocked || this._actionInFlight || this._fidelityBlocked; }
      if (name) { name.value = layer ? layer.name : ''; name.disabled = !layer || layer.effectiveLocked || this._actionInFlight || this._fidelityBlocked; }
      const selectionMask = this.querySelector('[data-action="selection-mask"]');
      if (selectionMask) {
        selectionMask.innerHTML = '<option value="">Choose selection, layer alpha, or visibility mask</option>';
        for (const mask of this._doc.masks.filter((item) => ['editing-selection', 'layer-alpha', 'visibility'].includes(item.purpose))) {
          const owner = mask.ownerId ? ` · ${this._layerName(mask.ownerId)}` : '';
          const option = document.createElement('option'); option.value = mask.id;
          option.textContent = `${mask.purpose}${owner} · r${mask.revision}`;
          selectionMask.appendChild(option);
        }
      }
      const context = this.querySelector('[data-action="context-mask"]');
      const edit = this.querySelector('[data-action="edit-mask"]');
      if (context) {
        context.innerHTML = '<option value="">Choose context mask</option>';
        for (const mask of this._doc.masks.filter((item) => item.purpose === 'context')) {
          const option = document.createElement('option'); option.value = mask.id; option.textContent = mask.id;
          context.appendChild(option);
        }
        if (!this._contextMaskId || !this._doc.masks.some((item) => item.id === this._contextMaskId && item.purpose === 'context')) {
          this._contextMaskId = this._doc.masks.slice().reverse().find((item) => item.purpose === 'context')?.id || null;
        }
        context.value = this._contextMaskId || '';
      }
      if (edit) {
        edit.innerHTML = '<option value="">No edit mask linked</option>';
        for (const mask of this._doc.masks.filter((item) => item.purpose === 'generation')) {
          const option = document.createElement('option'); option.value = mask.id; option.textContent = mask.id;
          edit.appendChild(option);
        }
        if (!this._editMaskId || !this._doc.masks.some((item) => item.id === this._editMaskId && item.purpose === 'generation')) this._editMaskId = null;
        edit.value = this._editMaskId || '';
      }
      const parent = this.querySelector('[data-action="parent-group"]');
      if (parent) {
        const selectedParent = parent.value;
        parent.innerHTML = '<option value="">Root level</option>';
        for (const layer of this._doc.layers.filter((item) => item.kind === 'group' && !item.deleted)) {
          const option = document.createElement('option'); option.value = layer.id; option.textContent = layer.name;
          parent.appendChild(option);
        }
        if (this._selectedLayerId && parent.querySelector(`option[value="${CSS.escape(this._selectedLayerId)}"]`)) parent.value = this._selectedLayerId;
        else if (selectedParent && parent.querySelector(`option[value="${CSS.escape(selectedParent)}"]`)) parent.value = selectedParent;
        else parent.value = '';
      }
      this._renderRelations();
    }

    _renderRelations() {
      const container = this.querySelector('[data-role="relations"]');
      if (!container) return;
      container.innerHTML = '';
      if (!this._doc.interactionGroups.length && !this._doc.variants.length) {
        const empty = document.createElement('div');
        empty.className = 'ncd-small';
        empty.textContent = 'No interaction groups or variants are recorded.';
        container.appendChild(empty);
        return;
      }
      for (const group of this._doc.interactionGroups) {
        const row = document.createElement('div');
        row.className = 'ncd-row';
        const relationNames = group.relationOrder.map((item) => {
          const id = item.layerId || item.memberLayerId || item.targetLayerId;
          const relation = item.relation || item.position || item.order || 'member';
          return `${this._layerName(id)} (${relation})`;
        });
        row.textContent = `Interaction order: ${relationNames.join(' → ')}`;
        row.title = `${group.groupId} · ${group.overlapNotes || ''}`;
        container.appendChild(row);
      }
      for (const variant of this._doc.variants) {
        const row = document.createElement('div');
        row.className = 'ncd-row';
        const members = variant.memberIds.map((id) => `${this._layerName(id)}${id === variant.activeMemberId ? ' (active)' : ' (inactive)'}`);
        row.textContent = `${variant.semanticRole}: ${members.join(' / ')}`;
        row.title = `${variant.variantSetId} · ${JSON.stringify(variant.sharedAnchor)}`;
        container.appendChild(row);
      }
    }

    _renderGuideList() {
      if (!this._doc) return;
      const container = this.querySelector('[data-role="guides"]');
      container.innerHTML = '';
      for (const guide of this._doc.guides) {
        const row = document.createElement('div');
        row.className = 'ncd-row';
        const label = document.createElement('span');
        label.textContent = `${guide.name} · ${guide.lifecycle} · r${guide.stateRevision}`;
        label.title = guide.semanticRole || guide.guideId;
        label.addEventListener('click', () => { this._selectedGuideId = guide.guideId; label.style.color = '#8dc5ff'; });
        row.appendChild(label);
        const lifecycleActions = guide.lifecycle === 'proposed' ? [['Activate', 'activate']] : [];
        if (guide.lifecycle === 'active') lifecycleActions.push(['Consumed', 'consume']);
        if (guide.lifecycle === 'replacement-pending') lifecycleActions.push(['Supersede', 'supersede']);
        if (guide.lifecycle !== 'safe-to-remove') lifecycleActions.push(['Retire', 'retire']);
        for (const [caption, lifecycle] of lifecycleActions) {
          const button = makeButton(caption, `guide-${lifecycle}`);
          button.dataset.guideId = guide.guideId;
          row.appendChild(button);
        }
        container.appendChild(row);
      }
    }

    _renderSelectionList() {
      if (!this._doc) return;
      const container = this.querySelector('[data-role="selections"]');
      container.innerHTML = '';
      for (const selection of this._doc.selections) {
        const row = document.createElement('div');
        row.className = `ncd-row ncd-selection-record${selection.id === this._activeSelectionId ? ' selected' : ''}`;
        const label = document.createElement('span');
        const source = this._doc.layers.find((layer) => layer.id === selection.sourceId);
        const bounds = selection.bounds || {};
        const box = `x${bounds.x1 ?? '?'} y${bounds.y1 ?? '?'} w${Number.isInteger(bounds.x1) && Number.isInteger(bounds.x2) ? bounds.x2 - bounds.x1 : '?'} h${Number.isInteger(bounds.y1) && Number.isInteger(bounds.y2) ? bounds.y2 - bounds.y1 : '?'}`;
        const lastRebase = selection.rebaseHistory && selection.rebaseHistory.length ? selection.rebaseHistory[selection.rebaseHistory.length - 1] : null;
        const rebaseText = lastRebase ? (lastRebase.geometryOnly ? 'geometry-only rebase; mask preserved' : 'content rebase; mask rederived') : 'not rebased';
        label.textContent = `${selection.id} · ${selection.state}${selection.staleReason ? ` · ${selection.staleReason}` : ''} · selection r${selection.selectionRevision}`;
        const details = document.createElement('span');
        details.className = 'ncd-selection-details';
        details.textContent = `Source: ${source ? source.name : selection.sourceId} (${selection.sourceId}) · source r${selection.sourceRevision} · digest ${selection.sourceContentDigest} · ${box} · ${selection.refinementHistory.length} refinements · ${rebaseText} · derived: ${selection.derivedLayerIds.map((id) => this._layerName(id)).join(', ') || 'none'}`;
        row.append(label, details);
        for (const item of (selection.refinementHistory || [])) {
          const entry = document.createElement('span'); entry.className = 'ncd-selection-details';
          entry.textContent = `Refinement r${item.selectionRevision}: ${item.details.operation || item.details.kind || 'mask edit'} · mask ${item.newMaskId}`;
          row.appendChild(entry);
        }
        if (lastRebase) {
          const entry = document.createElement('span'); entry.className = 'ncd-selection-details';
          entry.textContent = `Rebase r${lastRebase.selectionRevision}: ${lastRebase.reason} · source r${lastRebase.previousSourceRevision} → r${lastRebase.sourceRevision} · ${rebaseText}`;
          row.appendChild(entry);
        }
        const button = makeButton('Use', 'selection-activate');
        button.dataset.selectionId = selection.id;
        row.appendChild(button);
        row.addEventListener('click', () => {
          this._activeSelectionId = selection.id;
          this._renderSelectionList();
          this._renderDocument();
        });
        container.appendChild(row);
      }
    }

    _renderSelectionOverlay() {
      const selection = this._activeSelection();
      if (!selection || !this._overlayGroup) return;
      const mask = this._doc.masks.find((item) => item.id === selection.maskId);
      const asset = mask && this._assetRecord(mask.assetId);
      if (!asset || asset.external) return;
      this._loadImageAsset(asset, this._renderController.signal).then((source) => {
        if (!this._connected || !this._overlayGroup) return;
        this._tintMask(source, [30, 150, 255]).then((tinted) => {
          if (!this._connected || !this._overlayGroup) return;
          this._overlayGroup.add(new window.Konva.Image({ image: tinted, x: 0, y: 0, width: this._doc.width, height: this._doc.height,
            opacity: selection.state === 'current' ? 0.3 : 0.14, listening: false, name: 'ncd-selection-mask' }));
          this._overlayLayer.batchDraw();
        });
      }).catch((error) => { if (error.name !== 'AbortError') this._setStatus('Selection mask preview is unavailable.', 'warning'); });
    }

    async _commitDocAction(actionType, data, actions) {
      const requestData = actions ? { actions } : data;
      return this._sendAction(actionType, requestData);
    }
  }

  if (!customElements.get('creative-document-editor')) {
    customElements.define('creative-document-editor', CreativeDocumentEditor);
  }
})();
