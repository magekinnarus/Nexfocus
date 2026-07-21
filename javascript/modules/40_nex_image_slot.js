(() => {
    function appRoot() {
        if (typeof gradioApp === 'function') {
            return gradioApp();
        }
        return document;
    }

    function escapeSelector(value) {
        if (window.CSS && typeof window.CSS.escape === 'function') {
            return window.CSS.escape(value);
        }
        return String(value).replace(/([^a-zA-Z0-9_-])/g, '\\$1');
    }

    class NexImageSlot extends HTMLElement {
        connectedCallback() {
            if (this.dataset.nexReady === 'true') {
                return;
            }

            this.dataset.nexReady = 'true';
            this.bridgeId = this.dataset.bridgeId || '';
            this.label = this.dataset.label || 'Image';
            this.placeholder = this.dataset.placeholder || 'Click or drop an image here';
            this.uploadMode = this.dataset.uploadMode || 'bridge';
            this.pathFieldId = this.dataset.pathFieldId || '';
            this.workspaceFieldId = this.dataset.workspaceFieldId || '';
            this.toolGroup = this.dataset.toolGroup || '';
            this.preserveMetadata = this.dataset.preserveMetadata === 'true';
            this.bridgeRoot = null;
            this.bridgeObserver = null;
            this.appObserver = null;
            this.appSyncScheduled = false;
            this.pathField = null;
            this.workspaceField = null;
            this.apiFieldObserver = null;
            this.apiObservedPathElement = null;
            this.apiObservedWorkspaceElement = null;
            this.lastApiStateKey = '';
            this.apiUploadInFlight = false;
            this.serverResyncHandle = null;
            this.serverResyncToken = 0;
            this.onApiFieldChange = () => this.syncFromApiFields(true);
            this.onApiFieldMutation = () => this.syncFromApiFields(false);
            this.onServerSyncRequest = (event) => this.handleServerSyncRequest(event);
            this.objectUrl = null;
            this.render();
            this.bindEvents();
            this.markUploadState(false);
            this.observeApp();
            window.addEventListener('nex-slot:server-sync', this.onServerSyncRequest);
            if (this.uploadMode === 'api') {
                this.attachApiFieldListeners();
                this.attachApiFieldObservers();
                this.syncFromApiFields(false);
            } else {
                this.attachBridge();
            }
        }

        disconnectedCallback() {
            if (this.bridgeObserver) {
                this.bridgeObserver.disconnect();
                this.bridgeObserver = null;
            }
            if (this.appObserver) {
                this.appObserver.disconnect();
                this.appObserver = null;
            }
            window.removeEventListener('nex-slot:server-sync', this.onServerSyncRequest);
            this.clearServerResync();
            this.detachApiFieldObservers();
            this.detachApiFieldListeners();
            this.releaseObjectUrl();
        }

        renderToolGroup() {
            const toolGroups = {
                outpaint: [
                    '<div class="nex-slot__tools" role="group" aria-label="Outpaint mask tools">',
                    '  <button type="button" class="nex-slot__tool mask-tool-btn" id="outpaint-mask-brush">Brush</button>',
                    '  <button type="button" class="nex-slot__tool mask-tool-btn" id="outpaint-mask-erase">Erase</button>',
                    '  <button type="button" class="nex-slot__tool mask-tool-btn nex-slot__tool--subtle" id="outpaint-mask-clear">Clear Mask</button>',
                    '</div>'
                ].join(''),
                remove: [
                    '<div class="nex-slot__tools" role="group" aria-label="Remove mask tools">',
                    '  <button type="button" class="nex-slot__tool mask-tool-btn" id="remove-mask-brush">Brush</button>',
                    '  <button type="button" class="nex-slot__tool mask-tool-btn" id="remove-mask-erase">Erase</button>',
                    '  <button type="button" class="nex-slot__tool mask-tool-btn nex-slot__tool--subtle" id="remove-mask-clear">Clear Mask</button>',
                    '</div>'
                ].join(''),
                'inpaint-base': [
                    '<div class="nex-slot__tools" role="group" aria-label="Inpaint context mask tools">',
                    '  <button type="button" class="nex-slot__tool mask-tool-btn" id="inpaint-context-brush">Brush</button>',
                    '  <button type="button" class="nex-slot__tool mask-tool-btn" id="inpaint-context-erase">Erase</button>',
                    '  <button type="button" class="nex-slot__tool mask-tool-btn nex-slot__tool--subtle" id="inpaint-context-clear">Clear</button>',
                    '</div>'
                ].join(''),
                'inpaint-bb': [
                    '<div class="nex-slot__tools" role="group" aria-label="Inpaint BB mask tools">',
                    '  <button type="button" class="nex-slot__tool mask-tool-btn" id="inpaint-bb-brush">Brush</button>',
                    '  <button type="button" class="nex-slot__tool mask-tool-btn" id="inpaint-bb-erase">Erase</button>',
                    '  <button type="button" class="nex-slot__tool mask-tool-btn nex-slot__tool--subtle" id="inpaint-bb-clear">Clear</button>',
                    '</div>'
                ].join('')
            };
            return toolGroups[this.toolGroup] || '';
        }

        render() {
            const toolsMarkup = this.renderToolGroup();
            this.innerHTML = [
                '<div class="nex-slot">',
                '  <div class="nex-slot__header">',
                `    <div class="nex-slot__label">${this.label}</div>`,
                '    <div class="nex-slot__actions">',
                `      ${toolsMarkup}`,
                '      <button type="button" class="nex-slot__clear" hidden>Remove</button>',
                '    </div>',
                '  </div>',
                '  <div class="nex-slot__drop" draggable="true">',
                `    <div class="nex-slot__placeholder">${this.placeholder}</div>`,
                '    <img class="nex-slot__img" alt="" draggable="false">',
                '  </div>',
                '  <input class="nex-slot__input" type="file" accept="image/*" hidden>',
                '</div>'
            ].join('\n');

            this.slotRoot = this.querySelector('.nex-slot');
            this.dropZone = this.querySelector('.nex-slot__drop');
            this.previewImage = this.querySelector('.nex-slot__img');
            this.fileInput = this.querySelector('.nex-slot__input');
            this.clearButton = this.querySelector('.nex-slot__clear');
        }

        bindEvents() {
            this.dropZone.addEventListener('click', () => this.fileInput.click());

            this.dropZone.addEventListener('dragstart', (event) => {
                if (!this.dropZone.classList.contains('has-image')) {
                    event.preventDefault();
                    return;
                }
                const src = this.previewImage.currentSrc || this.previewImage.src;
                if (!src) {
                    event.preventDefault();
                    return;
                }
                const absoluteUrl = this.toAbsoluteUrl(src);
                event.dataTransfer.setData('text/plain', absoluteUrl);
                event.dataTransfer.setData('text/uri-list', absoluteUrl);
                
                const workspaceId = this.getFieldValue(this.workspaceFieldId);
                const pathValue = this.getFieldValue(this.pathFieldId);
                const payload = JSON.stringify({
                    kind: 'nex-image-source',
                    sourceKind: 'slot',
                    absoluteUrl: absoluteUrl,
                    workspaceId: workspaceId,
                    pathValue: pathValue,
                });
                event.dataTransfer.setData('application/json', payload);
            });

            this.fileInput.addEventListener('change', () => {
                const file = this.fileInput.files && this.fileInput.files[0];
                if (file) {
                    this.handleFile(file);
                }
            });

            this.clearButton.addEventListener('click', async (event) => {
                event.preventDefault();
                event.stopPropagation();
                await this.clearSlot();
            });

            ['dragenter', 'dragover'].forEach((type) => {
                this.dropZone.addEventListener(type, (event) => {
                    event.preventDefault();
                    this.slotRoot.classList.add('is-dragover');
                });
            });

            ['dragleave', 'drop'].forEach((type) => {
                this.dropZone.addEventListener(type, (event) => {
                    event.preventDefault();
                    this.slotRoot.classList.remove('is-dragover');
                });
            });

            this.dropZone.addEventListener('drop', async (event) => {
                const dataTransfer = event.dataTransfer;
                const file = dataTransfer && dataTransfer.files && dataTransfer.files[0];
                if (file) {
                    await this.handleFile(file);
                    return;
                }

                const droppedUrl = this.normalizeDroppedUrl(
                    (dataTransfer && (
                        dataTransfer.getData('text/uri-list') ||
                        dataTransfer.getData('text/plain')
                    )) || ''
                );
                if (!droppedUrl) {
                    return;
                }

                try {
                    const droppedFile = await this.fetchFileFromDroppedUrl(droppedUrl);
                    if (droppedFile) {
                        await this.handleFile(droppedFile);
                    }
                } catch (error) {
                    console.error('[nex-image-slot] URL drop failed:', error);
                }
            });
        }

        normalizeDroppedUrl(raw) {
            if (!raw || typeof raw !== 'string') {
                return '';
            }
            const lines = raw.split('\n').map((line) => line.trim()).filter(Boolean);
            for (const line of lines) {
                if (!line.startsWith('#')) {
                    return line;
                }
            }
            return '';
        }

        observeApp() {
            if (this.appObserver) {
                return;
            }

            this.appObserver = new MutationObserver(() => {
                if (this.appSyncScheduled) {
                    return;
                }
                this.appSyncScheduled = true;
                window.requestAnimationFrame(() => {
                    this.appSyncScheduled = false;
                    if (this.uploadMode === 'api') {
                        const listenersChanged = this.attachApiFieldListeners();
                        const observersChanged = this.attachApiFieldObservers();
                        if (listenersChanged || observersChanged) {
                            this.syncFromApiFields(false);
                        }
                    } else {
                        this.attachBridge();
                        this.syncFromBridge();
                    }
                });
            });
            this.appObserver.observe(appRoot(), { childList: true, subtree: true });
        }

        getFieldControl(fieldId) {
            if (!fieldId) {
                return null;
            }
            const selector = `#${escapeSelector(fieldId)}`;
            const root = appRoot();
            return root.querySelector(`${selector} textarea, ${selector} input`) ||
                document.querySelector(`${selector} textarea, ${selector} input`);
        }

        getFieldElement(fieldId) {
            const control = this.getFieldControl(fieldId);
            if (control) {
                return control;
            }
            if (!fieldId) {
                return null;
            }
            const selector = `#${escapeSelector(fieldId)}`;
            const root = appRoot();
            return root.querySelector(selector) || document.querySelector(selector);
        }
        readFieldValue(field) {
            if (!field) {
                return '';
            }
            if (typeof field.value === 'string' && field.value !== '') {
                return field.value;
            }
            const attrValue = field.getAttribute ? field.getAttribute('value') : '';
            if (typeof attrValue === 'string' && attrValue !== '') {
                return attrValue;
            }
            if (field.tagName === 'TEXTAREA') {
                return field.value || field.textContent || '';
            }
            const nested = field.querySelector ? field.querySelector('textarea, input') : null;
            if (nested && nested !== field) {
                return this.readFieldValue(nested);
            }
            return field.textContent || '';
        }

        getFieldValue(fieldId) {
            const field = this.getFieldElement(fieldId);
            return this.readFieldValue(field);
        }

        setFieldValue(fieldId, value) {
            const field = this.getFieldControl(fieldId);
            if (!field || field.value === value) {
                return;
            }
            field.value = value;
            field.dispatchEvent(new Event('input', { bubbles: true }));
            field.dispatchEvent(new Event('change', { bubbles: true }));
        }

        attachApiFieldListeners() {
            const nextPathField = this.getFieldControl(this.pathFieldId);
            const nextWorkspaceField = this.getFieldControl(this.workspaceFieldId);

            if (this.pathField === nextPathField && this.workspaceField === nextWorkspaceField) {
                return false;
            }

            this.detachApiFieldListeners();

            this.pathField = nextPathField;
            this.workspaceField = nextWorkspaceField;

            [this.pathField, this.workspaceField].forEach((field) => {
                if (!field) {
                    return;
                }
                field.addEventListener('input', this.onApiFieldChange);
                field.addEventListener('change', this.onApiFieldChange);
            });
            return true;
        }

        detachApiFieldListeners() {
            [this.pathField, this.workspaceField].forEach((field) => {
                if (!field) {
                    return;
                }
                field.removeEventListener('input', this.onApiFieldChange);
                field.removeEventListener('change', this.onApiFieldChange);
            });
            this.pathField = null;
            this.workspaceField = null;
        }

        attachApiFieldObservers() {
            const nextPathElement = this.getFieldElement(this.pathFieldId);
            const nextWorkspaceElement = this.getFieldElement(this.workspaceFieldId);

            if (this.apiObservedPathElement === nextPathElement && this.apiObservedWorkspaceElement === nextWorkspaceElement) {
                return false;
            }

            this.detachApiFieldObservers();
            this.apiObservedPathElement = nextPathElement;
            this.apiObservedWorkspaceElement = nextWorkspaceElement;

            const observeTargets = [nextPathElement, nextWorkspaceElement].filter(Boolean);
            if (!observeTargets.length) {
                return true;
            }

            this.apiFieldObserver = new MutationObserver(this.onApiFieldMutation);
            observeTargets.forEach((target) => {
                this.apiFieldObserver.observe(target, {
                    childList: true,
                    subtree: true,
                    characterData: true,
                    attributes: true,
                    attributeFilter: ['value'],
                });
            });
            return true;
        }

        detachApiFieldObservers() {
            if (this.apiFieldObserver) {
                this.apiFieldObserver.disconnect();
                this.apiFieldObserver = null;
            }
            this.apiObservedPathElement = null;
            this.apiObservedWorkspaceElement = null;
        }

        handleServerSyncRequest(event) {
            if (this.uploadMode !== 'api' || !this.pathFieldId) {
                return;
            }
            const detail = event && event.detail ? event.detail : {};
            const pathFieldIds = Array.isArray(detail.pathFieldIds) ? detail.pathFieldIds : [];
            if (!pathFieldIds.includes(this.pathFieldId)) {
                return;
            }
            this.scheduleServerResync(detail.mode === 'once' ? 'once' : 'live');
        }

        clearServerResync() {
            this.serverResyncToken += 1;
            if (this.serverResyncHandle) {
                window.clearTimeout(this.serverResyncHandle);
                this.serverResyncHandle = null;
            }
        }

        isMaskBaseSlot() {
            return ['inpaint_canvas', 'outpaint_input_slot', 'remove_base_image_slot'].includes(this.id || '');
        }

        dispatchBaseImageReplaced(reason = 'replace') {
            if (!this.isMaskBaseSlot()) {
                return;
            }
            window.dispatchEvent(new CustomEvent('nex-slot:base-replaced', {
                detail: {
                    slotId: this.id || '',
                    pathFieldId: this.pathFieldId || '',
                    workspaceFieldId: this.workspaceFieldId || '',
                    reason,
                },
            }));
        }

        scheduleServerResync(mode = 'live') {
            this.clearServerResync();
            const token = this.serverResyncToken;

            if (mode === 'once') {
                let attemptsRemaining = 5;
                const runDelayedSync = () => {
                    if (token !== this.serverResyncToken) {
                        return;
                    }
                    this.syncFromApiFields(true);
                    attemptsRemaining -= 1;
                    if (attemptsRemaining <= 0) {
                        this.serverResyncHandle = null;
                        return;
                    }
                    this.serverResyncHandle = window.setTimeout(runDelayedSync, 500);
                };
                this.serverResyncHandle = window.setTimeout(runDelayedSync, 400);
                return;
            }

            let attemptsRemaining = 5;
            const runSync = () => {
                if (token !== this.serverResyncToken) {
                    return;
                }
                this.syncFromApiFields(true);
                attemptsRemaining -= 1;
                if (attemptsRemaining <= 0) {
                    this.serverResyncHandle = null;
                    return;
                }
                this.serverResyncHandle = window.setTimeout(runSync, 500);
            };
            runSync();
        }

        markUploadState(isUploading) {
            this.dataset.uploading = isUploading ? 'true' : 'false';
            this.slotRoot?.classList.toggle('is-uploading', !!isUploading);
            window.dispatchEvent(new CustomEvent('nex-image-slot:upload-state', {
                detail: {
                    slotId: this.id || '',
                    bridgeId: this.bridgeId || '',
                    pathFieldId: this.pathFieldId || '',
                    workspaceFieldId: this.workspaceFieldId || '',
                    uploading: !!isUploading,
                },
            }));
        }

        async handleFile(file) {
            try {
                if (this.uploadMode === 'api') {
                    this.apiUploadInFlight = true;
                    this.markUploadState(true);
                    await this.prepareApiReplacement();
                    this.dispatchBaseImageReplaced('replace');
                }

                this.setPreview(URL.createObjectURL(file), true);
                if (this.uploadMode === 'api') {
                    await this.pushFileToApi(file);
                } else {
                    this.pushFileToBridge(file);
                }
            } catch (error) {
                console.error('[nex-image-slot] Upload failed:', error);
                this.clearPreview();
            } finally {
                if (this.uploadMode === 'api') {
                    this.apiUploadInFlight = false;
                    this.markUploadState(false);
                    this.syncFromApiFields(true);
                }
            }
        }

        toAbsoluteUrl(value) {
            if (!value) {
                return '';
            }
            try {
                return new URL(String(value), window.location.origin).toString();
            } catch (error) {
                return String(value);
            }
        }

        canFetchDirectly(rawUrl) {
            try {
                const url = new URL(this.toAbsoluteUrl(rawUrl));
                if (url.origin !== window.location.origin) {
                    return false;
                }
                return (
                    url.pathname.startsWith('/staging_api/image/') ||
                    url.pathname.startsWith('/image_api/image/') ||
                    url.pathname.startsWith('/runtime_surface_api/preview_image') ||
                    url.pathname.startsWith('/runtime_surface_api/completed_image/') ||
                    url.pathname.startsWith('/file=')
                );
            } catch (error) {
                return false;
            }
        }

        async buildFileFromResponse(url) {
            const absoluteUrl = this.toAbsoluteUrl(url);
            const response = await fetch(absoluteUrl);
            if (!response.ok) {
                throw new Error(`Image fetch failed with status ${response.status}`);
            }

            const blob = await response.blob();
            const filename = this.getApiFilename(absoluteUrl) || 'dropped_image.png';
            return new File([blob], filename, { type: blob.type || 'image/png' });
        }

        async fetchFileFromDroppedUrl(url) {
            if (this.canFetchDirectly(url)) {
                return this.buildFileFromResponse(url);
            }

            const formData = new FormData();
            formData.append('url', url);

            const uploadResponse = await fetch('/staging_api/upload', {
                method: 'POST',
                body: formData,
            });
            if (!uploadResponse.ok) {
                throw new Error(`Stage URL upload failed with status ${uploadResponse.status}`);
            }

            const payload = await uploadResponse.json();
            if (!payload || payload.status !== 'success' || !payload.url) {
                throw new Error('Stage URL upload returned an invalid payload');
            }

            const stagedFile = await this.buildFileFromResponse(payload.url);
            if (payload.file && payload.file !== stagedFile.name) {
                return new File([stagedFile], payload.file, { type: stagedFile.type || 'image/png' });
            }
            return stagedFile;
        }

        attachBridge() {
            const root = appRoot();
            const bridge = root.querySelector(`#${escapeSelector(this.bridgeId)}`) || document.getElementById(this.bridgeId);
            if (!bridge || bridge === this.bridgeRoot) {
                return Boolean(bridge);
            }

            if (this.bridgeObserver) {
                this.bridgeObserver.disconnect();
            }

            this.bridgeRoot = bridge;
            this.bridgeObserver = new MutationObserver(() => this.syncFromBridge());
            this.bridgeObserver.observe(bridge, {
                childList: true,
                subtree: true,
                attributes: true,
                attributeFilter: ['src', 'class', 'style']
            });
            this.syncFromBridge();
            return true;
        }

        getBridgeFileInput() {
            if (!this.bridgeRoot) {
                return null;
            }
            return this.bridgeRoot.querySelector('input[type="file"]');
        }

        getBridgeClearButton() {
            if (!this.bridgeRoot) {
                return null;
            }
            return this.bridgeRoot.querySelector('button[aria-label="Remove Image"], button[aria-label="Clear"]');
        }

        pushFileToBridge(file) {
            const bridgeInput = this.getBridgeFileInput();
            if (!bridgeInput) {
                window.setTimeout(() => this.pushFileToBridge(file), 150);
                return;
            }

            const transfer = new DataTransfer();
            transfer.items.add(file);
            bridgeInput.files = transfer.files;
            bridgeInput.dispatchEvent(new Event('input', { bubbles: true }));
            bridgeInput.dispatchEvent(new Event('change', { bubbles: true }));
        }

        buildWorkspaceId() {
            const base = (this.id || this.label || 'slot').replace(/[^a-zA-Z0-9]+/g, '_').replace(/^_+|_+$/g, '').toLowerCase() || 'slot';
            const stamp = Date.now().toString(36);
            const rand = Math.random().toString(36).slice(2, 8);
            return `${base}_${stamp}_${rand}`;
        }

        buildApiStateKey(workspaceId, pathValue) {
            if (!pathValue) {
                return '';
            }
            return workspaceId ? `${workspaceId}::${pathValue}` : `direct::${pathValue}`;
        }

        getApiFilename(pathValue) {
            if (!pathValue) {
                return 'base.png';
            }
            const parts = String(pathValue).split(/[\\/]+/).filter(Boolean);
            return parts.length ? parts[parts.length - 1] : 'base.png';
        }

        getApiPreviewUrl(workspaceId, pathValue = '', bustCache = false) {
            if (!pathValue) {
                return '';
            }
            let baseUrl = '';
            if (workspaceId) {
                const filename = this.getApiFilename(pathValue);
                baseUrl = `/image_api/image/${encodeURIComponent(workspaceId)}/${encodeURIComponent(filename)}`;
            } else {
                const normalizedPath = String(pathValue).replace(/\\/g, '/');
                if (normalizedPath.includes('/staging/')) {
                    const filename = this.getApiFilename(pathValue);
                    baseUrl = `/staging_api/image/${encodeURIComponent(filename)}`;
                } else {
                    baseUrl = `/file=${encodeURIComponent(pathValue)}`;
                }
            }
            return bustCache ? `${baseUrl}${baseUrl.includes('?') ? '&' : '?'}v=${Date.now()}` : baseUrl;
        }

        async prepareApiReplacement() {
            const existingWorkspaceId = this.getFieldValue(this.workspaceFieldId);
            if (existingWorkspaceId) {
                try {
                    await fetch(`/image_api/workspace/${encodeURIComponent(existingWorkspaceId)}`, { method: 'DELETE' });
                } catch (error) {
                    console.warn('[nex-image-slot] Previous workspace cleanup failed:', error);
                }
            }
            this.setFieldValue(this.pathFieldId, '');
            this.setFieldValue(this.workspaceFieldId, '');
            this.lastApiStateKey = '';
        }

        async pushFileToApi(file) {
            const workspaceId = this.buildWorkspaceId();
            this.setFieldValue(this.workspaceFieldId, workspaceId);

            const formData = new FormData();
            formData.append('workspace_id', workspaceId);
            if (this.preserveMetadata) {
                formData.append('preserve_metadata', 'true');
            }
            formData.append('file', file, file.name || 'upload.png');

            const response = await fetch('/image_api/upload', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Upload failed with status ${response.status}`);
            }

            const payload = await response.json();
            const resolvedWorkspaceId = payload.workspace_id || workspaceId;
            const resolvedPath = payload.path || '';
            this.setFieldValue(this.workspaceFieldId, resolvedWorkspaceId);
            this.setFieldValue(this.pathFieldId, resolvedPath);
            this.syncFromApiFields(true);
        }

        clearBridge() {
            const clearButton = this.getBridgeClearButton();
            if (clearButton) {
                clearButton.click();
                return;
            }

            const bridgeInput = this.getBridgeFileInput();
            if (!bridgeInput) {
                return;
            }

            bridgeInput.value = '';
            const transfer = new DataTransfer();
            bridgeInput.files = transfer.files;
            bridgeInput.dispatchEvent(new Event('input', { bubbles: true }));
            bridgeInput.dispatchEvent(new Event('change', { bubbles: true }));
        }

        async clearApiState() {
            const workspaceId = this.getFieldValue(this.workspaceFieldId);
            if (workspaceId) {
                try {
                    await fetch(`/image_api/workspace/${encodeURIComponent(workspaceId)}`, { method: 'DELETE' });
                } catch (error) {
                    console.warn('[nex-image-slot] Workspace cleanup failed:', error);
                }
            }
            this.setFieldValue(this.pathFieldId, '');
            this.setFieldValue(this.workspaceFieldId, '');
            this.lastApiStateKey = '';
        }

        async clearSlot() {
            if (this.uploadMode === 'api') {
                await this.clearApiState();
                this.dispatchBaseImageReplaced('clear');
            } else {
                this.clearBridge();
            }
            this.clearPreview();
        }

        syncFromBridge() {
            if (!this.bridgeRoot) {
                return;
            }

            const bridgeImage = this.bridgeRoot.querySelector('img');
            if (!bridgeImage) {
                this.clearPreview();
                return;
            }

            const src = bridgeImage.currentSrc || bridgeImage.src;
            if (!src) {
                this.clearPreview();
                return;
            }

            if (this.previewImage.dataset.source !== src) {
                this.setPreview(src, false);
            }
        }

        syncFromApiFields(forceBustCache = false) {
            if (this.apiUploadInFlight) {
                return;
            }
            const workspaceId = this.getFieldValue(this.workspaceFieldId);
            const pathValue = this.getFieldValue(this.pathFieldId);
            const nextStateKey = this.buildApiStateKey(workspaceId, pathValue);

            if (!nextStateKey) {
                this.lastApiStateKey = '';
                this.clearPreview();
                return;
            }

            const previewUrl = this.getApiPreviewUrl(
                workspaceId,
                pathValue,
                forceBustCache || this.lastApiStateKey !== nextStateKey
            );

            if (this.previewImage.dataset.source !== previewUrl) {
                this.setPreview(previewUrl, false);
            }
            this.lastApiStateKey = nextStateKey;
        }

        setPreview(src, ownsObjectUrl) {
            if (!src) {
                this.clearPreview();
                return;
            }

            if (this.objectUrl && this.objectUrl !== src) {
                this.releaseObjectUrl();
            }

            if (ownsObjectUrl) {
                this.objectUrl = src;
            }

            this.previewImage.src = src;
            this.previewImage.dataset.source = src;
            this.dropZone.classList.add('has-image');
            this.clearButton.hidden = false;
        }

        clearPreview() {
            this.releaseObjectUrl();
            this.previewImage.removeAttribute('src');
            this.previewImage.dataset.source = '';
            this.dropZone.classList.remove('has-image');
            this.clearButton.hidden = true;
            this.fileInput.value = '';
        }

        releaseObjectUrl() {
            if (this.objectUrl) {
                URL.revokeObjectURL(this.objectUrl);
                this.objectUrl = null;
            }
        }
    }

    if (!customElements.get('nex-image-slot')) {
        customElements.define('nex-image-slot', NexImageSlot);
    }
})();
