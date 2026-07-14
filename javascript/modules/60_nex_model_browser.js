(() => {
    const TAB_DEFS = {
        checkpoints: {
            label: 'Checkpoints',
            rootKeys: ['checkpoints'],
            subTabs: [
                { key: 'sdxl', label: 'SDXL', match: (record) => record.architecture === 'sdxl' && (record.sub_architecture === 'base' || record.sub_architecture === 'general') },
                { key: 'pony', label: 'Pony', match: (record) => record.architecture === 'sdxl' && record.sub_architecture === 'pony' },
                { key: 'illustrious', label: 'Illustrious', match: (record) => record.architecture === 'sdxl' && record.sub_architecture === 'illustrious' },
                { key: 'noob', label: 'Noob', match: (record) => record.architecture === 'sdxl' && record.sub_architecture === 'noob' },
            ],
        },
        loras: {
            label: 'LoRAs',
            rootKeys: ['loras'],
            subTabs: [
                { key: 'sdxl', label: 'SDXL', match: (record) => record.architecture === 'sdxl' && (record.sub_architecture === 'base' || record.sub_architecture === 'general') },
                { key: 'pony', label: 'Pony', match: (record) => record.architecture === 'sdxl' && record.sub_architecture === 'pony' },
                { key: 'illustrious', label: 'Illustrious', match: (record) => record.architecture === 'sdxl' && record.sub_architecture === 'illustrious' },
            ],
        },
        others: {
            label: 'Others',
            rootKeys: ['vae', 'embeddings'],
            subTabs: [
                { key: 'sdxl_vae', label: 'SDXL VAE', match: (record) => record.root_key === 'vae' && record.architecture === 'sdxl' },
                { key: 'sdxl_embeddings', label: 'SDXL Embeddings', match: (record) => record.root_key === 'embeddings' && record.architecture === 'sdxl' },
            ],
        },
    };

    const SECTION_KEYS = ['installed_registered', 'installed_unregistered', 'available_registered'];
    const SECTION_LABELS = {
        installed_registered: 'Installed and Registered',
        installed_unregistered: 'Installed and Unregistered',
        available_registered: 'Available for Download',
    };
    const MODEL_TYPE_OPTIONS = ['checkpoint', 'lora', 'vae', 'embedding'];
    const ARCHITECTURE_OPTIONS = ['unknown', 'sd15', 'sdxl'];
    const SUB_ARCHITECTURE_OPTIONS = ['general', 'none', 'base', 'pony', 'illustrious', 'noob'];
    const SOURCE_PROVIDER_OPTIONS = ['local', 'civitai', 'huggingface', 'github'];
    const NON_EDITABLE_ADD_MODEL_CATALOG_IDS = new Set([
        'user.local.models',
        'user.civitai.main',
        'user.huggingface.main',
        'user.github.main',
    ]);
    const DRAG_ROOTS = new Set(['checkpoints', 'vae', 'loras', 'embeddings']);
    const DROP_TARGETS = [
        { selector: '#model_base_dropdown', target: 'base_model', roots: ['checkpoints'] },
        { selector: '#model_vae_dropdown', target: 'vae_model', roots: ['vae'] },
        { selector: '[id^="lora_model_dropdown_"]', target: (node) => `lora_model:${String(node.id).split('_').pop()}`, roots: ['loras'] },
    ];
    const PROMPT_TARGETS = [
        { selector: '#positive_prompt', label: 'positive prompt' },
        { selector: '#negative_prompt', label: 'negative prompt' },
    ];

    function gradioRoot() {
        return window.gradioApp ? window.gradioApp() : document;
    }

    function escapeHtml(value) {
        return String(value ?? '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    async function fetchJson(url, options = {}) {
        const response = await fetch(url, {
            headers: { 'Content-Type': 'application/json' },
            ...options,
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(payload?.detail || response.statusText || 'Request failed');
        }
        return payload;
    }

    function sortRecords(records) {
        return records.slice().sort((a, b) => String(a.display_name || a.name).localeCompare(String(b.display_name || b.name)));
    }

    class NexModelBrowser extends HTMLElement {
        constructor() {
            super();
            this.state = {
                activeTab: 'checkpoints',
                activeSubTabs: {},
                tabData: {},
                loadingTabs: new Set(),
                selectedByRoot: {},
                drawer: null,
                jobs: {},
                pendingActivations: {},
                status: '',
                statusTone: 'info',
                thumbnailRevision: 0,
            };
            Object.entries(TAB_DEFS).forEach(([tabKey, def]) => {
                this.state.activeSubTabs[tabKey] = def.subTabs[0].key;
                def.rootKeys.forEach((rootKey) => {
                    this.state.selectedByRoot[rootKey] = new Set();
                });
            });
            this.handleClick = this.handleClick.bind(this);
            this.handleInput = this.handleInput.bind(this);
            this.handleChange = this.handleChange.bind(this);
            this.handleDragStart = this.handleDragStart.bind(this);
            this.thumbnailSession = String(Date.now());
            this.pollHandle = null;
        }

        connectedCallback() {
            if (this.dataset.ready === 'true') return;
            this.dataset.ready = 'true';
            this.addEventListener('click', this.handleClick);
            this.addEventListener('input', this.handleInput);
            this.addEventListener('change', this.handleChange);
            this.addEventListener('dragstart', this.handleDragStart);
            this.render();
            this.ensureTabLoaded(this.state.activeTab);
        }

        disconnectedCallback() {
            this.removeEventListener('click', this.handleClick);
            this.removeEventListener('input', this.handleInput);
            this.removeEventListener('change', this.handleChange);
            this.removeEventListener('dragstart', this.handleDragStart);
            if (this.pollHandle) {
                window.clearTimeout(this.pollHandle);
                this.pollHandle = null;
            }
        }

        queryById(id) {
            if (!id) return null;
            return document.getElementById(id) || gradioRoot().querySelector(`#${id}`);
        }

        sleep(ms) {
            return new Promise((resolve) => window.setTimeout(resolve, ms));
        }

        get refreshButton() {
            return this.queryById(this.dataset.refreshButtonId);
        }

        get applyDataField() {
            return this.queryById(this.dataset.applyDataId);
        }

        thumbnailUrl(record) {
            const params = new URLSearchParams({
                selector: record.id,
                rev: `${this.thumbnailSession}.${this.state.thumbnailRevision}.${record.thumbnail_library_relative || ''}`,
            });
            return `/api/models/thumbnail/file?${params.toString()}`;
        }

        embeddingToken(name) {
            const stem = String(name || '').replace(/\.[^.]+$/, '').trim();
            return stem ? `(embedding:${stem}:1.0)` : '';
        }

        normalizeSubArchitectureValue(rootKey, modelType, subArchitecture) {
            if (rootKey === 'vae' || rootKey === 'embeddings' || modelType === 'vae' || modelType === 'embedding') {
                return 'none';
            }
            return subArchitecture || 'general';
        }

        normalizeDrawerForm(form, rootKey = '') {
            return {
                ...form,
                sub_architecture: this.normalizeSubArchitectureValue(rootKey, form.model_type, form.sub_architecture),
            };
        }

        resolveTextField(wrapper) {
            return wrapper?.matches('input, textarea') ? wrapper : wrapper?.querySelector('input, textarea');
        }

        captureFieldFocus(input) {
            if (!input?.dataset?.field || !input.matches('input, textarea')) return null;
            return {
                field: input.dataset.field,
                start: typeof input.selectionStart === 'number' ? input.selectionStart : null,
                end: typeof input.selectionEnd === 'number' ? input.selectionEnd : null,
            };
        }

        restoreFieldFocus(focusState) {
            if (!focusState?.field) return;
            window.requestAnimationFrame(() => {
                const target = Array.from(this.querySelectorAll('.nmb-drawer [data-field]'))
                    .find((node) => node.dataset.field === focusState.field && node.matches('input, textarea'));
                if (!target) return;
                target.focus();
                if (typeof focusState.start === 'number' && typeof focusState.end === 'number' && typeof target.setSelectionRange === 'function') {
                    const valueLength = String(target.value || '').length;
                    const start = Math.max(0, Math.min(focusState.start, valueLength));
                    const end = Math.max(start, Math.min(focusState.end, valueLength));
                    target.setSelectionRange(start, end);
                }
            });
        }

        setStatus(message, tone = 'info') {
            this.state.status = message || '';
            this.state.statusTone = tone;
            this.render();
        }

        clearStatus() {
            this.state.status = '';
            this.state.statusTone = 'info';
        }

        switchTab(tabKey) {
            if (!TAB_DEFS[tabKey]) return;
            this.state.activeTab = tabKey;
            this.clearStatus();
            this.render();
            this.ensureTabLoaded(tabKey);
        }

        switchSubTab(subTabKey) {
            this.state.activeSubTabs[this.state.activeTab] = subTabKey;
            this.clearStatus();
            this.render();
        }

        tabSelectedCount(tabKey) {
            return TAB_DEFS[tabKey].rootKeys.reduce((count, rootKey) => count + this.state.selectedByRoot[rootKey].size, 0);
        }

        activeRootKeys() {
            return TAB_DEFS[this.state.activeTab].rootKeys;
        }

        currentSubTabs() {
            const top = TAB_DEFS[this.state.activeTab];
            if (this.state.activeTab !== 'others') return top.subTabs;
            const visible = top.subTabs.filter((subTab) => this.subTabCount(subTab.key) > 0);
            return visible.length ? visible : top.subTabs;
        }

        activeSubTabDef() {
            const subTabs = this.currentSubTabs();
            const activeKey = this.state.activeSubTabs[this.state.activeTab];
            const resolved = subTabs.find((subTab) => subTab.key === activeKey) || subTabs[0];
            if (resolved && resolved.key !== activeKey) {
                this.state.activeSubTabs[this.state.activeTab] = resolved.key;
            }
            return resolved;
        }

        activeDownloadSelectors() {
            const subTab = this.activeSubTabDef();
            const tabData = this.state.tabData[this.state.activeTab] || {};
            return this.activeRootKeys().flatMap((rootKey) => {
                const selected = this.state.selectedByRoot[rootKey] || new Set();
                return (tabData[rootKey]?.available_registered || [])
                    .filter((record) => selected.has(record.id) && subTab.match(record))
                    .map((record) => record.id);
            });
        }

        toggleSelection(rootKey, selector) {
            const selection = this.state.selectedByRoot[rootKey] || new Set();
            if (selection.has(selector)) selection.delete(selector);
            else selection.add(selector);
            this.state.selectedByRoot[rootKey] = selection;
            this.render();
        }

        clearActiveSelection() {
            const activeSelectorSet = new Set(this.activeDownloadSelectors());
            this.activeRootKeys().forEach((rootKey) => {
                const selection = this.state.selectedByRoot[rootKey] || new Set();
                activeSelectorSet.forEach((selector) => selection.delete(selector));
                this.state.selectedByRoot[rootKey] = selection;
            });
            this.render();
        }

        async ensureTabLoaded(tabKey, { force = false } = {}) {
            if (this.state.loadingTabs.has(tabKey)) return;
            if (!force && this.state.tabData[tabKey]) return;
            this.state.loadingTabs.add(tabKey);
            this.render();
            try {
                const rootPayloads = {};
                await Promise.all(TAB_DEFS[tabKey].rootKeys.map(async (rootKey) => {
                    const params = new URLSearchParams({ root_key: rootKey });
                    rootPayloads[rootKey] = await fetchJson(`/api/models/browser?${params.toString()}`);
                }));
                this.state.tabData[tabKey] = rootPayloads;
            } catch (error) {
                this.setStatus(error.message || 'Failed to load models.', 'error');
            } finally {
                this.state.loadingTabs.delete(tabKey);
                this.render();
            }
        }
        async refreshAllData() {
            this.state.thumbnailRevision += 1;
            try {
                await fetchJson('/api/models/refresh', { method: 'POST' });
                if (this.refreshButton) this.refreshButton.click();
                await Promise.all(Object.keys(TAB_DEFS).map((tabKey) => this.ensureTabLoaded(tabKey, { force: true })));
            } catch (error) {
                this.setStatus(error.message || 'Failed to refresh models.', 'error');
            }
        }

        async queueActiveDownloads() {
            const selectors = this.activeDownloadSelectors();
            if (!selectors.length) {
                this.setStatus('Select one or more available models in this subtab first.', 'warning');
                return;
            }
            this.setStatus('Queueing downloads...', 'info');
            try {
                const payload = await fetchJson('/api/models/downloads/batch', {
                    method: 'POST',
                    body: JSON.stringify({ root_keys: this.activeRootKeys(), selectors }),
                });
                if (!payload.jobs?.length) {
                    const details = (payload.skipped || []).slice(0, 3).map((item) => `${item.selector || 'model'}: ${String(item.reason || 'skipped').replaceAll('_', ' ')}`).join(' | ');
                    this.setStatus(details || 'No downloadable models were queued from this subtab.', 'warning');
                    return;
                }
                payload.jobs.forEach((job) => { this.state.jobs[job.job_id] = job; });
                this.setStatus(`Queued ${payload.jobs.length} download${payload.jobs.length === 1 ? '' : 's'} in this subtab.`, 'success');
                this.startPollingJobs();
                this.render();
            } catch (error) {
                this.setStatus(error.message || 'Failed to queue downloads.', 'error');
            }
        }

        startPollingJobs() {
            if (this.pollHandle) return;
            const poll = async () => {
                const pendingIds = Object.keys(this.state.jobs).filter((jobId) => {
                    const status = this.state.jobs[jobId]?.status;
                    return status !== 'succeeded' && status !== 'failed';
                });
                if (!pendingIds.length) {
                    this.pollHandle = null;
                    return;
                }
                if (this.offsetParent === null) {
                    this.pollHandle = window.setTimeout(poll, 1800);
                    return;
                }
                try {
                    const results = await Promise.all(pendingIds.map((jobId) => fetchJson(`/api/models/downloads/${jobId}`)));
                    let completed = false;
                    results.forEach((job) => {
                        this.state.jobs[job.job_id] = job;
                        if (job.status === 'succeeded' || job.status === 'failed') completed = true;
                        if (job.status === 'succeeded') {
                            Object.keys(this.state.selectedByRoot).forEach((rootKey) => this.state.selectedByRoot[rootKey].delete(job.entry_id));
                        }
                    });
                    const failed = results.filter((job) => job.status === 'failed');
                    if (failed.length) {
                        const job = failed[0];
                        this.setStatus(job.message || job.error || `Download failed for ${job.entry_id}.`, 'error');
                    }
                    if (completed) {
                        await this.refreshAllData();
                        await this.finalizePendingActivations();
                    }
                } catch (error) {
                    this.setStatus(error.message || 'Download polling failed.', 'error');
                }
                this.render();
                this.pollHandle = window.setTimeout(poll, 1200);
            };
            this.pollHandle = window.setTimeout(poll, 1200);
        }

        rootKeyForModelType(modelType) {
            return {
                checkpoint: 'checkpoints',
                lora: 'loras',
                vae: 'vae',
                embedding: 'embeddings',
            }[String(modelType || '').trim().toLowerCase()] || '';
        }

        drawerRootKey(drawer = this.state.drawer) {
            return drawer?.entry?.root_key || this.rootKeyForModelType(drawer?.form?.model_type || '');
        }

        defaultDrawerForm(overrides = {}) {
            return {
                display_name: '',
                name: '',
                installed_relative_path: '',
                relative_path: '',
                model_type: 'checkpoint',
                architecture: 'unknown',
                sub_architecture: 'general',
                thumbnail_library_relative: '',
                thumbnail_source_name: '',
                source_provider: 'local',
                source_version_id: '',
                source_url: '',
                source_input: '',
                alias: '',
                target_catalog_id: '',
                new_catalog_name: '',
                asset_group_key: '',
                catalog_id: '',
                catalog_label: '',
                filename: '',
                notes: '',
                companion_clip_selector: '',
                companion_clip_relative_path: '',
                ...overrides,
            };
        }

        activePanelDefaults() {
            const subKey = this.activeSubTabDef()?.key || '';
            let rootKey = this.activeRootKeys()[0] || 'checkpoints';
            if (this.state.activeTab === 'others') {
                if (subKey.includes('vae')) rootKey = 'vae';
                else rootKey = 'embeddings';
            }
            const modelType = {
                checkpoints: 'checkpoint',
                loras: 'lora',
                vae: 'vae',
                embeddings: 'embedding',
            }[rootKey] || 'checkpoint';
            const architecture = subKey.startsWith('sd15') ? 'sd15' : 'sdxl';
            let subArchitecture = 'base';
            if (['pony', 'illustrious', 'noob'].includes(subKey)) subArchitecture = subKey;
            else if (rootKey === 'vae' || rootKey === 'embeddings') subArchitecture = 'none';
            return { root_key: rootKey, model_type: modelType, architecture, sub_architecture: subArchitecture };
        }

        catalogOptionsForProvider(provider = '') {
            const normalizedProvider = String(provider || '').trim().toLowerCase();
            return (this.state.drawer?.personalCatalogs || []).filter((catalog) => {
                if (catalog.is_system_catalog) return false;
                if (!normalizedProvider) return true;
                return catalog.source_provider === normalizedProvider || catalog.source_provider === 'local';
            });
        }

        defaultLocalCatalogId(catalogs = this.state.drawer?.personalCatalogs || []) {
            return catalogs.find((catalog) => catalog.is_default_local_catalog)?.catalog_id || '';
        }

        defaultDisplayName(value = '') {
            const stem = String(value || '').replace(/\.[^.]+$/, '').trim();
            return stem.replace(/[_-]+/g, ' ').trim();
        }

        slugifyIdentifier(value = '', fallback = 'entry') {
            const normalized = String(value || '').trim().toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
            return normalized || fallback;
        }

        deriveFilenameFromAddSource(sourceProvider = '', sourceInput = '') {
            if (!['huggingface', 'github'].includes(String(sourceProvider || '').trim().toLowerCase())) return '';
            try {
                const parsed = new URL(String(sourceInput || '').trim());
                const candidate = decodeURIComponent((parsed.pathname || '').split('/').pop() || '').trim();
                return candidate || '';
            } catch (error) {
                return '';
            }
        }

        defaultAddModelCatalogLabel() {
            return 'Default personal download catalog';
        }

        addModelCatalogOptions() {
            return (this.state.drawer?.personalCatalogs || []).filter((catalog) => {
                return !catalog.is_system_catalog
                    && !catalog.is_default_local_catalog
                    && !NON_EDITABLE_ADD_MODEL_CATALOG_IDS.has(String(catalog.catalog_id || '').trim());
            });
        }

        renderProviderToggle(selectedProvider = 'huggingface', disabled = false) {
            const providers = [
                ['huggingface', 'HuggingFace'],
                ['civitai', 'CivitAI'],
                ['github', 'GitHub'],
            ];
            return `
                <div class="nmb-provider-toggle" role="group" aria-label="Source Provider">
                    ${providers.map(([value, label]) => `
                        <button type="button" class="nmb-provider-toggle__button ${selectedProvider === value ? 'is-active' : ''}" data-action="set-source-provider" data-provider="${escapeHtml(value)}" ${disabled ? 'disabled' : ''}>${escapeHtml(label)}</button>
                    `).join('')}
                </div>
            `;
        }

        renderAddModelCatalogSelect(provider, value = '', disabled = false) {
            void provider;
            const options = [`<option value="">${escapeHtml(this.defaultAddModelCatalogLabel())}</option>`];
            this.addModelCatalogOptions().forEach((catalog) => {
                const selected = catalog.catalog_id === value ? 'selected' : '';
                options.push(`<option value="${escapeHtml(catalog.catalog_id)}" ${selected}>${escapeHtml(catalog.catalog_label)} [${escapeHtml(catalog.source_provider)}]</option>`);
            });
            return `<select data-field="target_catalog_id" ${disabled ? 'disabled' : ''}>${options.join('')}</select>`;
        }

        buildPersonalCatalogDraft(name, provider) {
            const label = String(name || '').trim();
            const slug = this.slugifyIdentifier(label, 'catalog');
            return {
                catalog_id: `user.personal.${slug}`,
                catalog_label: label,
                filename: slug,
                source_provider: 'local',
            };
        }

        syncAddModelDerivedFields(previousForm = {}) {
            if (this.state.drawer?.mode !== 'add_model') return;
            const form = this.state.drawer.form || {};
            const previous = previousForm || {};
            const previousDerivedName = previous.name || this.deriveFilenameFromAddSource(previous.source_provider, previous.source_input);
            const currentDerivedName = form.name || this.deriveFilenameFromAddSource(form.source_provider, form.source_input);
            if (!form.name && currentDerivedName) {
                form.name = currentDerivedName;
            }

            const previousAutoDisplay = previousDerivedName ? this.defaultDisplayName(previousDerivedName) : '';
            const currentAutoDisplay = form.name ? this.defaultDisplayName(form.name) : '';
            const canUpdateDisplay = !previous.display_name || previous.display_name === previousAutoDisplay;
            if (currentAutoDisplay && canUpdateDisplay) {
                form.display_name = currentAutoDisplay;
            }

            const previousAutoAlias = this.slugifyIdentifier(previous.display_name || previousAutoDisplay, this.slugifyIdentifier(previousDerivedName, 'model'));
            const currentAutoAlias = this.slugifyIdentifier(form.display_name || currentAutoDisplay, this.slugifyIdentifier(form.name, 'model'));
            const canUpdateAlias = !previous.alias || previous.alias === previousAutoAlias;
            if (currentAutoAlias && canUpdateAlias) {
                form.alias = currentAutoAlias;
            }

            if (form.source_provider !== previous.source_provider) {
                const validCatalogIds = new Set(this.addModelCatalogOptions().map((catalog) => catalog.catalog_id));
                if (form.target_catalog_id && !validCatalogIds.has(form.target_catalog_id)) {
                    form.target_catalog_id = '';
                }
            }
            this.state.drawer.form = form;
        }

        resolveActivationConfig(rootKey, promptTarget = '') {
            if (rootKey === 'embeddings') {
                if (promptTarget === 'positive') {
                    return { promptTarget: '#positive_prompt', label: 'positive prompt' };
                }
                if (promptTarget === 'negative') {
                    return { promptTarget: '#negative_prompt', label: 'negative prompt' };
                }
                return null;
            }
            if (rootKey === 'checkpoints') {
                return { targetKey: 'base_model', acceptedRoots: ['checkpoints'] };
            }
            if (rootKey === 'vae') return { targetKey: 'vae_model', acceptedRoots: ['vae'] };
            if (rootKey === 'loras') return { targetKey: this.resolvePreferredLoraTarget(), acceptedRoots: ['loras'] };
            return null;
        }

        async finalizePendingActivations() {
            const pendingEntries = Object.entries(this.state.pendingActivations || {});
            for (const [key, activation] of pendingEntries) {
                const jobStates = (activation.jobIds || []).map((jobId) => this.state.jobs[jobId]?.status).filter(Boolean);
                if (!jobStates.length) continue;
                if (jobStates.some((status) => status === 'failed')) {
                    delete this.state.pendingActivations[key];
                    continue;
                }
                if (!jobStates.every((status) => status === 'succeeded')) continue;

                if (activation.promptTarget) {
                    this.insertEmbeddingIntoTarget(activation.token, activation.promptTarget, activation.label);
                } else {
                    await this.applyDropToTarget({ selector: activation.selector, rootKey: activation.rootKey, token: '' }, activation.targetKey, activation.acceptedRoots);
                }
                delete this.state.pendingActivations[key];
            }
        }

        async activateAvailableSelector(selector, rootKey, promptTarget = '') {
            if (!selector || !rootKey) return;
            const activation = this.resolveActivationConfig(rootKey, promptTarget);
            if (!activation) {
                this.setStatus(rootKey === 'embeddings'
                    ? 'Choose whether to insert the embedding into the positive or negative prompt.'
                    : `Activate is not supported for ${rootKey}.`, 'warning');
                return;
            }
            this.setStatus('Queueing download and activation...', 'info');
            try {
                const payload = await fetchJson('/api/models/download', {
                    method: 'POST',
                    body: JSON.stringify({ selector }),
                });
                const jobs = payload.jobs || [];
                jobs.forEach((job) => { this.state.jobs[job.job_id] = job; });
                if (!jobs.length) {
                    await this.refreshAllData();
                    if (activation.promptTarget) {
                        this.insertEmbeddingIntoTarget(this.embeddingToken(selector), activation.promptTarget, activation.label);
                    } else {
                        await this.applyDropToTarget({ selector, rootKey, token: '' }, activation.targetKey, activation.acceptedRoots);
                    }
                    return;
                }
                this.state.pendingActivations[selector] = {
                    selector,
                    rootKey,
                    token: this.embeddingToken((this.getRecordById(selector)?.name) || selector),
                    ...activation,
                    jobIds: jobs.map((job) => job.job_id),
                };
                this.setStatus(`Queued ${jobs.length} download${jobs.length === 1 ? '' : 's'} and will activate when ready.`, 'success');
                this.startPollingJobs();
                this.render();
            } catch (error) {
                this.setStatus(error.message || 'Failed to queue activation download.', 'error');
            }
        }

        getRecordById(selector) {
            for (const rootPayloads of Object.values(this.state.tabData || {})) {
                for (const payload of Object.values(rootPayloads || {})) {
                    for (const sectionKey of SECTION_KEYS) {
                        const match = (payload?.[sectionKey] || []).find((record) => record.id === selector);
                        if (match) return match;
                    }
                }
            }
            return null;
        }

        async loadPersonalCatalogs() {
            try {
                const payload = await fetchJson('/api/models/personal-catalogs');
                return payload.catalogs || [];
            } catch (error) {
                this.setStatus(error.message || 'Failed to load personal catalogs.', 'error');
                return [];
            }
        }

        async openDrawer(selector, sourceProvider = '', sourceVersionId = '', matchedSelector = '') {
            this.state.drawer = {
                loading: true,
                selector,
                matchedSelector: matchedSelector || '',
                mode: 'registration',
                form: this.defaultDrawerForm({
                    source_provider: sourceProvider || 'local',
                    source_version_id: sourceVersionId || '',
                }),
                suggestions: [],
                companionClip: null,
                error: '',
                installedLink: null,
                canEditCatalogFields: true,
                personalCatalogs: [],
            };
            this.render();
            try {
                const params = new URLSearchParams({ selector, suggest_limit: '3' });
                if (sourceProvider) params.set('source_provider', sourceProvider);
                if (sourceVersionId) params.set('source_version_id', sourceVersionId);
                if (matchedSelector) params.set('matched_selector', matchedSelector);
                const payload = await fetchJson(`/api/models/registration?${params.toString()}`);
                const companionClip = payload.companion_clip || null;
                const personalCatalogs = payload.personal_catalogs || [];
                this.state.drawer = {
                    ...this.state.drawer,
                    loading: false,
                    mode: 'registration',
                    matchedSelector: matchedSelector || this.state.drawer.matchedSelector,
                    entry: payload.entry,
                    companionClip,
                    personalCatalogs,
                    form: this.normalizeDrawerForm(this.defaultDrawerForm({
                        display_name: payload.entry.display_name || '',
                        name: payload.entry.name || '',
                        installed_relative_path: payload.entry.installed_relative_path || payload.entry.relative_path || '',
                        relative_path: payload.entry.relative_path || '',
                        model_type: payload.entry.model_type || 'checkpoint',
                        architecture: payload.entry.architecture || 'unknown',
                        sub_architecture: payload.entry.sub_architecture || 'general',
                        thumbnail_library_relative: payload.entry.thumbnail_library_relative || payload.thumbnail?.relative_path || '',
                        source_provider: sourceProvider || payload.entry.source_provider || 'local',
                        source_version_id: sourceVersionId || payload.entry.source_version_id || '',
                        source_url: payload.entry.source?.url || '',
                        target_catalog_id: this.defaultLocalCatalogId(personalCatalogs),
                        companion_clip_selector: companionClip?.recommended_selector || '',
                        companion_clip_relative_path: '',
                    }), payload.entry.root_key || ''),
                    suggestions: payload.suggestions || [],
                };
            } catch (error) {
                this.state.drawer = { ...this.state.drawer, loading: false, error: error.message || 'Failed to load registration details.' };
            }
            this.render();
        }

        async openInstalledDrawer(selector) {
            const loadedRecord = this.getRecordById(selector) || {};
            this.state.drawer = {
                loading: true,
                selector,
                matchedSelector: '',
                mode: 'installed_link',
                entry: loadedRecord,
                form: this.normalizeDrawerForm(this.defaultDrawerForm({
                    display_name: loadedRecord.display_name || '',
                    name: loadedRecord.name || '',
                    installed_relative_path: loadedRecord.installed_relative_path || '',
                    relative_path: loadedRecord.relative_path || '',
                    model_type: loadedRecord.model_type || 'checkpoint',
                    architecture: loadedRecord.architecture || 'unknown',
                    sub_architecture: loadedRecord.sub_architecture || 'general',
                    thumbnail_library_relative: loadedRecord.thumbnail_library_relative || '',
                    source_provider: loadedRecord.source_provider || 'local',
                    source_version_id: loadedRecord.source_version_id || '',
                    target_catalog_id: '',
                }), loadedRecord.root_key || ''),
                suggestions: [],
                error: '',
                installedLink: null,
                canEditCatalogFields: false,
                personalCatalogs: [],
            };
            this.render();
            try {
                const params = new URLSearchParams({ selector, suggest_limit: '3' });
                const payload = await fetchJson(`/api/models/installed-link?${params.toString()}`);
                const mergedEntry = { ...loadedRecord, ...(payload.entry || {}) };
                this.state.drawer = {
                    ...this.state.drawer,
                    loading: false,
                    mode: 'installed_link',
                    entry: mergedEntry,
                    installedLink: payload.installed_link || null,
                    suggestions: payload.suggestions || [],
                    canEditCatalogFields: Boolean(payload.can_edit_catalog_fields),
                    personalCatalogs: payload.personal_catalogs || [],
                    form: this.normalizeDrawerForm(this.defaultDrawerForm({
                        display_name: mergedEntry.display_name || '',
                        name: mergedEntry.name || '',
                        installed_relative_path: payload.installed_link?.installed_relative_path || mergedEntry.installed_relative_path || '',
                        relative_path: mergedEntry.relative_path || '',
                        model_type: mergedEntry.model_type || 'checkpoint',
                        architecture: mergedEntry.architecture || 'unknown',
                        sub_architecture: mergedEntry.sub_architecture || 'general',
                        thumbnail_library_relative: mergedEntry.thumbnail_library_relative || payload.thumbnail?.relative_path || '',
                        source_provider: mergedEntry.source_provider || 'local',
                        source_version_id: mergedEntry.source_version_id || '',
                        source_url: mergedEntry.source?.url || payload.entry?.source?.url || '',
                        target_catalog_id: '',
                    }), mergedEntry.root_key || ''),
                };
            } catch (error) {
                this.state.drawer = { ...this.state.drawer, loading: false, error: error.message || 'Failed to load installed model details.' };
            }
            this.render();
        }

        async openAddDrawer() {
            const defaults = this.activePanelDefaults();
            const personalCatalogs = await this.loadPersonalCatalogs();
            this.state.drawer = {
                loading: false,
                selector: '',
                matchedSelector: '',
                mode: 'add_model',
                form: this.normalizeDrawerForm(this.defaultDrawerForm({
                    source_provider: 'huggingface',
                    source_input: '',
                    model_type: defaults.model_type,
                    architecture: defaults.architecture,
                    sub_architecture: defaults.sub_architecture,
                    target_catalog_id: '',
                    new_catalog_name: '',
                    thumbnail_source_name: '',
                }), defaults.root_key),
                pendingThumbnailFile: null,
                suggestions: [],
                companionClip: null,
                error: '',
                installedLink: null,
                canEditCatalogFields: true,
                personalCatalogs,
            };
            this.syncAddModelDerivedFields();
            this.render();
        }

        openCreateCatalogDrawer() {
            this.state.drawer = {
                loading: false,
                selector: '',
                matchedSelector: '',
                mode: 'create_catalog',
                form: this.defaultDrawerForm({
                    source_provider: 'local',
                    catalog_id: '',
                    catalog_label: '',
                    filename: '',
                    notes: '',
                }),
                suggestions: [],
                companionClip: null,
                error: '',
                installedLink: null,
                canEditCatalogFields: true,
                personalCatalogs: [],
            };
            this.render();
        }

        openImportCatalogDrawer() {
            this.state.drawer = {
                loading: false,
                selector: '',
                matchedSelector: '',
                mode: 'import_catalog',
                form: this.defaultDrawerForm({ filename: '' }),
                importCatalog: null,
                importPreview: null,
                error: '',
                personalCatalogs: [],
            };
            this.render();
        }

        closeDrawer() {
            this.state.drawer = null;
            this.render();
        }

        async refreshSuggestions() {
            if (!this.state.drawer) return;
            const { selector, form, matchedSelector, mode } = this.state.drawer;
            if (mode === 'installed_link') {
                await this.openInstalledDrawer(selector);
                return;
            }
            if (mode === 'registration') {
                await this.openDrawer(selector, form.source_provider, form.source_version_id, matchedSelector);
            }
        }

        chooseSuggestion(selector) {
            if (!this.state.drawer) return;
            const suggestion = (this.state.drawer.suggestions || []).find((item) => item.entry.id === selector);
            if (!suggestion) return;
            this.state.drawer.matchedSelector = suggestion.entry.id;
            this.state.drawer.form = this.normalizeDrawerForm({
                ...this.state.drawer.form,
                display_name: suggestion.entry.display_name || this.state.drawer.form.display_name,
                name: suggestion.entry.name || this.state.drawer.form.name,
                relative_path: suggestion.entry.relative_path || this.state.drawer.form.relative_path,
                model_type: suggestion.entry.model_type || this.state.drawer.form.model_type,
                architecture: suggestion.entry.architecture || this.state.drawer.form.architecture,
                sub_architecture: suggestion.entry.sub_architecture || this.state.drawer.form.sub_architecture,
                thumbnail_library_relative: suggestion.entry.thumbnail_library_relative || this.state.drawer.form.thumbnail_library_relative,
                source_provider: suggestion.entry.source_provider || this.state.drawer.form.source_provider,
                source_version_id: suggestion.entry.source_version_id || this.state.drawer.form.source_version_id,
                source_url: suggestion.entry.source?.url || this.state.drawer.form.source_url,
            }, this.drawerRootKey());
            this.render();
        }

        async loadImportCatalogFile(file) {
            if (!this.state.drawer || !file) return;
            try {
                const raw = await file.text();
                const catalog = JSON.parse(raw);
                this.state.drawer.form.filename = file.name;
                this.state.drawer.importCatalog = catalog;
                this.state.drawer.importPreview = {
                    catalog_id: catalog.catalog_id || '',
                    catalog_label: catalog.catalog_label || '',
                    source_provider: catalog.source_provider || '',
                    entry_count: Array.isArray(catalog.entries) ? catalog.entries.length : 0,
                };
                this.setStatus(`Loaded ${file.name} for import.`, 'success');
                this.render();
            } catch (error) {
                this.state.drawer.error = error.message || 'Failed to read catalog JSON.';
                this.render();
            }
        }

        async uploadThumbnailFile(selector, file) {
            if (!selector || !file) return null;
            this.setStatus(`Uploading thumbnail ${file.name}...`, 'info');
            const formData = new FormData();
            formData.append('selector', selector);
            formData.append('file', file, file.name);
            const response = await fetch('/api/models/thumbnail/upload', {
                method: 'POST',
                body: formData,
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok) {
                throw new Error(payload?.detail || response.statusText || 'Thumbnail upload failed');
            }
            this.state.thumbnailRevision += 1;
            return payload;
        }

        async uploadDrawerThumbnail(file) {
            if (!this.state.drawer || !file || !this.state.drawer.selector) return;
            try {
                const payload = await this.uploadThumbnailFile(this.state.drawer.selector, file);
                this.state.drawer.entry = payload.entry || this.state.drawer.entry;
                this.state.drawer.form = this.normalizeDrawerForm({
                    ...this.state.drawer.form,
                    thumbnail_library_relative: payload.entry?.thumbnail_library_relative || payload.thumbnail?.relative_path || this.state.drawer.form.thumbnail_library_relative,
                }, this.drawerRootKey());
                this.setStatus(`Updated thumbnail for ${payload.entry?.display_name || payload.entry?.name || 'model'}.`, 'success');
                this.render();
            } catch (error) {
                this.setStatus(error.message || 'Failed to upload thumbnail.', 'error');
            }
        }

        async saveDrawer() {
            if (!this.state.drawer) return;
            const { selector, matchedSelector, form, mode } = this.state.drawer;
            this.state.drawer.loading = true;
            this.state.drawer.error = '';
            this.render();
            try {
                if (mode === 'registration' || mode === 'installed_link') {
                    const endpoint = mode === 'installed_link' ? '/api/models/installed-link' : '/api/models/registration';
                    await fetchJson(endpoint, {
                        method: 'POST',
                        body: JSON.stringify({
                            selector,
                            matched_selector: matchedSelector || undefined,
                            target_catalog_id: form.target_catalog_id || undefined,
                            updates: {
                                display_name: form.display_name,
                                name: form.name,
                                installed_relative_path: form.installed_relative_path,
                                relative_path: form.relative_path,
                                model_type: form.model_type,
                                architecture: form.architecture,
                                sub_architecture: form.sub_architecture,
                                source_provider: form.source_provider,
                                source_version_id: form.source_version_id,
                                source_url: form.source_url,
                                thumbnail_library_relative: form.thumbnail_library_relative,
                            },
                        }),
                    });
                    await this.refreshAllData();
                    this.setStatus(mode === 'installed_link' ? 'Installed model link saved.' : 'Model registration saved.', 'success');
                    this.closeDrawer();
                    return;
                }

                if (mode === 'add_model') {
                    let targetCatalogId = form.target_catalog_id || undefined;
                    if (String(form.new_catalog_name || '').trim()) {
                        const catalogDraft = this.buildPersonalCatalogDraft(form.new_catalog_name, form.source_provider);
                        await fetchJson('/api/models/personal-catalogs', {
                            method: 'POST',
                            body: JSON.stringify(catalogDraft),
                        });
                        targetCatalogId = catalogDraft.catalog_id;
                    }
                    const added = await fetchJson('/api/models/add', {
                        method: 'POST',
                        body: JSON.stringify({
                            source_provider: form.source_provider,
                            source_input: form.source_input,
                            target_catalog_id: targetCatalogId,
                            model_type: form.model_type,
                            architecture: form.architecture,
                            sub_architecture: form.sub_architecture,
                            name: form.name || undefined,
                            display_name: form.display_name || undefined,
                            alias: form.alias || undefined,
                        }),
                    });
                    if (this.state.drawer.pendingThumbnailFile && added?.entry?.id) {
                        await this.uploadThumbnailFile(added.entry.id, this.state.drawer.pendingThumbnailFile);
                    }
                    await this.refreshAllData();
                    this.setStatus('Added model to personal catalog.', 'success');
                    this.closeDrawer();
                    return;
                }

                if (mode === 'create_catalog') {
                    await fetchJson('/api/models/personal-catalogs', {
                        method: 'POST',
                        body: JSON.stringify({
                            catalog_id: form.catalog_id,
                            catalog_label: form.catalog_label,
                            source_provider: form.source_provider,
                            filename: form.filename || undefined,
                            notes: String(form.notes || '').split(/\r?\n/).map((item) => item.trim()).filter(Boolean),
                        }),
                    });
                    await this.refreshAllData();
                    this.setStatus('Created personal catalog.', 'success');
                    this.closeDrawer();
                    return;
                }

                if (mode === 'import_catalog') {
                    if (!this.state.drawer.importCatalog) {
                        throw new Error('Choose a catalog JSON file first.');
                    }
                    await fetchJson('/api/models/personal-catalogs/import', {
                        method: 'POST',
                        body: JSON.stringify({
                            filename: form.filename || undefined,
                            catalog: this.state.drawer.importCatalog,
                        }),
                    });
                    await this.refreshAllData();
                    this.setStatus('Imported personal catalog.', 'success');
                    this.closeDrawer();
                }
            } catch (error) {
                this.state.drawer.loading = false;
                this.state.drawer.error = error.message || 'Failed to save model panel changes.';
                this.render();
            }
        }

        recordsForSection(sectionKey) {
            const subTab = this.activeSubTabDef();
            const tabData = this.state.tabData[this.state.activeTab] || {};
            const records = this.activeRootKeys().flatMap((rootKey) => (tabData[rootKey]?.[sectionKey] || []).filter((record) => subTab.match(record)));
            return sortRecords(records);
        }

        subTabCount(subTabKey) {
            const subTab = TAB_DEFS[this.state.activeTab].subTabs.find((item) => item.key === subTabKey);
            if (!subTab) return 0;
            return SECTION_KEYS.reduce((total, sectionKey) => {
                const tabData = this.state.tabData[this.state.activeTab] || {};
                const sectionRecords = this.activeRootKeys().flatMap((rootKey) => tabData[rootKey]?.[sectionKey] || []);
                return total + sectionRecords.filter((record) => subTab.match(record)).length;
            }, 0);
        }

        setBridgeValue(wrapper, value) {
            if (!wrapper) return false;
            const input = this.resolveTextField(wrapper);
            if (!input) return false;
            input.value = value ?? '';
            input.dispatchEvent(new Event('input', { bubbles: true }));
            input.dispatchEvent(new Event('change', { bubbles: true }));
            return true;
        }

        resolveDragPayload(event) {
            const raw = event.dataTransfer?.getData('application/json') || event.dataTransfer?.getData('text/plain');
            if (!raw) return null;
            try {
                const payload = JSON.parse(raw);
                if (payload?.selector && payload?.rootKey) return payload;
            } catch (_) {
                return null;
            }
            return null;
        }

        async applyDropToTarget(payload, targetKey, acceptedRoots) {
            if (!payload?.selector || !payload?.rootKey) return;
            if (!acceptedRoots.includes(payload.rootKey)) {
                this.setStatus(`This drop target only accepts ${acceptedRoots.join(' / ')} models.`, 'warning');
                return;
            }
            const field = this.applyDataField;
            if (!field) {
                this.setStatus('Model apply bridge is not ready yet. Try reloading the UI.', 'error');
                return;
            }
            const applied = this.setBridgeValue(field, JSON.stringify({
                selector: payload.selector,
                target: targetKey,
                aspect_ratio: this.currentAspectRatioValue(),
                ts: Date.now(),
            }));
            if (!applied) {
                this.setStatus('Model apply bridge could not update the apply field. Try reloading the UI.', 'error');
                return;
            }
            this.setStatus(`Applying ${payload.rootKey} to ${targetKey.replace('_', ' ')}...`, 'info');
        }

        currentAspectRatioValue() {
            const root = gradioRoot();
            const checked = root.querySelector('#aspect_ratios_selection input[type="radio"]:checked') || root.querySelector('.aspect_ratios input[type="radio"]:checked');
            if (checked && checked.value) return String(checked.value);
            const checkedLabel = checked?.closest('label');
            const text = checkedLabel?.querySelector('span')?.textContent || checkedLabel?.textContent;
            return text ? String(text).trim() : '';
        }

        applyPromptDrop(payload, targetNode, label) {
            if (!payload?.token || payload.rootKey !== 'embeddings') return;
            const field = this.resolveTextField(targetNode);
            if (!field) {
                this.setStatus(`Could not find the ${label} field.`, 'error');
                return;
            }
            const currentValue = String(field.value || '').trim();
            const token = payload.token.trim();
            const nextValue = currentValue ? `${currentValue}, ${token}` : token;
            field.value = nextValue;
            field.dispatchEvent(new Event('input', { bubbles: true }));
            field.dispatchEvent(new Event('change', { bubbles: true }));
            field.focus();
            this.setStatus(`Inserted ${token} into the ${label}.`, 'success');
        }

        resolvePreferredLoraTarget() {
            const root = gradioRoot();
            const dropdowns = Array.from(root.querySelectorAll('[id^="lora_model_dropdown_"]'));
            if (!dropdowns.length) return 'lora_model:1';
            for (const node of dropdowns) {
                const field = this.resolveTextField(node);
                const value = String(field?.value || '').trim();
                if (!value || value === 'None') {
                    const suffix = String(node.id).split('_').pop() || '1';
                    return `lora_model:${suffix}`;
                }
            }
            const suffix = String(dropdowns[0].id).split('_').pop() || '1';
            return `lora_model:${suffix}`;
        }

        async applyInstalledSelector(selector, rootKey) {
            if (!selector || !rootKey) return;
            let targetKey = null;
            let acceptedRoots = [rootKey];
            if (rootKey === 'checkpoints') {
                targetKey = 'base_model';
                acceptedRoots = ['checkpoints'];
            } else if (rootKey === 'vae') {
                targetKey = 'vae_model';
            } else if (rootKey === 'loras') {
                targetKey = this.resolvePreferredLoraTarget();
            }
            if (!targetKey) {
                this.setStatus(`Apply is not supported for ${rootKey}.`, 'warning');
                return;
            }
            await this.applyDropToTarget({ selector, rootKey, token: '' }, targetKey, acceptedRoots);
        }

        insertEmbeddingIntoTarget(token, targetSelector, label) {
            if (!token) return;
            const root = gradioRoot();
            const targetNode = root.querySelector(targetSelector);
            if (!targetNode) {
                this.setStatus(`Could not find the ${label}.`, 'error');
                return;
            }
            this.applyPromptDrop({ rootKey: 'embeddings', token }, targetNode, label);
        }

        installDropTargets() {
            const root = gradioRoot();
            DROP_TARGETS.forEach((binding) => {
                root.querySelectorAll(binding.selector).forEach((node) => {
                    if (node.dataset.nmbDropReady === 'true') return;
                    node.dataset.nmbDropReady = 'true';
                    node.classList.add('nmb-browser-drop-target');
                    const targetKey = typeof binding.target === 'function' ? binding.target(node) : binding.target;
                    const acceptedRoots = binding.roots;
                    node.addEventListener('dragover', (event) => {
                        const payload = this.resolveDragPayload(event);
                        if (!payload || !acceptedRoots.includes(payload.rootKey)) return;
                        event.preventDefault();
                        node.classList.add('is-drop-ready');
                    });
                    node.addEventListener('dragleave', () => {
                        node.classList.remove('is-drop-ready');
                    });
                    node.addEventListener('drop', (event) => {
                        const payload = this.resolveDragPayload(event);
                        node.classList.remove('is-drop-ready');
                        if (!payload || !acceptedRoots.includes(payload.rootKey)) return;
                        event.preventDefault();
                        this.applyDropToTarget(payload, targetKey, acceptedRoots);
                    });
                });
            });
            PROMPT_TARGETS.forEach((binding) => {
                root.querySelectorAll(binding.selector).forEach((node) => {
                    if (node.dataset.nmbPromptDropReady === 'true') return;
                    node.dataset.nmbPromptDropReady = 'true';
                    node.classList.add('nmb-browser-drop-target');
                    node.addEventListener('dragover', (event) => {
                        const payload = this.resolveDragPayload(event);
                        if (!payload || payload.rootKey !== 'embeddings') return;
                        event.preventDefault();
                        node.classList.add('is-drop-ready');
                    });
                    node.addEventListener('dragleave', () => {
                        node.classList.remove('is-drop-ready');
                    });
                    node.addEventListener('drop', (event) => {
                        const payload = this.resolveDragPayload(event);
                        node.classList.remove('is-drop-ready');
                        if (!payload || payload.rootKey !== 'embeddings') return;
                        event.preventDefault();
                        this.applyPromptDrop(payload, node, binding.label);
                    });
                });
            });
        }

        handleDragStart(event) {
            const card = event.target.closest('.nmb-card[draggable="true"]');
            if (!card) return;
            const payload = {
                selector: card.dataset.selector,
                rootKey: card.dataset.rootKey,
                token: card.dataset.token || '',
            };
            event.dataTransfer.effectAllowed = 'copy';
            event.dataTransfer.setData('application/json', JSON.stringify(payload));
            event.dataTransfer.setData('text/plain', JSON.stringify(payload));
        }
        renderTopTabs() {
            return Object.entries(TAB_DEFS).map(([tabKey, tab]) => {
                const count = this.tabSelectedCount(tabKey);
                return `<button type="button" class="nmb-top-tab ${this.state.activeTab === tabKey ? 'is-active' : ''}" data-action="switch-top-tab" data-tab-key="${escapeHtml(tabKey)}">${escapeHtml(tab.label)}${count ? ` <span class="nmb-pill">${count}</span>` : ''}</button>`;
            }).join('');
        }

        renderSubTabs() {
            const activeKey = this.activeSubTabDef()?.key;
            return this.currentSubTabs().map((subTab) => {
                const count = this.subTabCount(subTab.key);
                return `<button type="button" class="nmb-sub-tab ${activeKey === subTab.key ? 'is-active' : ''}" data-action="switch-sub-tab" data-sub-tab-key="${escapeHtml(subTab.key)}">${escapeHtml(subTab.label)} <span class="nmb-sub-tab__count">${count}</span></button>`;
            }).join('');
        }

        renderCard(record, sectionKey) {
            const selectable = sectionKey === 'available_registered';
            const unregistered = sectionKey === 'installed_unregistered';
            const installedRegistered = sectionKey === 'installed_registered';
            const draggable = installedRegistered && DRAG_ROOTS.has(record.root_key);
            const selected = this.state.selectedByRoot[record.root_key]?.has(record.id);
            const surfaceAction = unregistered
                ? 'open-drawer'
                : (installedRegistered ? 'open-installed-drawer' : (selectable ? 'toggle-selection' : ''));
            const stateLabel = unregistered
                ? 'Registration Required'
                : (installedRegistered
                    ? 'Installed'
                    : (selectable ? (selected ? 'Selected for Download' : 'Available for Download') : 'Installed'));

            let actions = '';
            if (installedRegistered) {
                if (record.root_key === 'embeddings') {
                    actions = `
                        <div class="nmb-card__actions">
                            <button type="button" class="nmb-secondary nmb-card__action" data-action="insert-embedding-positive" data-token="${escapeHtml(this.embeddingToken(record.name))}">Insert Positive</button>
                            <button type="button" class="nmb-secondary nmb-card__action" data-action="insert-embedding-negative" data-token="${escapeHtml(this.embeddingToken(record.name))}">Insert Negative</button>
                            <button type="button" class="nmb-secondary nmb-card__action" data-action="open-installed-drawer" data-selector="${escapeHtml(record.id)}">Review</button>
                        </div>
                    `;
                } else {
                    actions = `
                        <div class="nmb-card__actions">
                            <button type="button" class="nmb-primary nmb-card__action" data-action="apply-installed-default" data-selector="${escapeHtml(record.id)}" data-root-key="${escapeHtml(record.root_key)}">Apply</button>
                            <button type="button" class="nmb-secondary nmb-card__action" data-action="open-installed-drawer" data-selector="${escapeHtml(record.id)}">Review</button>
                        </div>
                    `;
                }
            } else if (unregistered) {
                actions = `
                    <div class="nmb-card__actions">
                        <button type="button" class="nmb-primary nmb-card__action" data-action="open-drawer" data-selector="${escapeHtml(record.id)}">Register</button>
                    </div>
                `;
            } else if (selectable) {
                if (record.root_key === 'embeddings') {
                    actions = `
                        <div class="nmb-card__actions">
                            <button type="button" class="nmb-primary nmb-card__action" data-action="activate-available-positive" data-selector="${escapeHtml(record.id)}" data-root-key="${escapeHtml(record.root_key)}">Download + Positive</button>
                            <button type="button" class="nmb-secondary nmb-card__action" data-action="activate-available-negative" data-selector="${escapeHtml(record.id)}" data-root-key="${escapeHtml(record.root_key)}">Download + Negative</button>
                        </div>
                    `;
                } else {
                    actions = `
                        <div class="nmb-card__actions">
                            <button type="button" class="nmb-primary nmb-card__action" data-action="activate-available" data-selector="${escapeHtml(record.id)}" data-root-key="${escapeHtml(record.root_key)}">Download + Apply</button>
                        </div>
                    `;
                }
            }

            return `
                <div class="nmb-card ${selected ? 'is-selected' : ''} ${unregistered ? 'is-unregistered' : ''} ${draggable ? 'is-draggable' : ''}" data-root-key="${escapeHtml(record.root_key)}" data-selector="${escapeHtml(record.id)}">
                    <button type="button" class="nmb-card__surface" ${surfaceAction ? `data-action="${surfaceAction}"` : ''} ${surfaceAction ? `data-selector="${escapeHtml(record.id)}"` : ''} ${surfaceAction === 'toggle-selection' ? `data-root-key="${escapeHtml(record.root_key)}"` : ''} ${draggable ? 'draggable="true"' : ''} data-root-key="${escapeHtml(record.root_key)}" data-selector="${escapeHtml(record.id)}" data-token="${escapeHtml(this.embeddingToken(record.name))}">
                        <div class="nmb-card__media">
                            <img class="nmb-card__thumb" src="${escapeHtml(this.thumbnailUrl(record))}" alt="${escapeHtml(record.display_name || record.name || 'Model thumbnail')}" loading="lazy">
                            <div class="nmb-card__badge">${escapeHtml((record.display_name || record.name || 'N').slice(0, 3).toUpperCase())}</div>
                        </div>
                        <div class="nmb-card__content">
                            <div class="nmb-card__title">${escapeHtml(record.display_name || record.name)}</div>
                            <div class="nmb-card__filename">${escapeHtml(record.name)}</div>
                            <div class="nmb-card__meta">${escapeHtml(record.root_key)}${record.source_provider ? ` | ${escapeHtml(record.source_provider)}` : ''}${record.source_version_id ? ` | ${escapeHtml(record.source_version_id)}` : ''}</div>
                            <div class="nmb-card__state">${escapeHtml(stateLabel)}</div>
                        </div>
                    </button>
                    ${actions}
                </div>
            `;
        }

        renderSection(sectionKey) {
            const records = this.recordsForSection(sectionKey);
            const cards = records.length ? records.map((record) => this.renderCard(record, sectionKey)).join('') : '<div class="nmb-empty">No models in this section.</div>';
            return `
                <section class="nmb-section nmb-section--${escapeHtml(sectionKey)}">
                    <div class="nmb-section__header">
                        <div>
                            <h3 class="nmb-section__title">${escapeHtml(SECTION_LABELS[sectionKey])}</h3>
                            <div class="nmb-section__count">${records.length} model${records.length === 1 ? '' : 's'}</div>
                        </div>
                    </div>
                    <div class="nmb-card-grid">${cards}</div>
                </section>
            `;
        }

        renderJobs() {
            const jobs = Object.values(this.state.jobs).sort((a, b) => Number(b.created_at || 0) - Number(a.created_at || 0)).slice(0, 5);
            if (!jobs.length) return '';
            return `<div class="nmb-jobs">${jobs.map((job) => {
                const detail = job.error || job.message || '';
                return `<div class="nmb-job nmb-job--${escapeHtml(job.status || 'queued')}"><div class="nmb-job__main"><div class="nmb-job__title">${escapeHtml(job.entry_id)}</div>${detail ? `<div class="nmb-job__detail">${escapeHtml(detail)}</div>` : ''}</div><div class="nmb-job__status">${escapeHtml(job.status)}</div></div>`;
            }).join('')}</div>`;
        }

        renderSelect(field, options, value, disabled = false) {
            const resolved = value || options[0];
            return `<select data-field="${escapeHtml(field)}" ${disabled ? 'disabled' : ''}>${options.map((option) => `<option value="${escapeHtml(option)}" ${option === resolved ? 'selected' : ''}>${escapeHtml(option)}</option>`).join('')}</select>`;
        }

        renderCatalogSelect(field, provider, value = '', disabled = false, includeAutomatic = true) {
            const options = [];
            if (includeAutomatic) {
                options.push(`<option value="">Default personal catalog</option>`);
            }
            this.catalogOptionsForProvider(provider).forEach((catalog) => {
                const selected = catalog.catalog_id === value ? 'selected' : '';
                options.push(`<option value="${escapeHtml(catalog.catalog_id)}" ${selected}>${escapeHtml(catalog.catalog_label)} [${escapeHtml(catalog.source_provider)}]</option>`);
            });
            return `<select data-field="${escapeHtml(field)}" ${disabled ? 'disabled' : ''}>${options.join('')}</select>`;
        }

        renderRegistrationDrawer(drawer) {
            const form = drawer.form || {};
            const entry = drawer.entry || {};
            const installedLink = drawer.installedLink || {};
            const installedLinkMode = drawer.mode === 'installed_link';
            const canEditCatalogFields = !installedLinkMode || drawer.canEditCatalogFields;
            const drawerTitle = installedLinkMode ? 'Edit Installed Model' : 'Register Model';
            const drawerSubtitle = installedLinkMode
                ? 'Review the linked installed path and retarget the catalog entry if needed.'
                : 'Review this unregistered model and store it in a managed personal catalog.';
            const currentInstalledPath = form.installed_relative_path || installedLink.installed_relative_path || entry.installed_relative_path || entry.relative_path || '';
            const currentCatalogPath = entry.relative_path || form.relative_path || '';
            const saveLabel = installedLinkMode ? 'Save Installed Link' : 'Save Registration';
            const suggestions = (drawer.suggestions || []).length
                ? drawer.suggestions.map((item) => `
                    <button type="button" class="nmb-suggestion ${drawer.matchedSelector === item.entry.id ? 'is-selected' : ''}" data-action="choose-suggestion" data-selector="${escapeHtml(item.entry.id)}">
                        <div class="nmb-suggestion__title">${escapeHtml(item.entry.display_name || item.entry.name)}</div>
                        <div class="nmb-suggestion__meta">${escapeHtml(item.entry.source_provider || 'local')}${item.entry.source_version_id ? ` | ${escapeHtml(item.entry.source_version_id)}` : ''}</div>
                        <div class="nmb-suggestion__score">Score ${escapeHtml(item.score)}</div>
                    </button>
                `).join('')
                : '<div class="nmb-empty">No suggestions yet.</div>';
            return `
                <aside class="nmb-drawer ${drawer.loading ? 'is-loading' : ''}">
                    <div class="nmb-drawer__header">
                        <div>
                            <h3>${escapeHtml(drawerTitle)}</h3>
                            <p>${escapeHtml(drawerSubtitle)}</p>
                        </div>
                        <button type="button" class="nmb-secondary" data-action="close-drawer">Close</button>
                    </div>
                    <div class="nmb-drawer__current">
                        <div class="nmb-drawer__current-label">Selected Model</div>
                        <div class="nmb-drawer__current-name">${escapeHtml(entry.display_name || entry.name || form.name || 'Unnamed model')}</div>
                        <div class="nmb-drawer__current-path">Installed: ${escapeHtml(currentInstalledPath)}</div>
                        <div class="nmb-drawer__current-path">Catalog: ${escapeHtml(currentCatalogPath)}</div>
                        <div class="nmb-drawer__current-meta">${escapeHtml(entry.root_key || '')}${form.architecture ? ` | ${escapeHtml(form.architecture)}` : ''}${form.sub_architecture ? ` | ${escapeHtml(form.sub_architecture)}` : ''}</div>
                    </div>
                    ${drawer.error ? `<div class="nmb-status nmb-status--error">${escapeHtml(drawer.error)}</div>` : ''}
                    <div class="nmb-form-grid">
                        <label class="nmb-field--wide"><span>Target Catalog</span>${this.renderCatalogSelect('target_catalog_id', form.source_provider, form.target_catalog_id || '', drawer.loading, true)}</label>
                        <label class="nmb-field--wide"><span>Installed Relative Path</span><input data-field="installed_relative_path" value="${escapeHtml(form.installed_relative_path || '')}" placeholder="Path under the configured model root"></label>
                        <label><span>Display Name</span><input data-field="display_name" value="${escapeHtml(form.display_name || '')}" ${canEditCatalogFields ? '' : 'readonly'}></label>
                        <label><span>Canonical Name</span><input data-field="name" value="${escapeHtml(form.name || '')}" ${canEditCatalogFields ? '' : 'readonly'}></label>
                        <label class="nmb-field--wide"><span>Catalog Relative Path</span><input data-field="relative_path" value="${escapeHtml(form.relative_path || '')}" ${canEditCatalogFields ? '' : 'readonly'}></label>
                        <label><span>Model Type</span>${this.renderSelect('model_type', MODEL_TYPE_OPTIONS, form.model_type, !canEditCatalogFields)}</label>
                        <label><span>Architecture</span>${this.renderSelect('architecture', ARCHITECTURE_OPTIONS, form.architecture, !canEditCatalogFields)}</label>
                        <label><span>Sub-Architecture</span>${this.renderSelect('sub_architecture', SUB_ARCHITECTURE_OPTIONS, form.sub_architecture, !canEditCatalogFields)}</label>
                        <label><span>Source Provider</span>${this.renderSelect('source_provider', SOURCE_PROVIDER_OPTIONS, form.source_provider, !canEditCatalogFields)}</label>
                        <label><span>Version ID</span><input data-field="source_version_id" value="${escapeHtml(form.source_version_id || '')}" ${canEditCatalogFields ? '' : 'readonly'}></label>
                        <label class="nmb-field--wide nmb-field--picker">
                            <span>Thumbnail Path</span>
                            <div class="nmb-field__picker">
                                <input data-field="thumbnail_library_relative" value="${escapeHtml(form.thumbnail_library_relative || '')}" placeholder="Optional thumbnail path under the thumbnail library" ${canEditCatalogFields ? '' : 'readonly'}>
                                <button type="button" class="nmb-secondary nmb-field__picker-button" data-action="pick-thumbnail-file" ${canEditCatalogFields ? '' : 'disabled'}>Browse...</button>
                            </div>
                        </label>
                        <label class="nmb-field--wide"><span>Source URL</span><input data-field="source_url" value="${escapeHtml(form.source_url || '')}" placeholder="Optional" ${canEditCatalogFields ? '' : 'readonly'}></label>
                    </div>
                    <div class="nmb-drawer__thumbnail-actions">
                        <input type="file" accept="image/*" data-thumbnail-upload hidden ${canEditCatalogFields ? '' : 'disabled'}>
                        <button type="button" class="nmb-secondary" data-action="pick-thumbnail-file" ${canEditCatalogFields ? '' : 'disabled'}>Choose Thumbnail Image</button>
                    </div>
                    <div class="nmb-drawer__actions">
                        <button type="button" class="nmb-secondary" data-action="refresh-suggestions">Refresh Suggestions</button>
                        <button type="button" class="nmb-primary" data-action="save-drawer">${escapeHtml(saveLabel)}</button>
                    </div>
                    <div class="nmb-drawer__suggestions">
                        <h4>Possible Matches</h4>
                        <p class="nmb-drawer__hint">Selecting a suggestion pre-fills the canonical metadata. Save to confirm the catalog link for this installed file.</p>
                        <div class="nmb-suggestions">${suggestions}</div>
                    </div>
                </aside>
            `;
        }

        renderCreateCatalogDrawer(drawer) {
            const form = drawer.form || {};
            return `
                <aside class="nmb-drawer ${drawer.loading ? 'is-loading' : ''}">
                    <div class="nmb-drawer__header">
                        <div>
                            <h3>Create Personal Catalog</h3>
                            <p>Create a managed JSON catalog in the app-owned catalog folder.</p>
                        </div>
                        <button type="button" class="nmb-secondary" data-action="close-drawer">Close</button>
                    </div>
                    ${drawer.error ? `<div class="nmb-status nmb-status--error">${escapeHtml(drawer.error)}</div>` : ''}
                    <div class="nmb-form-grid">
                        <label><span>Catalog ID</span><input data-field="catalog_id" value="${escapeHtml(form.catalog_id || '')}" placeholder="user.local.my_catalog"></label>
                        <label><span>Catalog Label</span><input data-field="catalog_label" value="${escapeHtml(form.catalog_label || '')}" placeholder="My Personal Catalog"></label>
                        <label><span>Source Provider</span>${this.renderSelect('source_provider', SOURCE_PROVIDER_OPTIONS, form.source_provider || 'local', drawer.loading)}</label>
                        <label><span>Filename</span><input data-field="filename" value="${escapeHtml(form.filename || '')}" placeholder="optional_catalog_name"></label>
                        <label class="nmb-field--wide"><span>Notes</span><textarea data-field="notes" rows="4" placeholder="Optional notes, one per line">${escapeHtml(form.notes || '')}</textarea></label>
                    </div>
                    <div class="nmb-drawer__actions">
                        <button type="button" class="nmb-primary" data-action="save-drawer">Create Catalog</button>
                    </div>
                </aside>
            `;
        }

        renderImportCatalogDrawer(drawer) {
            const form = drawer.form || {};
            const preview = drawer.importPreview;
            return `
                <aside class="nmb-drawer ${drawer.loading ? 'is-loading' : ''}">
                    <div class="nmb-drawer__header">
                        <div>
                            <h3>Import Personal Catalog</h3>
                            <p>Import a user-provided catalog JSON into the managed catalog area.</p>
                        </div>
                        <button type="button" class="nmb-secondary" data-action="close-drawer">Close</button>
                    </div>
                    ${drawer.error ? `<div class="nmb-status nmb-status--error">${escapeHtml(drawer.error)}</div>` : ''}
                    <div class="nmb-form-grid">
                        <label class="nmb-field--wide nmb-field--picker">
                            <span>Catalog JSON File</span>
                            <div class="nmb-field__picker">
                                <input data-field="filename" value="${escapeHtml(form.filename || '')}" placeholder="Choose a .json file" readonly>
                                <button type="button" class="nmb-secondary nmb-field__picker-button" data-action="pick-import-file">Browse...</button>
                            </div>
                        </label>
                    </div>
                    ${preview ? `
                        <div class="nmb-drawer__current">
                            <div class="nmb-drawer__current-label">Import Preview</div>
                            <div class="nmb-drawer__current-name">${escapeHtml(preview.catalog_label || preview.catalog_id || form.filename || 'Catalog')}</div>
                            <div class="nmb-drawer__current-path">Catalog ID: ${escapeHtml(preview.catalog_id || 'unknown')}</div>
                            <div class="nmb-drawer__current-meta">${escapeHtml(preview.source_provider || 'unknown')} | ${escapeHtml(preview.entry_count)} entries</div>
                        </div>
                    ` : '<div class="nmb-empty">Choose a catalog JSON file to preview and import it.</div>'}
                    <div class="nmb-drawer__actions">
                        <input type="file" accept="application/json,.json" data-import-upload hidden>
                        <button type="button" class="nmb-primary" data-action="save-drawer" ${drawer.importCatalog ? '' : 'disabled'}>Import Catalog</button>
                    </div>
                </aside>
            `;
        }

        renderAddModelDrawer(drawer) {
            const form = drawer.form || {};
            const provider = form.source_provider || 'huggingface';
            const sourceLabel = provider === 'civitai' ? 'Version ID' : 'Download URL';
            const sourcePlaceholder = provider === 'civitai'
                ? 'CivitAI model version id'
                : provider === 'github'
                    ? 'https://github.com/.../releases/download/.../model.safetensors'
                    : 'https://huggingface.co/.../resolve/.../model.safetensors';
            return `
                <aside class="nmb-drawer ${drawer.loading ? 'is-loading' : ''}">
                    <div class="nmb-drawer__header">
                        <div>
                            <h3>Add Downloadable Model</h3>
                            <p>Add a Hugging Face, GitHub, or CivitAI model source into a managed personal catalog.</p>
                        </div>
                        <button type="button" class="nmb-secondary" data-action="close-drawer">Close</button>
                    </div>
                    ${drawer.error ? `<div class="nmb-status nmb-status--error">${escapeHtml(drawer.error)}</div>` : ''}
                    <div class="nmb-form-grid">
                        <label class="nmb-field--wide"><span>Source Provider</span>${this.renderProviderToggle(provider, drawer.loading)}</label>
                        <label class="nmb-field--wide"><span>${escapeHtml(sourceLabel)}</span><input data-field="source_input" value="${escapeHtml(form.source_input || '')}" placeholder="${escapeHtml(sourcePlaceholder)}"></label>
                        <label class="nmb-field--wide"><span>Select a catalog to add</span>${this.renderAddModelCatalogSelect(provider, form.target_catalog_id || '', drawer.loading)}</label>
                        <label class="nmb-field--wide"><span>Create a new catalog to add</span><input data-field="new_catalog_name" value="${escapeHtml(form.new_catalog_name || '')}"></label>
                        <label><span>Model Type</span>${this.renderSelect('model_type', MODEL_TYPE_OPTIONS, form.model_type, drawer.loading)}</label>
                        <label><span>Architecture</span>${this.renderSelect('architecture', ARCHITECTURE_OPTIONS.filter((option) => option !== 'unknown'), form.architecture || 'sdxl', drawer.loading)}</label>
                        <label><span>Sub-Architecture</span>${this.renderSelect('sub_architecture', SUB_ARCHITECTURE_OPTIONS, form.sub_architecture, drawer.loading)}</label>
                        <label><span>Filename</span><input data-field="name" value="${escapeHtml(form.name || '')}" placeholder=""></label>
                        <label><span>Display Name</span><input data-field="display_name" value="${escapeHtml(form.display_name || '')}" placeholder=""></label>
                        <label><span>Alias</span><input data-field="alias" value="${escapeHtml(form.alias || '')}" placeholder=""></label>
                        <label class="nmb-field--wide nmb-field--picker">
                            <span>Thumbnail Source Image</span>
                            <div class="nmb-field__picker">
                                <input data-field="thumbnail_source_name" value="${escapeHtml(form.thumbnail_source_name || '')}" placeholder="Optional local image for thumbnail generation" readonly>
                                <button type="button" class="nmb-secondary nmb-field__picker-button" data-action="pick-thumbnail-file">Browse...</button>
                            </div>
                        </label>
                    </div>
                    <div class="nmb-drawer__thumbnail-actions">
                        <input type="file" accept="image/*" data-thumbnail-upload hidden>
                    </div>
                    <div class="nmb-drawer__actions">
                        <button type="button" class="nmb-primary" data-action="save-drawer">Add Model</button>
                    </div>
                </aside>
            `;
        }

        renderDrawer() {
            const drawer = this.state.drawer;
            if (!drawer) return '';
            let content = '';
            if (drawer.mode === 'create_catalog') content = this.renderCreateCatalogDrawer(drawer);
            else if (drawer.mode === 'import_catalog') content = this.renderImportCatalogDrawer(drawer);
            else if (drawer.mode === 'add_model') content = this.renderAddModelDrawer(drawer);
            else content = this.renderRegistrationDrawer(drawer);
            return `<div class="nmb-overlay">${content}</div>`;
        }
        handleClick(event) {
            if (event.target === this.querySelector('.nmb-overlay')) {
                this.closeDrawer();
                return;
            }
            const actionNode = event.target.closest('[data-action]');
            if (!actionNode) return;
            const { action } = actionNode.dataset;
            if (action === 'switch-top-tab') return this.switchTab(actionNode.dataset.tabKey);
            if (action === 'switch-sub-tab') return this.switchSubTab(actionNode.dataset.subTabKey);
            if (action === 'toggle-selection') return this.toggleSelection(actionNode.dataset.rootKey, actionNode.dataset.selector);
            if (action === 'open-drawer') return this.openDrawer(actionNode.dataset.selector);
            if (action === 'open-installed-drawer') return this.openInstalledDrawer(actionNode.dataset.selector);
            if (action === 'open-add-drawer') return this.openAddDrawer();
            if (action === 'set-source-provider') {
                if (this.state.drawer?.mode !== 'add_model') return;
                const previousForm = { ...this.state.drawer.form };
                this.state.drawer.form.source_provider = actionNode.dataset.provider || 'huggingface';
                this.state.drawer.form = this.normalizeDrawerForm(this.state.drawer.form, this.drawerRootKey());
                this.syncAddModelDerivedFields(previousForm);
                this.render();
                return;
            }
            if (action === 'open-create-catalog') return this.openCreateCatalogDrawer();
            if (action === 'open-import-catalog') return this.openImportCatalogDrawer();
            if (action === 'close-drawer') return this.closeDrawer();
            if (action === 'refresh-suggestions') return this.refreshSuggestions();
            if (action === 'choose-suggestion') return this.chooseSuggestion(actionNode.dataset.selector);
            if (action === 'apply-installed-default') return this.applyInstalledSelector(actionNode.dataset.selector, actionNode.dataset.rootKey);
            if (action === 'activate-available') return this.activateAvailableSelector(actionNode.dataset.selector, actionNode.dataset.rootKey);
            if (action === 'activate-available-positive') return this.activateAvailableSelector(actionNode.dataset.selector, actionNode.dataset.rootKey, 'positive');
            if (action === 'activate-available-negative') return this.activateAvailableSelector(actionNode.dataset.selector, actionNode.dataset.rootKey, 'negative');
            if (action === 'insert-embedding-positive') return this.insertEmbeddingIntoTarget(actionNode.dataset.token, '#positive_prompt', 'positive prompt');
            if (action === 'insert-embedding-negative') return this.insertEmbeddingIntoTarget(actionNode.dataset.token, '#negative_prompt', 'negative prompt');
            if (action === 'pick-thumbnail-file') {
                const uploadInput = this.querySelector('[data-thumbnail-upload]');
                if (uploadInput && !uploadInput.disabled) uploadInput.click();
                return;
            }
            if (action === 'pick-import-file') {
                const uploadInput = this.querySelector('[data-import-upload]');
                if (uploadInput) uploadInput.click();
                return;
            }
            if (action === 'save-drawer') return this.saveDrawer();
            if (action === 'download-active') return this.queueActiveDownloads();
            if (action === 'clear-selection') return this.clearActiveSelection();
            if (action === 'refresh-browser') return this.refreshAllData();
        }

        handleInput(event) {
            if (!this.state.drawer) return;
            const input = event.target.closest('[data-field]');
            if (!input) return;
            const previousForm = { ...this.state.drawer.form };
            const focusState = this.captureFieldFocus(input);
            this.state.drawer.form[input.dataset.field] = input.value;
            this.state.drawer.form = this.normalizeDrawerForm(this.state.drawer.form, this.drawerRootKey());
            if (this.state.drawer.mode === 'add_model') {
                this.syncAddModelDerivedFields(previousForm);
                const rerenderFields = new Set(['source_input', 'name', 'display_name']);
                if (input.matches('select') || rerenderFields.has(input.dataset.field)) {
                    this.render();
                    this.restoreFieldFocus(focusState);
                }
                return;
            }
            if (input.matches('select')) {
                if (input.dataset.field === 'source_provider') {
                    const catalogIds = new Set(this.catalogOptionsForProvider(this.state.drawer.form.source_provider).map((catalog) => catalog.catalog_id));
                    if (this.state.drawer.form.target_catalog_id && !catalogIds.has(this.state.drawer.form.target_catalog_id)) {
                        this.state.drawer.form.target_catalog_id = '';
                    }
                }
                this.render();
            }
        }

        handleChange(event) {
            const importInput = event.target.closest('[data-import-upload]');
            if (importInput?.files?.length) {
                const [file] = importInput.files;
                importInput.value = '';
                this.loadImportCatalogFile(file);
                return;
            }
            const uploadInput = event.target.closest('[data-thumbnail-upload]');
            if (uploadInput?.files?.length) {
                const [file] = uploadInput.files;
                uploadInput.value = '';
                if (this.state.drawer?.mode === 'add_model') {
                    this.state.drawer.pendingThumbnailFile = file;
                    this.state.drawer.form.thumbnail_source_name = file.name;
                    this.render();
                    return;
                }
                this.uploadDrawerThumbnail(file);
                return;
            }
            this.handleInput(event);
        }

        render() {
            const loading = this.state.loadingTabs.has(this.state.activeTab) && !this.state.tabData[this.state.activeTab];
            const selectedCount = this.activeDownloadSelectors().length;
            const body = loading
                ? '<div class="nmb-empty">Loading models...</div>'
                : SECTION_KEYS.map((sectionKey) => this.renderSection(sectionKey)).join('');
            this.innerHTML = `
                <div class="nmb-shell ${this.state.drawer ? 'has-drawer' : ''}">
                    <div class="nmb-main">
                        <div class="nmb-top-tabs">${this.renderTopTabs()}</div>
                        <div class="nmb-sub-tabs">${this.renderSubTabs()}</div>
                        <div class="nmb-toolbar">
                            <div class="nmb-toolbar__actions">
                                <button type="button" class="nmb-action-btn" data-action="open-add-drawer" data-type="manage">Add Model</button>
                                <button type="button" class="nmb-action-btn" data-action="open-import-catalog" data-type="manage">Import Catalog</button>
                                <button type="button" class="nmb-action-btn" data-action="clear-selection" data-type="clear">Clear</button>
                                <button type="button" class="nmb-action-btn" data-action="refresh-browser" data-type="reload">Reload</button>
                                <button type="button" class="nmb-action-btn nmb-primary-action" data-action="download-active" data-type="download" ${selectedCount ? '' : 'disabled'}>Download Selected${selectedCount ? ` (${selectedCount})` : ''}</button>
                            </div>
                        </div>
                        ${this.state.status ? `<div class="nmb-status nmb-status--${escapeHtml(this.state.statusTone)}">${escapeHtml(this.state.status)}</div>` : ''}
                        ${this.renderJobs()}
                        <div class="nmb-sections">${body}</div>
                    </div>
                    ${this.renderDrawer()}
                </div>
            `;
            window.requestAnimationFrame(() => this.installDropTargets());
        }
    }

    if (!customElements.get('nex-model-browser')) {
        customElements.define('nex-model-browser', NexModelBrowser);
    }
})();

























