(function () {
    let panel = null;
    let imagesContainer = null;
    let lastImagesJson = '';
    let isDragging = false;
    let dragOffset = { x: 0, y: 0 };
    let pollInterval = null;
    let currentGimpQueue = [];
    let selectedImage = null;
    let currentCompareMap = {};
    let pendingRevealName = '';
    let latestImages = [];
    let activeImageDragCount = 0;
    let renderPendingAfterDrag = false;
    let selectedCompareNames = new Set();
    let isFetchInFlight = false;
    let fetchQueued = false;
    let fetchQueuedForceRender = false;
    let panelDirty = false;
    let dragGuardHandle = null;
    let markerPicker = null;
    let activeMarkerName = '';
    const markerIconOptions = [
        { id: 'star', glyph: '★', label: 'Star' },
        { id: 'flag', glyph: '⚑', label: 'Flag' },
        { id: 'pin', glyph: '◆', label: 'Pin' },
        { id: 'bookmark', glyph: '⬢', label: 'Bookmark' },
    ];
    const markerColorOptions = [
        { id: 'red', label: 'Red' },
        { id: 'amber', label: 'Amber' },
        { id: 'green', label: 'Green' },
        { id: 'blue', label: 'Blue' },
        { id: 'violet', label: 'Violet' },
        { id: 'gray', label: 'Gray' },
    ];
    const markerShapeOptions = [
        { id: 'star', glyph: '\u2605', label: 'Star' },
        { id: 'flag', glyph: '\u2691', label: 'Flag' },
        { id: 'circle', glyph: '\u25CF', label: 'Circle' },
        { id: 'triangle', glyph: '\u25B2', label: 'Triangle' },
    ];

    function isPanelVisible() {
        return !!(panel && panel.style.display !== 'none');
    }

    function isPanelMinimized() {
        return !!(panel && panel.classList.contains('minimized'));
    }

    function isCompareActive() {
        return Object.keys(currentCompareMap).length > 0;
    }

    function scheduleDragGuardReset() {
        if (dragGuardHandle) {
            window.clearTimeout(dragGuardHandle);
        }
        dragGuardHandle = window.setTimeout(() => {
            activeImageDragCount = 0;
            dragGuardHandle = null;
            refreshAfterDragIfNeeded();
        }, 2500);
    }

    function clearDragGuardReset() {
        if (!dragGuardHandle) {
            return;
        }
        window.clearTimeout(dragGuardHandle);
        dragGuardHandle = null;
    }

    function finalizeImageDrag() {
        activeImageDragCount = 0;
        clearDragGuardReset();
        refreshAfterDragIfNeeded();
    }

    function escapeSelector(value) {
        if (window.CSS && typeof window.CSS.escape === 'function') {
            return window.CSS.escape(value);
        }
        return String(value).replace(/([^a-zA-Z0-9_-])/g, '\\$1');
    }

    function getMarkerIconGlyph(iconId) {
        const option = markerIconOptions.find((item) => item.id === iconId);
        return option ? option.glyph : '★';
    }

    function getImageByName(name) {
        return latestImages.find((img) => img.name === name) || null;
    }

    function getMarkerShapeGlyph(iconId) {
        const legacyAlias = {
            pin: 'circle',
            bookmark: 'triangle',
        };
        const resolvedId = legacyAlias[iconId] || iconId;
        const option = markerShapeOptions.find((item) => item.id === resolvedId);
        return option ? option.glyph : '\u25CF';
    }

    function normalizeMarkerShapeId(iconId) {
        const legacyAlias = {
            pin: 'circle',
            bookmark: 'triangle',
        };
        const resolvedId = legacyAlias[iconId] || iconId;
        return markerShapeOptions.some((item) => item.id === resolvedId)
            ? resolvedId
            : markerShapeOptions[0].id;
    }

    function closeMarkerPicker() {
        if (!markerPicker) {
            return;
        }
        markerPicker.classList.remove('is-open');
        markerPicker.style.display = 'none';
        activeMarkerName = '';
    }

    function ensureMarkerPicker() {
        if (markerPicker) {
            return markerPicker;
        }

        markerPicker = document.createElement('div');
        markerPicker.className = 'staging-marker-picker';
        markerPicker.innerHTML = `
            <div class="staging-marker-picker__section">
                <div class="staging-marker-picker__label">Icon</div>
                <div class="staging-marker-picker__icons"></div>
            </div>
            <div class="staging-marker-picker__section">
                <div class="staging-marker-picker__label">Color</div>
                <div class="staging-marker-picker__colors"></div>
            </div>
            <div class="staging-marker-picker__section">
                <div class="staging-marker-picker__label">Label</div>
                <input class="staging-marker-picker__input" type="text" maxlength="48" placeholder="Optional note">
            </div>
            <div class="staging-marker-picker__actions">
                <button type="button" class="staging-marker-picker__action staging-marker-picker__action--save">Save</button>
                <button type="button" class="staging-marker-picker__action staging-marker-picker__action--clear">Clear</button>
            </div>
        `;

        const iconsContainer = markerPicker.querySelector('.staging-marker-picker__icons');
        markerShapeOptions.forEach((option) => {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'staging-marker-picker__icon';
            button.dataset.icon = option.id;
            button.title = option.label;
            button.textContent = option.glyph;
            button.addEventListener('click', () => {
                markerPicker.querySelectorAll('.staging-marker-picker__icon').forEach((el) => {
                    el.classList.toggle('is-active', el.dataset.icon === option.id);
                });
            });
            iconsContainer.appendChild(button);
        });

        const colorsContainer = markerPicker.querySelector('.staging-marker-picker__colors');
        markerColorOptions.forEach((option) => {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = `staging-marker-picker__color color-${option.id}`;
            button.dataset.color = option.id;
            button.title = option.label;
            button.addEventListener('click', () => {
                markerPicker.querySelectorAll('.staging-marker-picker__color').forEach((el) => {
                    el.classList.toggle('is-active', el.dataset.color === option.id);
                });
            });
            colorsContainer.appendChild(button);
        });

        markerPicker.querySelector('.staging-marker-picker__action--save').addEventListener('click', saveMarkerFromPicker);
        markerPicker.querySelector('.staging-marker-picker__action--clear').addEventListener('click', clearMarkerFromPicker);
        markerPicker.addEventListener('click', (event) => event.stopPropagation());
        document.body.appendChild(markerPicker);
        return markerPicker;
    }

    function openMarkerPicker(anchorElement, image) {
        if (!anchorElement || !image) {
            return;
        }
        const picker = ensureMarkerPicker();
        const marker = image.marker || {};
        const iconId = normalizeMarkerShapeId(marker.icon || markerShapeOptions[0].id);
        const colorId = marker.color || markerColorOptions[0].id;
        const label = marker.label || '';

        activeMarkerName = image.name;
        picker.querySelectorAll('.staging-marker-picker__icon').forEach((el) => {
            el.classList.toggle('is-active', el.dataset.icon === iconId);
        });
        picker.querySelectorAll('.staging-marker-picker__color').forEach((el) => {
            el.classList.toggle('is-active', el.dataset.color === colorId);
        });
        picker.querySelector('.staging-marker-picker__input').value = label;

        const rect = anchorElement.getBoundingClientRect();
        picker.style.display = 'block';
        picker.classList.add('is-open');
        const pickerRect = picker.getBoundingClientRect();
        const maxLeft = Math.max(12, window.innerWidth - pickerRect.width - 12);
        const preferredTop = rect.bottom + 8;
        const fallbackTop = rect.top - pickerRect.height - 8;
        picker.style.left = `${Math.min(maxLeft, Math.max(12, rect.left - 10))}px`;
        picker.style.top = `${fallbackTop >= 12 ? fallbackTop : Math.max(12, preferredTop)}px`;
    }

    function updateLocalMarker(name, marker) {
        latestImages = latestImages.map((img) => {
            if (img.name !== name) {
                return img;
            }
            return { ...img, marker: marker || null };
        });
        lastImagesJson = JSON.stringify({ images: latestImages, gimp_queue: currentGimpQueue });
    }

    async function persistMarker(name, marker) {
        const res = await fetch('/staging_api/marker', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ name, marker }),
        });
        const result = await res.json();
        if (result.status !== 'success') {
            throw new Error(result.detail || result.message || 'Failed to save marker');
        }
        updateLocalMarker(name, result.marker || null);
        if (isPanelVisible() && !isPanelMinimized()) {
            renderImages(latestImages);
        } else {
            panelDirty = true;
        }
        closeMarkerPicker();
    }

    async function saveMarkerFromPicker() {
        if (!activeMarkerName || !markerPicker) {
            return;
        }
        const activeIcon = markerPicker.querySelector('.staging-marker-picker__icon.is-active');
        const activeColor = markerPicker.querySelector('.staging-marker-picker__color.is-active');
        const labelInput = markerPicker.querySelector('.staging-marker-picker__input');

        if (!activeIcon || !activeColor) {
            return;
        }
        try {
            await persistMarker(activeMarkerName, {
                icon: activeIcon.dataset.icon,
                color: activeColor.dataset.color,
                label: labelInput.value.trim(),
            });
        } catch (error) {
            console.error('[Staging] Marker save error:', error);
        }
    }

    async function clearMarkerFromPicker() {
        if (!activeMarkerName) {
            return;
        }
        try {
            await persistMarker(activeMarkerName, null);
        } catch (error) {
            console.error('[Staging] Marker clear error:', error);
        }
    }

    function createPanel() {
        if (panel) return;

        panel = document.createElement('div');
        panel.id = 'floating-staging-panel';
        panel.className = 'floating-panel';
        panel.innerHTML = `
            <div class="panel-header" id="staging-panel-header">
                <span class="panel-title">Staging Area</span>
                <div class="panel-controls">
                    <button id="staging-panel-compare" class="staging-panel-action staging-panel-action--compare" title="Compare selected images" disabled>Compare</button>
                    <button id="staging-panel-refresh" class="staging-panel-action staging-panel-action--refresh" title="Refresh Now">Refresh</button>
                    <button id="staging-panel-clear" class="staging-panel-action staging-panel-action--clear" title="Clear All Staging">Clear</button>
                    <button id="staging-panel-toggle" class="panel-window-btn" title="Minimize">-</button>
                    <button id="staging-panel-close" class="panel-window-btn" title="Close Palette">x</button>
                </div>
            </div>
            <div class="panel-content">
                <div id="staging-images-grid" class="staging-grid">
                    <div class="empty-msg">Drop images here to stage</div>
                </div>
            </div>
            <div class="resize-handle" id="staging-panel-resize"></div>
        `;

        document.body.appendChild(panel);
        imagesContainer = panel.querySelector('#staging-images-grid');
        ensureMarkerPicker();

        // Snap Logic: default position at top-right, aligned with tab titles
        const snapToDefault = () => {
            panel.style.top = '60px'; // Raised from 100px to align with tabs
            panel.style.right = '20px';
            panel.style.left = 'auto'; // Clear drag left property
        };

        // Drag logic
        const header = panel.querySelector('#staging-panel-header');
        header.addEventListener('mousedown', startDragging);
        document.addEventListener('mousemove', (e) => {
            drag(e);
            resize(e);
        });
        document.addEventListener('mouseup', () => {
            stopDragging();
            stopResizing();
        });

        // Resize logic
        const resizeHandle = panel.querySelector('#staging-panel-resize');
        resizeHandle.addEventListener('mousedown', startResizing);

        // Controls
        panel.querySelector('#staging-panel-compare').addEventListener('click', openCompareFromSelection);
        panel.querySelector('#staging-panel-refresh').addEventListener('click', fetchImages);
        panel.querySelector('#staging-panel-clear').addEventListener('click', clearStaging);

        panel.querySelector('#staging-panel-toggle').addEventListener('click', () => {
            const isMinimized = panel.classList.toggle('minimized');
            if (isMinimized) {
                // Snap on minimize
                snapToDefault();
            }
        });

        panel.querySelector('#staging-panel-close').addEventListener('click', () => {
            panel.style.display = 'none';
        });

        // Drop zone for uploads
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            panel.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            }, false);
        });

        panel.addEventListener('dragover', () => panel.classList.add('drag-over'));
        panel.addEventListener('dragleave', () => panel.classList.remove('drag-over'));
        panel.addEventListener('drop', handleDrop);
        panel.addEventListener('click', closeMarkerPicker);
        document.addEventListener('click', closeMarkerPicker);
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') {
                closeMarkerPicker();
            }
        });

        // Load position and size from localStorage
        const pos = JSON.parse(localStorage.getItem('staging-panel-pos') || '{"top": "60px", "right": "20px"}');
        const size = JSON.parse(localStorage.getItem('staging-panel-size') || '{"width": "440px", "height": "auto"}');
        Object.assign(panel.style, pos);
        Object.assign(panel.style, size);

        fetchImages();
        pollInterval = setInterval(fetchImages, 3000);
        updateCompareButtonState();
    }

    function openPanel() {
        if (!panel) {
            return;
        }
        panel.style.display = 'flex';
        panel.classList.remove('minimized');
        const wasDirty = panelDirty;
        if (wasDirty) {
            panelDirty = false;
            renderImages(latestImages);
        }
        fetchImages(wasDirty || latestImages.length === 0);
    }

    function startDragging(e) {
        if (e.target.tagName === 'BUTTON') return;
        isDragging = true;
        const rect = panel.getBoundingClientRect();
        dragOffset = {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
        panel.style.right = 'auto';
        panel.style.left = rect.left + 'px';
        panel.style.top = rect.top + 'px';
        panel.classList.add('dragging');
        e.preventDefault();
    }

    function drag(e) {
        if (!isDragging) return;
        panel.style.left = (e.clientX - dragOffset.x) + 'px';
        panel.style.top = (e.clientY - dragOffset.y) + 'px';
    }

    function stopDragging() {
        if (!isDragging) return;
        isDragging = false;
        panel.classList.remove('dragging');
        // Save position
        localStorage.setItem('staging-panel-pos', JSON.stringify({
            top: panel.style.top,
            left: panel.style.left
        }));
    }

    let isResizing = false;
    let resizeStartSize = { w: 0, h: 0 };
    let resizeStartPos = { x: 0, y: 0 };

    function startResizing(e) {
        isResizing = true;
        resizeStartSize = {
            w: panel.offsetWidth,
            h: panel.offsetHeight
        };
        resizeStartPos = {
            x: e.clientX,
            y: e.clientY
        };
        panel.classList.add('resizing');
        e.preventDefault();
        e.stopPropagation();
    }

    function resize(e) {
        if (!isResizing) return;
        const deltaX = e.clientX - resizeStartPos.x;
        const deltaY = e.clientY - resizeStartPos.y;
        panel.style.width = (resizeStartSize.w + deltaX) + 'px';
        panel.style.height = (resizeStartSize.h + deltaY) + 'px';
    }

    function stopResizing() {
        if (!isResizing) return;
        isResizing = false;
        panel.classList.remove('resizing');
        // Save size
        localStorage.setItem('staging-panel-size', JSON.stringify({
            width: panel.style.width,
            height: panel.style.height
        }));
    }

    async function fetchImages(forceRender = false) {
        if (document.visibilityState === 'hidden') {
            return;
        }
        if (isFetchInFlight) {
            fetchQueued = true;
            fetchQueuedForceRender = fetchQueuedForceRender || forceRender;
            return;
        }
        try {
            isFetchInFlight = true;
            const res = await fetch('/staging_api/images');
            const data = await res.json();
            currentGimpQueue = Array.isArray(data.gimp_queue) ? data.gimp_queue : [];
            latestImages = Array.isArray(data.images) ? data.images : [];
            const json = JSON.stringify({ images: data.images, gimp_queue: currentGimpQueue });
            if (!forceRender && json === lastImagesJson) return; // No change
            lastImagesJson = json;

            if (activeImageDragCount > 0) {
                renderPendingAfterDrag = true;
                return;
            }
            if (!isPanelVisible() || isPanelMinimized()) {
                panelDirty = true;
                return;
            }
            renderImages(latestImages);
        } catch (e) {
            console.error('[Staging] Fetch error:', e);
        } finally {
            isFetchInFlight = false;
            if (fetchQueued) {
                const queuedForceRender = fetchQueuedForceRender;
                fetchQueued = false;
                fetchQueuedForceRender = false;
                window.setTimeout(() => fetchImages(queuedForceRender), 0);
            }
        }
    }

    function refreshAfterDragIfNeeded() {
        if (activeImageDragCount > 0 || !renderPendingAfterDrag) {
            return;
        }
        renderPendingAfterDrag = false;
        if (!isPanelVisible() || isPanelMinimized()) {
            panelDirty = true;
            return;
        }
        renderImages(latestImages);
    }

    function updateCompareBadges() {
        if (!panel) {
            return;
        }
        panel.querySelectorAll('.staging-item').forEach((item) => {
            const imageName = item.dataset.imageName || '';
            const compareBadge = currentCompareMap[imageName] || '';
            item.classList.toggle('is-in-compare', !!compareBadge);
            if (compareBadge) {
                item.dataset.compareSlot = compareBadge;
            } else {
                delete item.dataset.compareSlot;
            }

            let badge = item.querySelector('.staging-compare-badge');
            if (compareBadge) {
                if (!badge) {
                    badge = document.createElement('div');
                    badge.className = 'staging-compare-badge';
                    item.appendChild(badge);
                }
                badge.textContent = compareBadge;
            } else if (badge) {
                badge.remove();
            }
        });
    }

    function appendMarkerBadge(item, marker) {
        if (!marker || !marker.icon || !marker.color) {
            return;
        }
        const badge = document.createElement('div');
        badge.className = `staging-marker-badge color-${marker.color}`;
        badge.title = marker.label ? `${marker.icon}: ${marker.label}` : `Marker: ${marker.icon}`;

        const icon = document.createElement('span');
        icon.className = 'staging-marker-badge__icon';
        icon.textContent = getMarkerShapeGlyph(marker.icon);
        badge.appendChild(icon);

        if (marker.label) {
            const label = document.createElement('span');
            label.className = 'staging-marker-badge__label';
            label.textContent = marker.label;
            badge.appendChild(label);
        }

        item.appendChild(badge);
    }

    function syncSelectedCompareNames(images) {
        const names = new Set((images || []).map((img) => img.name));
        const nextSelected = new Set();
        selectedCompareNames.forEach((name) => {
            if (names.has(name)) {
                nextSelected.add(name);
            }
        });
        selectedCompareNames = nextSelected;
        updateCompareButtonState();
    }

    function updateCompareButtonState() {
        const button = panel ? panel.querySelector('#staging-panel-compare') : null;
        if (!button) {
            return;
        }
        const count = selectedCompareNames.size;
        button.disabled = count === 0;
        button.textContent = count > 0 ? `Compare ${count}` : 'Compare';
        button.title = count > 0 ? `Open compare viewer for ${count} selected image${count === 1 ? '' : 's'}` : 'Select staged images to compare';
    }

    function toggleCompareSelection(name) {
        if (!name) {
            return;
        }
        if (selectedCompareNames.has(name)) {
            selectedCompareNames.delete(name);
        } else {
            if (selectedCompareNames.size >= 4) {
                return;
            }
            selectedCompareNames.add(name);
        }
        panel?.querySelector(`.staging-item[data-image-name="${escapeSelector(name)}"]`)?.classList.toggle('is-selected', selectedCompareNames.has(name));
        updateCompareButtonState();
    }

    function getSelectedCompareItems() {
        const selectedNames = Array.from(selectedCompareNames);
        return selectedNames
            .map((name) => latestImages.find((img) => img.name === name))
            .filter(Boolean)
            .slice(0, 4)
            .map((img) => ({
                stagingName: img.name,
                relativeUrl: img.url,
                absoluteUrl: window.location.origin + img.url,
                filename: img.name,
            }));
    }

    function openCompareFromSelection() {
        const items = getSelectedCompareItems();
        if (!items.length) {
            return;
        }
        window.dispatchEvent(new CustomEvent('nex-compare:open', {
            detail: { items },
        }));
    }

    function clearCompareSelection() {
        if (selectedCompareNames.size === 0) {
            return;
        }
        selectedCompareNames = new Set();
        panel?.querySelectorAll('.staging-item.is-selected').forEach((item) => {
            item.classList.remove('is-selected');
        });
        updateCompareButtonState();
    }

    function renderImages(images) {
        syncSelectedCompareNames(images);
        if (images.length === 0) {
            imagesContainer.innerHTML = '<div class="empty-msg">Drop images here to stage</div>';
            return;
        }

        imagesContainer.innerHTML = '';
        images.forEach(img => {
            const item = document.createElement('div');
            item.className = 'staging-item';
            item.dataset.imageName = img.name;
            item.classList.toggle('is-selected', selectedCompareNames.has(img.name));
            if (currentGimpQueue.includes(img.name)) {
                item.classList.add('gimp-targeted');
            }
            const compareBadge = currentCompareMap[img.name];
            if (compareBadge) {
                item.classList.add('is-in-compare');
                item.dataset.compareSlot = compareBadge;
            }

            // Standard img (NOT draggable=true directly to prevent browser drag feedback lag on huge source images)
            const imgEl = document.createElement('img');
            imgEl.src = img.url;
            imgEl.alt = img.name;
            imgEl.draggable = false;
            imgEl.loading = 'lazy';
            imgEl.decoding = 'async';
            imgEl.title = 'Drag to slots';

            // Make the wrapper container draggable to allow dragging the small rendered thumbnail
            // rather than the high-resolution source image, preventing browser freeze/drag cancels.
            item.draggable = true;

            // Critical for dragging into Gradio slots: set absolute URL in dataTransfer
            item.addEventListener('dragstart', (e) => {
                if (e.target.closest('.item-actions') || e.target.tagName === 'BUTTON') {
                    e.preventDefault();
                    return;
                }
                activeImageDragCount += 1;
                scheduleDragGuardReset();
                const absoluteUrl = window.location.origin + img.url;
                const payload = JSON.stringify({
                    kind: 'nex-image-source',
                    sourceKind: 'staging',
                    absoluteUrl: absoluteUrl,
                    relativeUrl: img.url,
                    stagingName: img.name,
                });
                e.dataTransfer.setData('text/plain', absoluteUrl);
                e.dataTransfer.setData('text/uri-list', absoluteUrl);
                e.dataTransfer.setData('application/json', payload);
                e.dataTransfer.setData('fooocus/staging-internal', 'true'); // Flag to prevent self-drop
            });
            item.addEventListener('dragend', () => {
                finalizeImageDrag();
            });

            item.appendChild(imgEl);
            appendMarkerBadge(item, img.marker);

            if (compareBadge) {
                const badge = document.createElement('div');
                badge.className = 'staging-compare-badge';
                badge.textContent = compareBadge;
                item.appendChild(badge);
            }

            // Action buttons container
            const actionsContainer = document.createElement('div');
            actionsContainer.className = 'item-actions';

            const markerBtn = document.createElement('button');
            markerBtn.className = 'item-action-btn btn-marker';
            markerBtn.innerHTML = img.marker ? getMarkerShapeGlyph(img.marker.icon) : 'M';
            markerBtn.title = img.marker && img.marker.label ? `Edit marker: ${img.marker.label}` : 'Set marker';
            markerBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                openMarkerPicker(markerBtn, img);
            });
            actionsContainer.appendChild(markerBtn);

            // GIMP button
            const gimpBtn = document.createElement('button');
            gimpBtn.className = 'item-action-btn btn-gimp';
            gimpBtn.innerHTML = 'G';
            gimpBtn.title = 'Queue for GIMP import';
            gimpBtn.onclick = async (e) => {
                e.stopPropagation();
                try {
                    const res = await fetch(`/staging_api/gimp_target?name=${encodeURIComponent(img.name)}`, {
                        method: 'POST'
                    });
                    const result = await res.json();
                    if (result.status === 'success') {
                        currentGimpQueue = Array.isArray(result.queue) ? result.queue : [];
                        panel.querySelectorAll('.staging-item').forEach(el => {
                            const isQueued = currentGimpQueue.includes(el.dataset.imageName);
                            el.classList.toggle('gimp-targeted', isQueued);
                        });
                    }
                } catch (err) {
                    console.error('[Staging] GIMP Target error:', err);
                }
            };
            actionsContainer.appendChild(gimpBtn);

            // Delete button
            const delBtn = document.createElement('button');
            delBtn.className = 'item-action-btn btn-delete';
            delBtn.innerHTML = 'X';
            delBtn.title = 'Remove from staging';
            delBtn.onclick = (e) => {
                e.stopPropagation();
                deleteImage(img.name);
            };
            actionsContainer.appendChild(delBtn);

            item.appendChild(actionsContainer);
            item.addEventListener('click', (event) => {
                if (event.target.closest('.item-actions')) {
                    return;
                }
                toggleCompareSelection(img.name);
            });

            imagesContainer.appendChild(item);
        });

        if (pendingRevealName) {
            flashRevealTarget(pendingRevealName);
        }
        updateCompareButtonState();
    }

    function flashRevealTarget(name) {
        if (!panel || !name) {
            return;
        }
        const target = panel.querySelector(`.staging-item[data-image-name="${escapeSelector(name)}"]`);
        if (!target) {
            return;
        }
        pendingRevealName = '';
        target.classList.add('is-revealed');
        target.scrollIntoView({ behavior: 'smooth', block: 'center' });
        window.setTimeout(() => target.classList.remove('is-revealed'), 1800);
    }

    async function deleteImage(name) {
        if (!confirm('Remove this image from staging?')) return;
        try {
            const res = await fetch(`/staging_api/delete?name=${encodeURIComponent(name)}`, {
                method: 'DELETE'
            });
            const result = await res.json();
            if (result.status === 'success') {
                selectedCompareNames.delete(name);
                updateCompareButtonState();
                fetchImages();
            }
        } catch (e) {
            console.error('[Staging] Delete error:', e);
        }
    }

    async function clearStaging() {
        if (!confirm('Clear ALL images from staging?')) return;
        try {
            const res = await fetch('/staging_api/clear', {
                method: 'POST'
            });
            const result = await res.json();
            if (result.status === 'success') {
                selectedCompareNames = new Set();
                window.dispatchEvent(new CustomEvent('nex-compare:close-request'));
                updateCompareButtonState();
                fetchImages();
            }
        } catch (e) {
            console.error('[Staging] Clear error:', e);
        }
    }

    async function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        panel.classList.remove('drag-over');

        const dataTransfer = e.dataTransfer;

        // Prevent duplicate upload if dragged from within the staging panel
        if (dataTransfer.types.includes('fooocus/staging-internal')) {
            return;
        }

        const files = dataTransfer.files;
        const html = dataTransfer.getData('text/html');
        const plain = dataTransfer.getData('text/plain');

        let url = null;
        if (html) {
            const doc = new DOMParser().parseFromString(html, 'text/html');
            const img = doc.querySelector('img');
            if (img && img.src) url = img.src;
        }

        if (!url && plain) {
            if (plain.startsWith('http') || plain.startsWith('file') || plain.startsWith('data:image')) {
                url = plain;
            }
        }

        if (files.length > 0) {
            for (let file of files) {
                const formData = new FormData();
                formData.append('file', file);
                await uploadImage(formData);
            }
        } else if (url) {
            const formData = new FormData();
            formData.append('url', url);
            await uploadImage(formData);
        } else {
            console.warn('[Staging] No supported data found in drop.');
        }
    }

    async function uploadImage(formData) {
        try {
            const res = await fetch('/staging_api/upload', {
                method: 'POST',
                body: formData
            });
            const result = await res.json();
            if (result.status === 'success') {
                fetchImages(true);
            }
        } catch (e) {
            console.error('[Staging] Upload error:', e);
        }
    }

    // Initialize when Gradio is ready
    function init() {
        if (window.gradioApp) {
            createPanel();

            window.addEventListener('nex-compare:state-change', (event) => {
                const nextCompareMap = (event && event.detail && event.detail.stagingMap) || {};
                const wasActive = isCompareActive();
                currentCompareMap = nextCompareMap;
                const nowActive = isCompareActive();
                if (!wasActive && !nowActive) {
                    return;
                }
                if (latestImages.length > 0 && isPanelVisible()) {
                    updateCompareBadges();
                }
            });

            window.addEventListener('nex-staging:open-request', () => {
                openPanel();
            });

            window.addEventListener('nex-staging:reveal-request', (event) => {
                const name = event && event.detail && event.detail.name;
                if (!name) {
                    return;
                }
                pendingRevealName = name;
                openPanel();
                fetchImages(true);
                window.setTimeout(() => flashRevealTarget(name), 180);
            });

            window.addEventListener('nex-compare:closed', () => {
                clearCompareSelection();
            });

            window.addEventListener('drop', finalizeImageDrag);
            window.addEventListener('dragend', finalizeImageDrag);
            window.addEventListener('blur', finalizeImageDrag);
            document.addEventListener('visibilitychange', () => {
                if (document.visibilityState === 'visible' && isPanelVisible()) {
                    fetchImages(panelDirty);
                }
            });

            // Global listener for the launcher button (survives Gradio DOM swaps)
            document.addEventListener('click', (e) => {
                const launcher = e.target.closest('#staging-panel-launcher');
                if (launcher && panel) {
                    openPanel();
                    panel.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    // Pop effect
                    panel.style.transform = 'scale(1.05)';
                    setTimeout(() => panel.style.transform = 'scale(1)', 200);
                }
            });
        } else {
            setTimeout(init, 500);
        }
    }

    init();
})();


