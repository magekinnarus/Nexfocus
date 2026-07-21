(() => {
    const MIN_ZOOM = 1;
    const MAX_ZOOM = 8;
    const ZOOM_STEP = 0.16;
    const VIEWPORT_MARGIN = 12;
    const MIN_COMPACT_CONTENT_WIDTH = 360;
    const MIN_COMPACT_CONTENT_HEIGHT = 180;

    const state = {
        panel: null,
        header: null,
        content: null,
        modeButton: null,
        closeButton: null,
        resizeHandle: null,
        items: [],
        mode: "full",
        camera: {
            zoom: 1,
            centerX: 0.5,
            centerY: 0.5,
        },
        compactRect: null,
        dragState: null,
        resizeState: null,
        panState: null,
        syncHandle: null,
        syncTimeout: null,
        settleHandle: null,
    };

    function clamp(value, min, max) {
        return Math.min(max, Math.max(min, value));
    }

    function toAbsoluteUrl(value) {
        if (!value) {
            return "";
        }
        try {
            return new URL(String(value), window.location.origin).toString();
        } catch (error) {
            return String(value);
        }
    }

    function toRelativeUrl(value) {
        if (!value) {
            return "";
        }
        try {
            const absolute = new URL(String(value), window.location.origin);
            if (absolute.origin === window.location.origin) {
                return `${absolute.pathname}${absolute.search}`;
            }
        } catch (error) {
            return String(value);
        }
        return String(value);
    }

    function getFilenameFromUrl(value) {
        const raw = String(value || "").trim();
        if (!raw) {
            return "compare_image.png";
        }
        const relative = toRelativeUrl(raw);
        if (relative.startsWith("/file=")) {
            const decoded = decodeURIComponent(relative.slice("/file=".length));
            const parts = decoded.split(/[\\/]+/).filter(Boolean);
            return parts.length ? parts[parts.length - 1] : "compare_image.png";
        }
        const parts = relative.split("/").filter(Boolean);
        return decodeURIComponent(parts.length ? parts[parts.length - 1].split("?")[0] : "compare_image.png");
    }

    function getHeaderHeight() {
        return state.header?.offsetHeight || 40;
    }

    function getViewportSize() {
        return {
            width: window.innerWidth || document.documentElement.clientWidth || 1,
            height: window.innerHeight || document.documentElement.clientHeight || 1,
        };
    }

    function getFullContentAspectRatio() {
        const viewport = getViewportSize();
        const contentHeight = Math.max(1, viewport.height - getHeaderHeight());
        return viewport.width / contentHeight;
    }

    function createDefaultCompactRect() {
        const aspectRatio = Math.max(0.1, getFullContentAspectRatio());
        const viewport = getViewportSize();
        const maxContentHeight = Math.max(
            MIN_COMPACT_CONTENT_HEIGHT,
            Math.min(
                viewport.height - (VIEWPORT_MARGIN * 2) - getHeaderHeight(),
                (viewport.width - (VIEWPORT_MARGIN * 2)) / aspectRatio,
            ),
        );
        const preferredContentHeight = clamp(
            Math.min(320, viewport.height * 0.32),
            MIN_COMPACT_CONTENT_HEIGHT,
            maxContentHeight,
        );
        return {
            left: 24,
            top: 88,
            contentHeight: preferredContentHeight,
        };
    }

    function clampCompactContentHeight(contentHeight) {
        const aspectRatio = Math.max(0.1, getFullContentAspectRatio());
        const viewport = getViewportSize();
        const maxHeight = Math.max(
            MIN_COMPACT_CONTENT_HEIGHT,
            Math.min(
                viewport.height - (VIEWPORT_MARGIN * 2) - getHeaderHeight(),
                (viewport.width - (VIEWPORT_MARGIN * 2)) / aspectRatio,
            ),
        );
        const minHeight = Math.max(
            MIN_COMPACT_CONTENT_HEIGHT,
            MIN_COMPACT_CONTENT_WIDTH / aspectRatio,
        );
        return clamp(contentHeight, minHeight, maxHeight);
    }

    function applyCompactRect(rect) {
        if (!state.panel) {
            return;
        }

        const nextRect = rect || state.compactRect || createDefaultCompactRect();
        const aspectRatio = Math.max(0.1, getFullContentAspectRatio());
        const contentHeight = clampCompactContentHeight(nextRect.contentHeight);
        const contentWidth = contentHeight * aspectRatio;
        const panelWidth = contentWidth;
        const panelHeight = contentHeight + getHeaderHeight();
        const viewport = getViewportSize();
        const left = clamp(
            nextRect.left,
            VIEWPORT_MARGIN,
            Math.max(VIEWPORT_MARGIN, viewport.width - panelWidth - VIEWPORT_MARGIN),
        );
        const top = clamp(
            nextRect.top,
            VIEWPORT_MARGIN,
            Math.max(VIEWPORT_MARGIN, viewport.height - panelHeight - VIEWPORT_MARGIN),
        );

        state.compactRect = {
            left,
            top,
            contentHeight,
        };

        state.panel.style.left = `${left}px`;
        state.panel.style.top = `${top}px`;
        state.panel.style.width = `${panelWidth}px`;
        state.panel.style.height = `${panelHeight}px`;
        state.panel.style.right = "auto";
        state.panel.style.bottom = "auto";
    }

    function ensurePanel() {
        if (state.panel) {
            return;
        }

        const panel = document.createElement("div");
        panel.id = "nex-compare-overlay";
        panel.className = "floating-panel nex-compare-overlay is-hidden";
        panel.innerHTML = `
            <div class="panel-header nex-compare-overlay__header">
                <span class="panel-title">Compare Viewer</span>
                <div class="panel-controls">
                    <button id="nex-compare-mode" class="panel-window-btn" title="Switch to windowed view">[]</button>
                    <button id="nex-compare-close" class="panel-window-btn" title="Close">x</button>
                </div>
            </div>
            <div class="panel-content nex-compare-overlay__content">
                <div class="nco-empty">Select up to 4 staged images, then press Compare.</div>
            </div>
            <div class="resize-handle nex-compare-overlay__resize"></div>
        `;

        document.body.appendChild(panel);
        state.panel = panel;
        state.header = panel.querySelector(".nex-compare-overlay__header");
        state.content = panel.querySelector(".nex-compare-overlay__content");
        state.modeButton = panel.querySelector("#nex-compare-mode");
        state.closeButton = panel.querySelector("#nex-compare-close");
        state.resizeHandle = panel.querySelector(".nex-compare-overlay__resize");

        state.modeButton.addEventListener("click", toggleMode);
        state.closeButton.addEventListener("click", closeCompare);
        state.resizeHandle.addEventListener("mousedown", startResize);
        state.header.addEventListener("mousedown", startDrag);
        panel.addEventListener("wheel", handleWheel, { passive: false });
        panel.addEventListener("mousedown", startPan);
        document.addEventListener("mousemove", handlePointerMove);
        document.addEventListener("mouseup", handlePointerUp);
        document.addEventListener("keydown", handleKeyDown);
        window.addEventListener("resize", handleWindowResize);
    }

    function resetCamera() {
        state.camera.zoom = 1;
        state.camera.centerX = 0.5;
        state.camera.centerY = 0.5;
    }

    function normalizeItems(rawItems) {
        return Array.isArray(rawItems)
            ? rawItems.slice(0, 4).map((item, index) => ({
                slotLabel: `C${index + 1}`,
                stagingName: item.stagingName || "",
                filename: item.filename || getFilenameFromUrl(item.absoluteUrl || item.relativeUrl || item.url || ""),
                absoluteUrl: toAbsoluteUrl(item.absoluteUrl || item.relativeUrl || item.url || ""),
            })).filter((item) => item.absoluteUrl)
            : [];
    }

    function openCompare(rawItems) {
        ensurePanel();
        state.items = normalizeItems(rawItems);
        if (!state.items.length) {
            closeCompare();
            return;
        }

        resetCamera();
        state.mode = "full";
        state.compactRect = state.compactRect || createDefaultCompactRect();
        state.panel.classList.remove("is-hidden");
        applyWindowMode();
        render();
        announceCompareState();
    }

    function closeCompare() {
        ensurePanel();
        state.items = [];
        state.dragState = null;
        state.resizeState = null;
        if (state.panState) {
            state.panState.viewport?.classList.remove("is-panning");
        }
        state.panState = null;
        state.panel.classList.add("is-hidden");
        state.panel.classList.remove("dragging", "resizing", "is-compact");
        state.content.innerHTML = `<div class="nco-empty">Select up to 4 staged images, then press Compare.</div>`;
        resetCamera();
        announceCompareState();
        window.dispatchEvent(new CustomEvent("nex-compare:closed"));
    }

    function applyWindowMode() {
        if (!state.panel) {
            return;
        }

        if (state.mode === "compact") {
            state.panel.classList.add("is-compact");
            applyCompactRect(state.compactRect || createDefaultCompactRect());
            state.modeButton.title = "Expand to full window";
            queueSync();
            return;
        }

        state.panel.classList.remove("is-compact");
        state.panel.style.top = "0";
        state.panel.style.left = "0";
        state.panel.style.width = "100vw";
        state.panel.style.height = "100vh";
        state.panel.style.right = "auto";
        state.panel.style.bottom = "auto";
        state.modeButton.title = "Switch to windowed view";
        queueSync();
    }

    function queueSettledSync() {
        if (state.settleHandle) {
            clearTimeout(state.settleHandle);
            state.settleHandle = null;
        }
        state.settleHandle = window.setTimeout(() => {
            state.settleHandle = null;
            queueSync();
        }, 120);
    }

    function toggleMode() {
        if (!state.panel || state.panel.classList.contains("is-hidden")) {
            return;
        }
        if (state.mode === "full") {
            const currentRect = state.panel.getBoundingClientRect();
            const contentHeight = clampCompactContentHeight(
                Math.max(MIN_COMPACT_CONTENT_HEIGHT, Math.min(currentRect.height - getHeaderHeight(), 320)),
            );
            state.compactRect = state.compactRect || {
                left: 24,
                top: 88,
                contentHeight,
            };
            state.mode = "compact";
        } else {
            const rect = state.panel.getBoundingClientRect();
            const contentHeight = clampCompactContentHeight(rect.height - getHeaderHeight());
            state.compactRect = {
                left: rect.left,
                top: rect.top,
                contentHeight,
            };
            state.mode = "full";
        }
        applyWindowMode();
        queueSettledSync();
    }

    function render() {
        if (!state.content) {
            return;
        }

        if (!state.items.length) {
            state.content.innerHTML = `<div class="nco-empty">Select up to 4 staged images, then press Compare.</div>`;
            return;
        }

        state.content.innerHTML = `
            <div class="nco-grid" style="--nco-cols:${state.items.length};">
                ${state.items.map((item, index) => `
                    <article class="nco-cell" data-slot-index="${index}">
                        <span class="nco-label">${item.slotLabel}</span>
                        <div class="nco-viewport" data-slot-index="${index}">
                            <img
                                class="nco-image"
                                data-slot-index="${index}"
                                src="${item.absoluteUrl}"
                                alt="${item.filename}"
                                draggable="false"
                            >
                        </div>
                    </article>
                `).join("")}
            </div>
        `;

        state.items.forEach((item, index) => {
            const image = getImage(index);
            if (image) {
                image.addEventListener("load", queueSync, { once: true });
            }
        });

        queueSync();
    }

    function getViewport(index) {
        return state.panel?.querySelector(`.nco-viewport[data-slot-index="${index}"]`) || null;
    }

    function getImage(index) {
        return state.panel?.querySelector(`.nco-image[data-slot-index="${index}"]`) || null;
    }

    function fitContain(naturalWidth, naturalHeight, viewportWidth, viewportHeight) {
        if (!naturalWidth || !naturalHeight || !viewportWidth || !viewportHeight) {
            return null;
        }
        const imageRatio = naturalWidth / naturalHeight;
        const viewportRatio = viewportWidth / viewportHeight;
        if (imageRatio > viewportRatio) {
            const width = viewportWidth;
            return {
                width,
                height: width / imageRatio,
            };
        }
        const height = viewportHeight;
        return {
            width: height * imageRatio,
            height,
        };
    }

    function getMetrics(index, cameraOverride = state.camera) {
        const viewport = getViewport(index);
        const image = getImage(index);
        if (!viewport || !image) {
            return null;
        }

        const viewportWidth = viewport.clientWidth || 0;
        const viewportHeight = viewport.clientHeight || 0;
        const naturalWidth = image.naturalWidth || 0;
        const naturalHeight = image.naturalHeight || 0;
        const fitted = fitContain(naturalWidth, naturalHeight, viewportWidth, viewportHeight);
        if (!fitted) {
            return null;
        }

        const zoom = clamp(cameraOverride.zoom, MIN_ZOOM, MAX_ZOOM);
        const displayWidth = fitted.width * zoom;
        const displayHeight = fitted.height * zoom;
        const minCenterX = displayWidth <= viewportWidth ? 0.5 : viewportWidth / (2 * displayWidth);
        const maxCenterX = displayWidth <= viewportWidth ? 0.5 : 1 - minCenterX;
        const minCenterY = displayHeight <= viewportHeight ? 0.5 : viewportHeight / (2 * displayHeight);
        const maxCenterY = displayHeight <= viewportHeight ? 0.5 : 1 - minCenterY;
        const centerX = clamp(cameraOverride.centerX, minCenterX, maxCenterX);
        const centerY = clamp(cameraOverride.centerY, minCenterY, maxCenterY);
        const left = displayWidth <= viewportWidth
            ? (viewportWidth - displayWidth) / 2
            : (viewportWidth / 2) - (centerX * displayWidth);
        const top = displayHeight <= viewportHeight
            ? (viewportHeight - displayHeight) / 2
            : (viewportHeight / 2) - (centerY * displayHeight);

        return {
            viewport,
            image,
            viewportWidth,
            viewportHeight,
            displayWidth,
            displayHeight,
            minCenterX,
            maxCenterX,
            minCenterY,
            maxCenterY,
            centerX,
            centerY,
            left,
            top,
        };
    }

    function clampCameraFromReference(index) {
        const metrics = getMetrics(index);
        if (!metrics) {
            return;
        }
        state.camera.centerX = metrics.centerX;
        state.camera.centerY = metrics.centerY;
    }

    function applySlotLayout(index) {
        const cell = state.panel?.querySelector(`.nco-cell[data-slot-index="${index}"]`) || null;
        const metrics = getMetrics(index);
        if (!cell || !metrics) {
            return;
        }

        metrics.image.style.width = `${metrics.displayWidth}px`;
        metrics.image.style.height = `${metrics.displayHeight}px`;
        metrics.image.style.left = `${metrics.left}px`;
        metrics.image.style.top = `${metrics.top}px`;
        cell.classList.toggle("is-zoomed", state.camera.zoom > 1.0001);
    }

    function syncAll() {
        if (!state.items.length) {
            return;
        }
        clampCameraFromReference(0);
        state.items.forEach((item, index) => applySlotLayout(index));
    }

    function queueSync() {
        if (state.syncHandle) {
            cancelAnimationFrame(state.syncHandle);
        }
        if (state.syncTimeout) {
            clearTimeout(state.syncTimeout);
            state.syncTimeout = null;
        }

        state.syncHandle = requestAnimationFrame(() => {
            state.syncHandle = null;
            syncAll();
            state.syncTimeout = window.setTimeout(() => {
                state.syncTimeout = null;
                syncAll();
            }, 32);
        });
    }

    function handleWheel(event) {
        if (!state.items.length || !event.ctrlKey) {
            return;
        }

        const viewport = event.target.closest(".nco-viewport");
        if (!viewport || !state.panel?.contains(viewport)) {
            return;
        }

        const slotIndex = Number(viewport.dataset.slotIndex || -1);
        const currentMetrics = getMetrics(slotIndex);
        if (!currentMetrics) {
            return;
        }

        event.preventDefault();

        const nextZoom = clamp(
            state.camera.zoom + (event.deltaY < 0 ? ZOOM_STEP : -ZOOM_STEP),
            MIN_ZOOM,
            MAX_ZOOM,
        );
        if (Math.abs(nextZoom - state.camera.zoom) < 0.0001) {
            return;
        }

        const rect = viewport.getBoundingClientRect();
        const localX = event.clientX - rect.left;
        const localY = event.clientY - rect.top;
        const focusX = currentMetrics.displayWidth > 0
            ? clamp((localX - currentMetrics.left) / currentMetrics.displayWidth, 0, 1)
            : 0.5;
        const focusY = currentMetrics.displayHeight > 0
            ? clamp((localY - currentMetrics.top) / currentMetrics.displayHeight, 0, 1)
            : 0.5;

        if (nextZoom <= 1.0001) {
            resetCamera();
            queueSync();
            return;
        }

        const nextMetrics = getMetrics(slotIndex, {
            zoom: nextZoom,
            centerX: state.camera.centerX,
            centerY: state.camera.centerY,
        });
        if (!nextMetrics) {
            return;
        }

        state.camera.zoom = nextZoom;
        state.camera.centerX = clamp(
            focusX - ((localX - (nextMetrics.viewportWidth / 2)) / nextMetrics.displayWidth),
            0,
            1,
        );
        state.camera.centerY = clamp(
            focusY - ((localY - (nextMetrics.viewportHeight / 2)) / nextMetrics.displayHeight),
            0,
            1,
        );

        queueSync();
    }

    function startPan(event) {
        if (!state.items.length || !event.ctrlKey || event.button !== 0 || state.camera.zoom <= 1.0001) {
            return;
        }

        const viewport = event.target.closest(".nco-viewport");
        if (!viewport || !state.panel?.contains(viewport)) {
            return;
        }

        const slotIndex = Number(viewport.dataset.slotIndex || -1);
        const metrics = getMetrics(slotIndex);
        if (!metrics) {
            return;
        }

        state.panState = {
            viewport,
            slotIndex,
            startX: event.clientX,
            startY: event.clientY,
            startCenterX: state.camera.centerX,
            startCenterY: state.camera.centerY,
            displayWidth: metrics.displayWidth,
            displayHeight: metrics.displayHeight,
        };
        viewport.classList.add("is-panning");
        event.preventDefault();
    }

    function startDrag(event) {
        if (!state.panel || state.mode !== "compact" || event.button !== 0) {
            return;
        }
        if (event.target.closest(".panel-controls")) {
            return;
        }

        const rect = state.panel.getBoundingClientRect();
        state.dragState = {
            offsetX: event.clientX - rect.left,
            offsetY: event.clientY - rect.top,
        };
        state.panel.classList.add("dragging");
        event.preventDefault();
    }

    function startResize(event) {
        if (!state.panel || state.mode !== "compact" || event.button !== 0) {
            return;
        }

        const rect = state.panel.getBoundingClientRect();
        state.resizeState = {
            startX: event.clientX,
            startY: event.clientY,
            left: rect.left,
            top: rect.top,
            startContentHeight: clampCompactContentHeight(rect.height - getHeaderHeight()),
            aspectRatio: Math.max(0.1, getFullContentAspectRatio()),
        };
        state.panel.classList.add("resizing");
        event.preventDefault();
        event.stopPropagation();
    }

    function handlePointerMove(event) {
        if (state.dragState && state.mode === "compact") {
            applyCompactRect({
                left: event.clientX - state.dragState.offsetX,
                top: event.clientY - state.dragState.offsetY,
                contentHeight: state.compactRect?.contentHeight || createDefaultCompactRect().contentHeight,
            });
            queueSync();
        }

        if (state.resizeState && state.mode === "compact") {
            const deltaX = event.clientX - state.resizeState.startX;
            const deltaY = event.clientY - state.resizeState.startY;
            const vectorX = state.resizeState.aspectRatio;
            const vectorY = 1;
            const scaleDelta = ((deltaX * vectorX) + (deltaY * vectorY)) / ((vectorX * vectorX) + (vectorY * vectorY));
            const contentHeight = clampCompactContentHeight(state.resizeState.startContentHeight + scaleDelta);
            applyCompactRect({
                left: state.resizeState.left,
                top: state.resizeState.top,
                contentHeight,
            });
            queueSync();
        }

        if (!state.panState) {
            return;
        }

        state.camera.centerX = state.panState.startCenterX - ((event.clientX - state.panState.startX) / Math.max(1, state.panState.displayWidth));
        state.camera.centerY = state.panState.startCenterY - ((event.clientY - state.panState.startY) / Math.max(1, state.panState.displayHeight));
        queueSync();
    }

    function handlePointerUp() {
        if (state.dragState && state.panel) {
            state.panel.classList.remove("dragging");
            state.dragState = null;
        }
        if (state.resizeState && state.panel) {
            state.panel.classList.remove("resizing");
            state.resizeState = null;
        }
        if (state.panState) {
            state.panState.viewport?.classList.remove("is-panning");
            state.panState = null;
        }
    }

    function handleWindowResize() {
        if (!state.panel || state.panel.classList.contains("is-hidden")) {
            return;
        }
        if (state.mode === "compact") {
            applyCompactRect(state.compactRect || createDefaultCompactRect());
        } else {
            applyWindowMode();
        }
        queueSettledSync();
    }

    function handleKeyDown(event) {
        if (!state.panel || state.panel.classList.contains("is-hidden")) {
            return;
        }
        if (event.defaultPrevented) {
            return;
        }
        if (event.metaKey || event.altKey) {
            return;
        }
        const target = event.target;
        if (
            target &&
            (target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName))
        ) {
            return;
        }

        const key = (event.key || "").toLowerCase();
        if (key === "escape") {
            closeCompare();
            event.preventDefault();
            return;
        }
        if (!event.ctrlKey && key === "r") {
            resetCamera();
            queueSync();
            event.preventDefault();
        }
    }

    function announceCompareState() {
        const stagingMap = {};
        state.items.forEach((item, index) => {
            if (item.stagingName) {
                stagingMap[item.stagingName] = `C${index + 1}`;
            }
        });
        window.dispatchEvent(new CustomEvent("nex-compare:state-change", {
            detail: { stagingMap },
        }));
    }

    window.addEventListener("nex-compare:open", (event) => {
        openCompare(event?.detail?.items || []);
    });

    window.addEventListener("nex-compare:close-request", closeCompare);
})();
