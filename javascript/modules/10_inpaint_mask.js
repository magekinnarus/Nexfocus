(() => {
  const ACTIVE_MASK_OPACITY = "0.6";
  const INACTIVE_MASK_OPACITY = "0.35";

  const MODES = {
    context: {
      rootId: "inpaint_canvas",
      fieldId: "inpaint_context_mask_data",
      canvasId: "inpaint-context-mask-overlay",
      statusId: "inpaint-mask-status",
      emptyStatus: "Load a source image, then paint the Step 1 context mask.",
      capturedStatus: "Step 1 context mask captured.",
      clearedStatus: "Step 1 context mask cleared.",
    },
    bb: {
      rootId: "inpaint_bb_canvas",
      fieldId: "inpaint_bb_mask_data",
      canvasId: "inpaint-bb-mask-overlay",
      statusId: "inpaint-mask-status",
      emptyStatus: "Load a BB image, then paint the Step 2 BB mask.",
      capturedStatus: "Step 2 BB mask captured.",
      clearedStatus: "Step 2 BB mask cleared.",
    },
    outpaint_bb: {
      rootId: "outpaint_bb_canvas",
      fieldId: "outpaint_bb_mask_data",
      canvasId: "outpaint-bb-mask-overlay",
      statusId: "outpaint-mask-status",
      emptyStatus: "Load a BB image, then paint the Step 2 BB mask.",
      capturedStatus: "Step 2 BB mask captured.",
      clearedStatus: "Step 2 BB mask cleared.",
    },
    remove: {
      rootId: "remove_base_image_slot",
      fieldId: "remove_mask_data",
      canvasId: "remove-mask-overlay",
      statusId: "remove-mask-status",
      emptyStatus: "Load a base image, then paint the Remove mask.",
      capturedStatus: "Remove mask captured.",
      clearedStatus: "Remove mask cleared.",
    },
  };

  const MODE_AFFECTED_SLOTS = {
    context: [
      "inpaint_context_mask_image_path",
      "inpaint_bb_image_path",
      "inpaint_mask_image_path",
    ],
    bb: ["inpaint_mask_image_path"],
    outpaint_bb: ["outpaint_mask_image_path"],
    remove: ["remove_mask_image_path"],
  };

  const state = {
    activeMode: "context",
    tool: "brush",
    brushSize: 36,
    resizeBound: false,
    initialized: false,
    shortcutsBound: false,
    enabledModes: {
      context: false,
      bb: false,
      outpaint_bb: false,
      remove: false,
    },
    retries: {
      context: 0,
      bb: 0,
      outpaint_bb: 0,
      remove: 0,
    },
    refreshPending: {
      context: false,
      bb: false,
      outpaint_bb: false,
      remove: false,
    },
    surfaces: {
      context: createSurface("context"),
      bb: createSurface("bb"),
      outpaint_bb: createSurface("outpaint_bb"),
      remove: createSurface("remove"),
    },
  };

  function createSurface(mode) {
    return {
      mode,
      root: null,
      host: null,
      img: null,
      canvas: null,
      ctx: null,
      sourceKey: null,
      observersAttached: false,
      drawing: false,
      lastPoint: null,
    };
  }

  function getSurface(mode) {
    return state.surfaces[mode];
  }

  function getRoot(mode) {
    return document.getElementById(MODES[mode].rootId);
  }

  function getImage(root) {
    return root ? root.querySelector("img") : null;
  }

  function getHost(root) {
    if (!root) return null;
    return root.querySelector(".image-container") || root;
  }

  function getMaskField(mode) {
    const fieldId = MODES[mode].fieldId;
    return document.querySelector(`#${fieldId} textarea, #${fieldId} input`);
  }

  function getStatus(mode = state.activeMode) {
    return document.getElementById(MODES[mode].statusId);
  }

  function setStatus(text, mode = state.activeMode) {
    const status = getStatus(mode);
    if (status) {
      status.textContent = text;
    }
  }

  function dispatchServerSync(pathFieldIds, mode = "live") {
    const uniqueIds = [...new Set((pathFieldIds || []).filter(Boolean))];
    if (!uniqueIds.length) return;
    window.dispatchEvent(
      new CustomEvent("nex-slot:server-sync", {
        detail: {
          pathFieldIds: uniqueIds,
          mode: mode === "once" ? "once" : "live",
        },
      }),
    );
  }

  function dispatchModeServerSync(mode, syncMode = "live") {
    dispatchServerSync(MODE_AFFECTED_SLOTS[mode] || [], syncMode);
  }

  window.nexDispatchSlotServerSync = dispatchServerSync;

  function setMaskValue(mode, value) {
    const field = getMaskField(mode);
    if (!field || field.value === value) return;
    field.value = value;
    field.dispatchEvent(new Event("input", { bubbles: true }));
    field.dispatchEvent(new Event("change", { bubbles: true }));
    dispatchModeServerSync(mode, "live");
  }

  function getFieldValue(fieldId) {
    const field = document.querySelector(`#${fieldId} textarea, #${fieldId} input`);
    return field ? String(field.value || "").trim() : "";
  }

  function getFieldControl(fieldId) {
    return document.querySelector(`#${fieldId} textarea, #${fieldId} input`);
  }

  function setFieldValue(fieldId, value) {
    const field = getFieldControl(fieldId);
    if (!field) return;
    const nextValue = value == null ? "" : String(value);
    const isCheckbox = field.type === "checkbox";
    if (isCheckbox) {
      const nextChecked =
        value === true || value === "true" || value === 1 || value === "1";
      if (field.checked === nextChecked) return;
      field.checked = nextChecked;
      field.value = nextChecked ? "true" : "false";
    } else {
      if (field.value === nextValue) return;
      field.value = nextValue;
    }
    field.dispatchEvent(new Event("input", { bubbles: true }));
    field.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function slotHasVisibleImage(slotId) {
    const slot = document.getElementById(slotId);
    const dropZone = slot ? slot.querySelector(".nex-slot__drop") : null;
    return !!(dropZone && dropZone.classList.contains("has-image"));
  }

  function slotUploadPending(slotId, pathFieldId, workspaceFieldId) {
    const slot = document.getElementById(slotId);
    if (!slot) {
      return false;
    }
    if (slot.dataset.uploading === "true") {
      return true;
    }

    if (!slotHasVisibleImage(slotId)) {
      return false;
    }

    const pathValue = pathFieldId ? getFieldValue(pathFieldId) : "";
    const workspaceValue = workspaceFieldId ? getFieldValue(workspaceFieldId) : "";
    return !pathValue && !workspaceValue;
  }

  function handleBaseSlotReplaced(event) {
    const detail = event && event.detail ? event.detail : {};
    const slotId = detail.slotId || "";

    if (slotId === "inpaint_canvas") {
      setFieldValue("inpaint_context_mask_image_path", "");
      setFieldValue("inpaint_context_mask_workspace_id", "");
      setFieldValue("inpaint_bb_image_path", "");
      setFieldValue("inpaint_bb_workspace_id", "");
      setFieldValue("inpaint_mask_image_path", "");
      setFieldValue("inpaint_mask_workspace_id", "");
      setFieldValue("inpaint_context_mask_data", "");
      setFieldValue("inpaint_bb_mask_data", "");
      setFieldValue("inpaint_step2_checkbox", false);
      state.enabledModes.bb = false;
      if (state.activeMode === "bb") {
        state.activeMode = "context";
      }
      updateModeButtons();
      dispatchServerSync(
        [
          "inpaint_context_mask_image_path",
          "inpaint_bb_image_path",
          "inpaint_mask_image_path",
        ],
        "live",
      );
      refreshAll();
      return;
    }

    if (slotId === "outpaint_input_slot") {
      setFieldValue("outpaint_bb_image_path", "");
      setFieldValue("outpaint_bb_workspace_id", "");
      setFieldValue("outpaint_mask_image_path", "");
      setFieldValue("outpaint_mask_workspace_id", "");
      setFieldValue("outpaint_bb_mask_data", "");
      setFieldValue("outpaint_step2_checkbox", false);
      state.enabledModes.outpaint_bb = false;
      updateModeButtons();
      dispatchServerSync(
        ["outpaint_bb_image_path", "outpaint_mask_image_path"],
        "live",
      );
      refreshAll();
      return;
    }

    if (slotId === "remove_base_image_slot") {
      setFieldValue("remove_mask_image_path", "");
      setFieldValue("remove_mask_workspace_id", "");
      setFieldValue("remove_mask_data", "");
      state.enabledModes.remove = false;
      updateModeButtons();
      dispatchServerSync(["remove_mask_image_path"], "live");
      refreshAll();
    }
  }

  function updateModeButtons() {
    const contextBtn = document.getElementById("inpaint-mask-mode-context");
    const bbBtn = document.getElementById("inpaint-mask-mode-bb");
    const outpaintBtn = document.getElementById("outpaint-mask-mode-bb");
    const inpaintDisableBtn = document.getElementById(
      "inpaint-mask-mode-disable",
    );
    const outpaintDisableBtn = document.getElementById(
      "outpaint-mask-mode-disable",
    );
    const removeBtn = document.getElementById("remove-mask-mode-bb");
    const removeDisableBtn = document.getElementById(
      "remove-mask-mode-disable",
    );

    if (contextBtn)
      contextBtn.classList.toggle("active", state.enabledModes.context);
    if (bbBtn) bbBtn.classList.toggle("active", state.enabledModes.bb);
    if (outpaintBtn)
      outpaintBtn.classList.toggle("active", state.enabledModes.outpaint_bb);
    if (removeBtn)
      removeBtn.classList.toggle("active", state.enabledModes.remove);

    if (inpaintDisableBtn)
      inpaintDisableBtn.classList.toggle(
        "active",
        !state.enabledModes.context && !state.enabledModes.bb,
      );
    if (outpaintDisableBtn)
      outpaintDisableBtn.classList.toggle(
        "active",
        !state.enabledModes.outpaint_bb,
      );
    if (removeDisableBtn)
      removeDisableBtn.classList.toggle("active", !state.enabledModes.remove);
  }
  function updateToolButtons() {
    [
      "inpaint-mask-brush",
      "inpaint-context-brush",
      "inpaint-bb-brush",
      "outpaint-mask-brush",
      "remove-mask-brush",
    ].forEach((id) => {
      const button = document.getElementById(id);
      if (button) button.classList.toggle("active", state.tool === "brush");
    });
    [
      "inpaint-mask-erase",
      "inpaint-context-erase",
      "inpaint-bb-erase",
      "outpaint-mask-erase",
      "remove-mask-erase",
    ].forEach((id) => {
      const button = document.getElementById(id);
      if (button) button.classList.toggle("active", state.tool === "erase");
    });
  }

  function currentModeName() {
    if (state.activeMode === "context") return "Context mask";
    if (state.activeMode === "bb") return "Inpaint BB mask";
    if (state.activeMode === "outpaint_bb") return "Outpaint BB mask";
    if (state.activeMode === "remove") return "Remove mask";
    return "Mask";
  }

  function setTool(tool) {
    state.tool = tool;
    updateToolButtons();
    setStatus(
      tool === "brush"
        ? `${currentModeName()} brush active.`
        : `${currentModeName()} eraser active.`,
    );
  }

  function isEditableTarget(target) {
    if (!target) return false;
    if (target.isContentEditable) return true;
    const tagName = target.tagName ? target.tagName.toLowerCase() : "";
    if (tagName === "input" || tagName === "textarea" || tagName === "select")
      return true;
    return !!target.closest(
      'input, textarea, select, [contenteditable=""], [contenteditable="true"]',
    );
  }

  function getBrushSizeInput(mode = state.activeMode) {
    if (mode === "outpaint_bb")
      return document.getElementById("outpaint-mask-size");
    if (mode === "remove") return document.getElementById("remove-mask-size");
    return document.getElementById("inpaint-mask-size");
  }

  function setBrushSize(nextSize, mode = state.activeMode) {
    const input = getBrushSizeInput(mode);
    const min = input ? parseInt(input.min, 10) || 8 : 8;
    const max = input ? parseInt(input.max, 10) || 160 : 160;
    const clamped = Math.max(min, Math.min(max, nextSize));
    state.brushSize = clamped;
    if (input) {
      input.value = String(clamped);
      input.dispatchEvent(new Event("input", { bubbles: true }));
    }
    return clamped;
  }

  function handleMaskHotkeys(event) {
    if (
      event.defaultPrevented ||
      event.altKey ||
      event.ctrlKey ||
      event.metaKey
    )
      return;
    if (isEditableTarget(event.target)) return;
    if (!state.enabledModes[state.activeMode]) return;

    const key = (event.key || "").toLowerCase();
    if (!key) return;

    if (key === "b") {
      setTool("brush");
      event.preventDefault();
      return;
    }

    if (key === "e") {
      setTool("erase");
      event.preventDefault();
      return;
    }

    if (key === "c") {
      clearMask(state.activeMode);
      event.preventDefault();
      return;
    }

    if (key === "q" || key === "w") {
      const delta = key === "q" ? -4 : 4;
      const nextSize = setBrushSize(state.brushSize + delta, state.activeMode);
      setStatus(`${currentModeName()} brush size: ${nextSize}px`);
      event.preventDefault();
      return;
    }

    if (key === "r") {
      refreshMaskingSystem();
      event.preventDefault();
    }
  }

  function setActiveMode(mode) {
    if (!MODES[mode]) return;

    // --- Isolation Logic ---
    if (mode === "context") {
      state.enabledModes.context = true;
      state.enabledModes.bb = false;
    } else if (mode === "bb") {
      state.enabledModes.context = false;
      state.enabledModes.bb = true;
    } else if (mode === "outpaint_bb") {
      state.enabledModes.outpaint_bb = true;
    } else if (mode === "remove") {
      state.enabledModes.remove = true;
    }

    state.activeMode = mode;
    updateModeButtons();
    syncCanvasInteractivity();

    const surface = getSurface(mode);
    refreshMode(mode); // Trigger immediate refresh for priority activation

    if (
      !surface.img ||
      !surface.img.src ||
      surface.img.style.display === "none"
    ) {
      setStatus(MODES[mode].emptyStatus);
      return;
    }
    setStatus(
      mode === "context"
        ? "Step 1 context mask ready."
        : mode === "bb"
          ? "Step 2 Inpaint BB mask ready."
          : mode === "outpaint_bb"
            ? "Step 2 Outpaint BB mask ready."
            : "Remove mask ready.",
    );
  }

  function disableMasking(group) {
    if (group === "inpaint") {
      state.enabledModes.context = false;
      state.enabledModes.bb = false;
    } else if (group === "outpaint") {
      state.enabledModes.outpaint_bb = false;
    } else if (group === "remove") {
      state.enabledModes.remove = false;
    }
    updateModeButtons();
    refreshAll();
    setStatus(
      group === "inpaint"
        ? "Inpaint masking disabled."
        : group === "outpaint"
          ? "Outpaint masking disabled."
          : "Remove masking disabled.",
    );
  }

  function hasPaint(surface) {
    if (!surface.ctx || !surface.canvas) return false;
    const alpha = surface.ctx.getImageData(
      0,
      0,
      surface.canvas.width,
      surface.canvas.height,
    ).data;
    for (let i = 3; i < alpha.length; i += 4) {
      if (alpha[i] !== 0) return true;
    }
    return false;
  }

  function exportMask(mode) {
    if (state.exportTimers && state.exportTimers[mode]) {
      clearTimeout(state.exportTimers[mode]);
    }

    if (!state.exportTimers) state.exportTimers = {};

    if (mode === state.activeMode) {
      setStatus("Processing mask...");
    }

    state.exportTimers[mode] = setTimeout(() => {
      const surface = getSurface(mode);
      if (!surface.canvas || !surface.ctx) return;
      if (!hasPaint(surface)) {
        setMaskValue(mode, "");
        if (mode === state.activeMode) {
          setStatus(MODES[mode].emptyStatus);
        }
        return;
      }
      setMaskValue(mode, surface.canvas.toDataURL("image/png"));
      if (mode === state.activeMode) {
        setStatus(MODES[mode].capturedStatus);
      }
    }, 800); // Debounce: wait 800ms after the last stroke
  }

  function clearMask(mode = state.activeMode) {
    const surface = getSurface(mode);
    if (!surface.ctx || !surface.canvas) return;
    surface.ctx.clearRect(0, 0, surface.canvas.width, surface.canvas.height);
    setMaskValue(mode, "");
    if (mode === state.activeMode) {
      setStatus(MODES[mode].clearedStatus);
    }
  }

  function pointerToCanvas(surface, event) {
    if (!surface.canvas) return null;
    const rect = surface.canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    return {
      x: (event.clientX - rect.left) * (surface.canvas.width / rect.width),
      y: (event.clientY - rect.top) * (surface.canvas.height / rect.height),
    };
  }

  function drawLine(surface, from, to) {
    if (!surface.ctx) return;
    surface.ctx.save();
    surface.ctx.lineCap = "round";
    surface.ctx.lineJoin = "round";
    surface.ctx.lineWidth = state.brushSize;
    if (state.tool === "erase") {
      surface.ctx.globalCompositeOperation = "destination-out";
      surface.ctx.strokeStyle = "rgba(0,0,0,1)";
    } else {
      surface.ctx.globalCompositeOperation = "source-over";
      surface.ctx.strokeStyle = "rgba(255,255,255,1)";
    }
    surface.ctx.beginPath();
    surface.ctx.moveTo(from.x, from.y);
    surface.ctx.lineTo(to.x, to.y);
    surface.ctx.stroke();
    surface.ctx.restore();
  }

  function attachCanvasEvents(mode) {
    const surface = getSurface(mode);
    surface.canvas.addEventListener("pointerdown", (event) => {
      if (
        state.activeMode !== mode ||
        !state.enabledModes[mode] ||
        !surface.img ||
        !surface.img.src
      )
        return;
      const point = pointerToCanvas(surface, event);
      if (!point) return;
      surface.drawing = true;
      surface.lastPoint = point;
      drawLine(surface, point, point);
      try {
        surface.canvas.setPointerCapture(event.pointerId);
      } catch (e) {
        console.warn("[Masking] Failed to set pointer capture:", e);
      }
      event.preventDefault();
    });

    surface.canvas.addEventListener("pointermove", (event) => {
      if (
        state.activeMode !== mode ||
        !state.enabledModes[mode] ||
        !surface.drawing
      )
        return;
      const point = pointerToCanvas(surface, event);
      if (!point || !surface.lastPoint) return;
      drawLine(surface, surface.lastPoint, point);
      surface.lastPoint = point;
      event.preventDefault();
    });

    const stopDrawing = (event) => {
      if (!surface.drawing) return;
      surface.drawing = false;
      surface.lastPoint = null;
      try {
        if (event && event.pointerId !== undefined) {
          surface.canvas.releasePointerCapture(event.pointerId);
        }
      } catch (e) {}
      exportMask(mode);
      if (event && event.preventDefault) event.preventDefault();
    };

    surface.canvas.addEventListener("pointerup", stopDrawing);
    surface.canvas.addEventListener("pointercancel", stopDrawing);
    surface.canvas.addEventListener("pointerleave", stopDrawing);
  }

  function ensureCanvas(mode) {
    const surface = getSurface(mode);
    if (surface.canvas || !surface.host) return;
    surface.host.style.position = "relative";
    const canvas = document.createElement("canvas");
    canvas.id = MODES[mode].canvasId;
    Object.assign(canvas.style, {
      position: "absolute",
      zIndex: "20",
      cursor: "crosshair",
      touchAction: "none",
      opacity: INACTIVE_MASK_OPACITY,
      pointerEvents: "none",
    });
    surface.host.appendChild(canvas);
    surface.canvas = canvas;
    surface.ctx = canvas.getContext("2d", { willReadFrequently: true });
    attachCanvasEvents(mode);
  }

  function getContainSize(img) {
    const {
      clientWidth: width,
      clientHeight: height,
      naturalWidth,
      naturalHeight,
    } = img;
    if (!width || !height || !naturalWidth || !naturalHeight) return null;

    const imgRatio = naturalWidth / naturalHeight;
    const containerRatio = width / height;

    let w, h, x, y;
    if (imgRatio > containerRatio) {
      w = width;
      h = width / imgRatio;
      x = 0;
      y = (height - h) / 2;
    } else {
      h = height;
      w = height * imgRatio;
      x = (width - w) / 2;
      y = 0;
    }
    return { w, h, x, y };
  }

  function syncCanvasToImage(mode, clearForNewImage = false) {
    const surface = getSurface(mode);
    if (!surface.canvas || !surface.img || !surface.host) return;

    const size = getContainSize(surface.img);
    if (!size) {
      surface.canvas.style.display = "none";
      return;
    }

    const hostRect = surface.host.getBoundingClientRect();
    const imgRect = surface.img.getBoundingClientRect();

    surface.canvas.style.display = "block";
    surface.canvas.style.left = `${imgRect.left - hostRect.left + size.x}px`;
    surface.canvas.style.top = `${imgRect.top - hostRect.top + size.y}px`;
    surface.canvas.style.width = `${size.w}px`;
    surface.canvas.style.height = `${size.h}px`;

    if (
      clearForNewImage ||
      surface.canvas.width !== surface.img.naturalWidth ||
      surface.canvas.height !== surface.img.naturalHeight
    ) {
      surface.canvas.width = surface.img.naturalWidth;
      surface.canvas.height = surface.img.naturalHeight;
      surface.ctx.clearRect(0, 0, surface.canvas.width, surface.canvas.height);
      setMaskValue(mode, "");
      if (mode === state.activeMode) {
        setStatus(MODES[mode].emptyStatus);
      }
    }
  }

  function syncCanvasInteractivity() {
    Object.keys(MODES).forEach((mode) => {
      const surface = getSurface(mode);
      if (!surface.canvas) return;
      const active =
        mode === state.activeMode &&
        state.enabledModes[mode] &&
        surface.img &&
        surface.img.src &&
        surface.img.style.display !== "none";
      surface.canvas.style.pointerEvents = active ? "auto" : "none";
      surface.canvas.style.cursor = active ? "crosshair" : "default";
      surface.canvas.style.opacity = active
        ? ACTIVE_MASK_OPACITY
        : INACTIVE_MASK_OPACITY;
      surface.canvas.style.display = state.enabledModes[mode]
        ? "block"
        : "none";
    });
  }

  function ensureObservers(mode) {
    const surface = getSurface(mode);
    if (surface.observersAttached || !surface.root) return;
    const observer = new MutationObserver(() => {
      if (!state.refreshPending[mode]) {
        state.refreshPending[mode] = true;
        window.requestAnimationFrame(() => {
          state.refreshPending[mode] = false;
          refreshMode(mode);
        });
      }
    });
    observer.observe(surface.root, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["src", "style", "class"],
    });
    surface.observersAttached = true;
    if (!state.resizeBound) {
      window.addEventListener("resize", refreshAll);
      state.resizeBound = true;
    }
  }

  function attachControls() {
    if (state.initialized) return;

    const brushBtn = document.getElementById("inpaint-mask-brush");
    const eraseBtn = document.getElementById("inpaint-mask-erase");
    const clearBtn = document.getElementById("inpaint-mask-clear");
    const contextBrushBtn = document.getElementById("inpaint-context-brush");
    const contextEraseBtn = document.getElementById("inpaint-context-erase");
    const contextClearBtn = document.getElementById("inpaint-context-clear");
    const bbBrushBtn = document.getElementById("inpaint-bb-brush");
    const bbEraseBtn = document.getElementById("inpaint-bb-erase");
    const bbClearBtn = document.getElementById("inpaint-bb-clear");
    const sizeInput = document.getElementById("inpaint-mask-size");
    const contextBtn = document.getElementById("inpaint-mask-mode-context");
    const bbBtn = document.getElementById("inpaint-mask-mode-bb");
    const inpaintDisableBtn = document.getElementById(
      "inpaint-mask-mode-disable",
    );
    const replaceBbBtn = document.getElementById("inpaint-mask-refresh-bb");
    const replaceBbNonceField = document.querySelector(
      "#inpaint_replace_bb_nonce textarea, #inpaint_replace_bb_nonce input",
    );

    if (brushBtn && !brushBtn.dataset.bound) {
      brushBtn.dataset.bound = "1";
      brushBtn.addEventListener("click", () => setTool("brush"));
    }
    if (eraseBtn && !eraseBtn.dataset.bound) {
      eraseBtn.dataset.bound = "1";
      eraseBtn.addEventListener("click", () => setTool("erase"));
    }
    if (clearBtn && !clearBtn.dataset.bound) {
      clearBtn.dataset.bound = "1";
      clearBtn.addEventListener("click", () => clearMask());
    }
    if (contextBrushBtn && !contextBrushBtn.dataset.bound) {
      contextBrushBtn.dataset.bound = "1";
      contextBrushBtn.addEventListener("click", () => {
        setActiveMode("context");
        setTool("brush");
      });
    }
    if (contextEraseBtn && !contextEraseBtn.dataset.bound) {
      contextEraseBtn.dataset.bound = "1";
      contextEraseBtn.addEventListener("click", () => {
        setActiveMode("context");
        setTool("erase");
      });
    }
    if (contextClearBtn && !contextClearBtn.dataset.bound) {
      contextClearBtn.dataset.bound = "1";
      contextClearBtn.addEventListener("click", () => {
        setActiveMode("context");
        clearMask("context");
      });
    }
    if (bbBrushBtn && !bbBrushBtn.dataset.bound) {
      bbBrushBtn.dataset.bound = "1";
      bbBrushBtn.addEventListener("click", () => {
        setActiveMode("bb");
        setTool("brush");
      });
    }
    if (bbEraseBtn && !bbEraseBtn.dataset.bound) {
      bbEraseBtn.dataset.bound = "1";
      bbEraseBtn.addEventListener("click", () => {
        setActiveMode("bb");
        setTool("erase");
      });
    }
    if (bbClearBtn && !bbClearBtn.dataset.bound) {
      bbClearBtn.dataset.bound = "1";
      bbClearBtn.addEventListener("click", () => {
        setActiveMode("bb");
        clearMask("bb");
      });
    }
    if (sizeInput && !sizeInput.dataset.bound) {
      sizeInput.dataset.bound = "1";
      sizeInput.addEventListener("input", () => {
        state.brushSize = parseInt(sizeInput.value, 10) || 36;
      });
      state.brushSize = parseInt(sizeInput.value, 10) || 36;
    }
    if (contextBtn && !contextBtn.dataset.bound) {
      contextBtn.dataset.bound = "1";
      contextBtn.addEventListener("click", () => setActiveMode("context"));
    }
    if (bbBtn && !bbBtn.dataset.bound) {
      bbBtn.dataset.bound = "1";
      bbBtn.addEventListener("click", () => setActiveMode("bb"));
    }
    if (replaceBbBtn && !replaceBbBtn.dataset.bound) {
      replaceBbBtn.dataset.bound = "1";
      replaceBbBtn.addEventListener("click", () => {
        const basePending = slotUploadPending(
          "inpaint_canvas",
          "inpaint_input_image_path",
          "inpaint_input_workspace_id",
        );
        if (basePending) {
          setStatus("Press the Refresh button, reload the Base Image, and try again.");
          return;
        }

        const contextPending = slotUploadPending(
          "inpaint_context_mask_canvas",
          "inpaint_context_mask_image_path",
          "inpaint_context_mask_workspace_id",
        );
        const contextMaskData = getFieldValue("inpaint_context_mask_data");
        if (contextPending || (slotHasVisibleImage("inpaint_context_mask_canvas") && !getFieldValue("inpaint_context_mask_image_path") && !contextMaskData)) {
          setStatus("Press the Refresh button, reload the context mask, and try again.");
          return;
        }

        const nonceField = document.querySelector(
          "#inpaint_replace_bb_nonce textarea, #inpaint_replace_bb_nonce input",
        );
        if (!nonceField) {
          setStatus("Press the Refresh button, remove all the loaded images and try again.");
          return;
        }
        setStatus("Refreshing BB image...");
        nonceField.value = String(Date.now());
        nonceField.dispatchEvent(new Event("input", { bubbles: true }));
        nonceField.dispatchEvent(new Event("change", { bubbles: true }));
        dispatchServerSync(
          ["inpaint_bb_image_path", "inpaint_mask_image_path"],
          "once",
        );
      });
    }
    if (inpaintDisableBtn && !inpaintDisableBtn.dataset.bound) {
      inpaintDisableBtn.dataset.bound = "1";
      inpaintDisableBtn.addEventListener("click", () =>
        disableMasking("inpaint"),
      );
    }

    // --- Outpaint Tab Controls ---
    // --- Outpaint Tab Controls ---
    const outpaintBrushBtn = document.getElementById("outpaint-mask-brush");
    const outpaintEraseBtn = document.getElementById("outpaint-mask-erase");
    const outpaintClearBtn = document.getElementById("outpaint-mask-clear");
    const outpaintSizeInput = document.getElementById("outpaint-mask-size");
    const outpaintModeBtn = document.getElementById("outpaint-mask-mode-bb");
    const outpaintDisableBtn = document.getElementById(
      "outpaint-mask-mode-disable",
    );
    const removeBrushBtn = document.getElementById("remove-mask-brush");
    const removeEraseBtn = document.getElementById("remove-mask-erase");
    const removeClearBtn = document.getElementById("remove-mask-clear");
    const removeSizeInput = document.getElementById("remove-mask-size");
    const removeModeBtn = document.getElementById("remove-mask-mode-bb");
    const removeDisableBtn = document.getElementById(
      "remove-mask-mode-disable",
    );

    if (outpaintBrushBtn && !outpaintBrushBtn.dataset.bound) {
      outpaintBrushBtn.dataset.bound = "1";
      outpaintBrushBtn.addEventListener("click", () => {
        setActiveMode("outpaint_bb");
        setTool("brush");
      });
    }
    if (outpaintEraseBtn && !outpaintEraseBtn.dataset.bound) {
      outpaintEraseBtn.dataset.bound = "1";
      outpaintEraseBtn.addEventListener("click", () => {
        setActiveMode("outpaint_bb");
        setTool("erase");
      });
    }
    if (outpaintClearBtn && !outpaintClearBtn.dataset.bound) {
      outpaintClearBtn.dataset.bound = "1";
      outpaintClearBtn.addEventListener("click", () => {
        setActiveMode("outpaint_bb");
        clearMask("outpaint_bb");
      });
    }
    if (outpaintSizeInput && !outpaintSizeInput.dataset.bound) {
      outpaintSizeInput.dataset.bound = "1";
      outpaintSizeInput.addEventListener("input", () => {
        state.brushSize = parseInt(outpaintSizeInput.value, 10) || 36;
      });
    }
    if (outpaintModeBtn && !outpaintModeBtn.dataset.bound) {
      outpaintModeBtn.dataset.bound = "1";
      outpaintModeBtn.addEventListener("click", () =>
        setActiveMode("outpaint_bb"),
      );
    }
    if (outpaintDisableBtn && !outpaintDisableBtn.dataset.bound) {
      outpaintDisableBtn.dataset.bound = "1";
      outpaintDisableBtn.addEventListener("click", () =>
        disableMasking("outpaint"),
      );
    }
    if (removeBrushBtn && !removeBrushBtn.dataset.bound) {
      removeBrushBtn.dataset.bound = "1";
      removeBrushBtn.addEventListener("click", () => {
        setActiveMode("remove");
        setTool("brush");
      });
    }
    if (removeEraseBtn && !removeEraseBtn.dataset.bound) {
      removeEraseBtn.dataset.bound = "1";
      removeEraseBtn.addEventListener("click", () => {
        setActiveMode("remove");
        setTool("erase");
      });
    }
    if (removeClearBtn && !removeClearBtn.dataset.bound) {
      removeClearBtn.dataset.bound = "1";
      removeClearBtn.addEventListener("click", () => {
        setActiveMode("remove");
        clearMask("remove");
      });
    }
    if (removeSizeInput && !removeSizeInput.dataset.bound) {
      removeSizeInput.dataset.bound = "1";
      removeSizeInput.addEventListener("input", () => {
        state.brushSize = parseInt(removeSizeInput.value, 10) || 36;
      });
    }
    if (removeModeBtn && !removeModeBtn.dataset.bound) {
      removeModeBtn.dataset.bound = "1";
      removeModeBtn.addEventListener("click", () => setActiveMode("remove"));
    }
    if (removeDisableBtn && !removeDisableBtn.dataset.bound) {
      removeDisableBtn.dataset.bound = "1";
      removeDisableBtn.addEventListener("click", () =>
        disableMasking("remove"),
      );
    }

    updateModeButtons();
    if (!state.shortcutsBound) {
      document.addEventListener("keydown", handleMaskHotkeys);
      state.shortcutsBound = true;
    }

    updateToolButtons();

    // Mark as initialized if we found the main controls
    if (brushBtn && outpaintBrushBtn && removeBrushBtn) {
      state.initialized = true;
    }

    ["inpaint-mask-reset", "outpaint-mask-reset", "remove-mask-reset"].forEach(
      (id) => {
        const button = document.getElementById(id);
        if (button && !button.dataset.bound) {
          button.dataset.bound = "1";
          button.addEventListener("click", () => refreshMaskingSystem());
        }
      },
    );
  }

  function refreshMode(mode) {
    const root = getRoot(mode);
    // --- Auto-Disable if Tab is Hidden ---
    if (root && root.offsetParent === null) {
      const surface = getSurface(mode);
      if (surface.canvas) surface.canvas.style.display = "none";
      return;
    }

    if (!state.enabledModes[mode]) {
      const surface = getSurface(mode);
      if (surface.canvas) surface.canvas.style.display = "none";
      return;
    }

    const surface = getSurface(mode);
    surface.root = getRoot(mode);
    if (!surface.root) return;
    surface.host = getHost(surface.root);

    surface.img = getImage(surface.root);
    ensureCanvas(mode);
    ensureObservers(mode);

    if (
      !surface.img ||
      !surface.img.src ||
      surface.img.style.display === "none"
    ) {
      if (surface.canvas) {
        surface.canvas.style.display = "none";
      }
      surface.sourceKey = null;
      if (mode === state.activeMode) {
        setStatus(MODES[mode].emptyStatus);
      }
      syncCanvasInteractivity();
      return;
    }

    // --- Priority Activation: Retry if Connection is Queued ---
    if (surface.img.naturalWidth === 0 && state.retries[mode] < 10) {
      state.retries[mode]++;
      window.setTimeout(() => refreshMode(mode), 200);
      return;
    }
    state.retries[mode] = 0; // Reset on success

    const nextKey = `${surface.img.currentSrc || surface.img.src}|${surface.img.naturalWidth}x${surface.img.naturalHeight}`;
    const clearForNewImage = nextKey !== surface.sourceKey;
    surface.sourceKey = nextKey;
    syncCanvasToImage(mode, clearForNewImage);
    syncCanvasInteractivity();
  }

  function refreshAll() {
    attachControls();
    refreshMode("context");
    refreshMode("bb");
    refreshMode("outpaint_bb");
    refreshMode("remove");
  }

  function start() {
    window.addEventListener("nex-slot:base-replaced", handleBaseSlotReplaced);
    refreshAll();
    window.setInterval(refreshAll, 500);
  }

  function refreshMaskingSystem() {
    Object.keys(MODES).forEach((mode) => {
      const surface = getSurface(mode);
      if (surface.drawing) {
        surface.drawing = false;
        surface.lastPoint = null;
      }
      if (surface.canvas && surface.canvas.parentNode) {
        surface.canvas.parentNode.removeChild(surface.canvas);
      }
      surface.canvas = null;
      surface.ctx = null;
      surface.observersAttached = false;
      surface.root = null;
      surface.host = null;
      surface.img = null;
      surface.sourceKey = null;
    });

    state.initialized = false;
    document
      .querySelectorAll(
        ".mask-tool-btn, .mask-workflow-toolbar input, .mask-workflow-toolbar button",
      )
      .forEach((element) => {
        delete element.dataset.bound;
      });

    refreshAll();
    setStatus("Masking controls refreshed.");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})();
