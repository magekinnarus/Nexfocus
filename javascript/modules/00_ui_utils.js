function gradioApp() {
    const elems = document.getElementsByTagName('gradio-app');
    const elem = elems.length == 0 ? document : elems[0];

    if (elem !== document) {
        elem.getElementById = function (id) {
            return document.getElementById(id);
        };
    }
    return elem.shadowRoot ? elem.shadowRoot : elem;
}

var uiUpdateCallbacks = [];
var uiAfterUpdateCallbacks = [];
var aspectRatioLabelObserver = null;
var aspectRatioLabelTarget = null;

function onUiUpdate(callback) {
    uiUpdateCallbacks.push(callback);
}

function onAfterUiUpdate(callback) {
    uiAfterUpdateCallbacks.push(callback);
}

function executeCallbacks(queue, arg) {
    for (const callback of queue) {
        try {
            callback(arg);
        } catch (e) {
            console.error("error running callback", callback, ":", e);
        }
    }
}

function htmlDecode(input) {
    var doc = new DOMParser().parseFromString(input, "text/html");
    return doc.documentElement.textContent;
}

// Localization System
var re_num = /^[.\d]+$/;
var original_lines = {};
var translated_lines = {};

function hasLocalization() {
    return window.localization && Object.keys(window.localization).length > 0;
}

function getTranslation(text) {
    if (!text) return undefined;
    if (translated_lines[text] === undefined) {
        original_lines[text] = 1;
    }
    var tl = localization[text];
    if (tl !== undefined) {
        translated_lines[tl] = 1;
    }
    return tl;
}

function processTextNode(node) {
    var text = node.textContent.trim();
    if (!text || !node.parentElement) return;
    var parentType = node.parentElement.nodeName;
    if (parentType == 'SCRIPT' || parentType == 'STYLE' || parentType == 'TEXTAREA') return;
    if (re_num.test(text)) return;

    var tl = getTranslation(text);
    if (tl !== undefined) {
        node.textContent = tl;
        if (text && node.parentElement) {
            node.parentElement.setAttribute("data-original-text", text);
        }
    }
}

function processNode(node) {
    if (node.nodeType == 3) {
        processTextNode(node);
        return;
    }
    if (node.title) {
        let tl = getTranslation(node.title);
        if (tl !== undefined) node.title = tl;
    }
    if (node.placeholder) {
        let tl = getTranslation(node.placeholder);
        if (tl !== undefined) node.placeholder = tl;
    }

    // Find text nodes
    var walk = document.createTreeWalker(node, NodeFilter.SHOW_TEXT, null, false);
    var n;
    while ((n = walk.nextNode())) {
        processTextNode(n);
    }
}

function refresh_style_localization() {
    const stylesNode = gradioApp().querySelector('.style_selections');
    if (!stylesNode) return;
    processNode(stylesNode);
}

function get_selected_aspect_ratio_value(fallbackValue) {
    const root = gradioApp();
    const checked = root.querySelector('#aspect_ratios_selection input[type="radio"]:checked') ||
        root.querySelector('.aspect_ratios input[type="radio"]:checked');
    if (checked && checked.value) {
        return htmlDecode(String(checked.value));
    }

    const checkedLabel = checked ? checked.closest('label') : null;
    const checkedText = checkedLabel ? (checkedLabel.querySelector('span')?.textContent || checkedLabel.textContent) : '';
    if (checkedText && String(checkedText).trim()) {
        return htmlDecode(String(checkedText).trim());
    }

    if (fallbackValue === undefined || fallbackValue === null) {
        return '';
    }
    return htmlDecode(String(fallbackValue));
}

function refresh_aspect_ratios_label(value) {
    const root = gradioApp();
    const label = root.querySelector('#aspect_ratios_accordion summary .label-wrap span') ||
        root.querySelector('#aspect_ratios_accordion .label-wrap span') ||
        root.querySelector('#aspect_ratios_accordion summary span') ||
        root.querySelector('#aspect_ratios_accordion span');

    if (!label) {
        return;
    }

    let translation = getTranslation("Aspect Ratios");
    if (typeof translation == "undefined") {
        translation = "Aspect Ratios";
    }
    const selectedValue = get_selected_aspect_ratio_value(value);
    label.textContent = selectedValue ? `${translation} ${selectedValue}` : translation;
}

function bind_aspect_ratio_label_sync() {
    const root = gradioApp();
    const selectionRoot = root.querySelector('#aspect_ratios_selection') || root.querySelector('.aspect_ratios');
    if (!selectionRoot) {
        return;
    }

    if (aspectRatioLabelTarget !== selectionRoot) {
        if (aspectRatioLabelObserver) {
            aspectRatioLabelObserver.disconnect();
        }

        aspectRatioLabelTarget = selectionRoot;
        aspectRatioLabelObserver = new MutationObserver(function () {
            window.requestAnimationFrame(function () {
                refresh_aspect_ratios_label();
            });
        });
        aspectRatioLabelObserver.observe(selectionRoot, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['checked', 'value', 'aria-checked']
        });
        selectionRoot.addEventListener('change', function () {
            window.requestAnimationFrame(function () {
                refresh_aspect_ratios_label();
            });
        });
    }

    window.requestAnimationFrame(function () {
        refresh_aspect_ratios_label();
    });
}

function localizeWholePage() {
    processNode(gradioApp());
}

window.cancelGenerateForever = function() {
    window.nex_interrupt_requested = true;
};

document.addEventListener("DOMContentLoaded", function () {
    const mutationObserver = new MutationObserver(function (m) {
        if (gradioApp().querySelector('#generate_button')) {
            executeCallbacks(uiUpdateCallbacks, m);
            // scheduleAfterUiUpdate omitted for simplicity if not heavily used
        }
    });
    mutationObserver.observe(gradioApp(), { childList: true, subtree: true });

    if (hasLocalization()) {
        localizeWholePage();
    }

    bind_aspect_ratio_label_sync();

});

onUiUpdate(function () {
    bind_aspect_ratio_label_sync();
});
