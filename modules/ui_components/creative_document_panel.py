"""Small Gradio host for the bounded W03 creative-document custom element."""

from __future__ import annotations

import ipaddress

import args_manager
import gradio as gr

from modules.creative_document_editor_api import issue_editor_session


EDITOR_SESSION_FIELD_ID = "creative_document_session_capability"


def _editor_capability(request: gr.Request) -> str:
    remote = bool(getattr(args_manager.args, "share", False))
    bind_host = getattr(args_manager.args, "listen", None)
    if bind_host:
        host = str(bind_host).strip().strip("[]")
        try:
            remote = remote or not ipaddress.ip_address(host).is_loopback
        except ValueError:
            remote = remote or host.lower() != "localhost"
    return issue_editor_session(request, remote_exposure=remote)


def create_creative_document_panel(root: gr.Blocks) -> gr.Textbox:
    """Mount the editor and issue a short-lived capability through Gradio."""

    with gr.Tab(label="Creative Document", id="creative_document_workspace", elem_id="creative_document_workspace"):
        gr.HTML(
            '<creative-document-editor id="nex-creative-document-editor" '
            f'data-session-field="{EDITOR_SESSION_FIELD_ID}"></creative-document-editor>'
        )
        # This field carries only an ephemeral session capability. Creative
        # state, project identity, and edits stay in the document service.
        capability = gr.Textbox(
            value="",
            show_label=False,
            container=False,
            elem_id=EDITOR_SESSION_FIELD_ID,
            elem_classes=["nex-creative-document-session-field"],
        )
    root.load(_editor_capability, inputs=None, outputs=capability, show_progress="hidden")
    return capability
