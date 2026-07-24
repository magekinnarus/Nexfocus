import os
import sys
import types

sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.environment_profile import EnvironmentProfile
from backend.memory_governor import MemoryGovernor, MemoryPhase, MemorySnapshot, normalize_phase
from backend import sdxl_runtime_policy


class DummyTask:
    pass


def _stub_snapshot(phase, notes=None, task=None, *, free_ram_mb=None, free_vram_mb=None):
    payload = dict(notes or {})
    if task is not None:
        payload.setdefault('task_type', task.__class__.__name__)
    return MemorySnapshot(
        timestamp=0.0,
        phase=phase,
        total_vram_mb=4096.0,
        free_vram_mb=free_vram_mb,
        total_ram_mb=8192.0,
        free_ram_mb=free_ram_mb,
        notes=payload,
    )


def test_normalize_phase_maps_legacy_aliases_to_w02_contract():
    assert normalize_phase('prepare') == MemoryPhase.MODEL_REFRESH.value
    assert normalize_phase('image_input') == MemoryPhase.IMAGE_INPUT_PREPARE.value
    assert normalize_phase('control') == MemoryPhase.CONTROL_APPLY.value
    assert normalize_phase('postprocess') == MemoryPhase.FINALIZE.value
    assert normalize_phase(MemoryPhase.DECODE) == MemoryPhase.DECODE.value


def test_phase_scope_balances_nested_transitions_and_restores_idle():
    governor = MemoryGovernor()
    governor.capture_snapshot = lambda notes=None, task=None: _stub_snapshot(governor.current_phase(), notes=notes, task=task)
    task = DummyTask()

    with governor.phase_scope(MemoryPhase.TASK, task=task, notes={'stage': 'outer'}, end_notes={'done': True}):
        assert governor.current_phase() == MemoryPhase.TASK.value

        with governor.phase_scope('prepare', notes={'stage': 'inner'}, end_notes={'done': 'inner'}):
            assert governor.current_phase() == MemoryPhase.MODEL_REFRESH.value

        assert governor.current_phase() == MemoryPhase.TASK.value

    assert governor.current_phase() == MemoryPhase.IDLE.value

    history = governor.history()
    assert [snapshot.phase for snapshot in history] == [
        MemoryPhase.TASK.value,
        MemoryPhase.MODEL_REFRESH.value,
        MemoryPhase.TASK.value,
        MemoryPhase.IDLE.value,
    ]
    assert history[0].notes['task_type'] == 'DummyTask'
    assert history[1].notes['stage'] == 'inner'
    assert history[2].notes['done'] == 'inner'
    assert history[3].notes['done'] is True


def test_configure_environment_applies_profile_overrides():
    governor = MemoryGovernor()
    profile = EnvironmentProfile(
        name='custom',
        display_name='Custom',
        source='test',
        total_ram_mb=16384.0,
        total_vram_mb=4096.0,
        is_colab=False,
        policy_overrides={
            'low_ram_headroom_mb': 5120.0,
            'checkpoint_switch_ram_headroom_mb': 6144.0,
        },
    )

    governor.configure_environment(profile=profile)

    assert governor.profile_name() == 'custom'
    assert governor.policy.low_ram_headroom_mb == 5120.0
    assert governor.policy.checkpoint_switch_ram_headroom_mb == 6144.0
    assert governor.policy_summary()['profile'] == 'custom'


def test_can_afford_reports_ram_shortfall():
    governor = MemoryGovernor()
    governor.capture_snapshot = lambda notes=None, task=None: _stub_snapshot(
        governor.current_phase(),
        notes=notes,
        task=task,
        free_ram_mb=1536.0,
        free_vram_mb=2048.0,
    )

    affordance = governor.can_afford(required_ram_mb=256.0, minimum_free_ram_mb=1400.0, phase=MemoryPhase.MODEL_REFRESH)

    assert affordance.allowed is False
    assert affordance.phase == MemoryPhase.MODEL_REFRESH.value
    assert affordance.free_ram_after_mb == 1280.0
    assert 'below floor=1400.0MB' in affordance.reason


def test_phase_residency_plan_uses_the_maintained_baseline_for_all_tasks():
    governor = MemoryGovernor()

    plan = governor.plan_for_task(task=DummyTask(), phase=MemoryPhase.PROMPT_ENCODE)

    assert plan.pinned == ("clip",)
    assert plan.warm == ("unet", "vae")
    assert plan.notes["source"] == "profile_phase_residency"
