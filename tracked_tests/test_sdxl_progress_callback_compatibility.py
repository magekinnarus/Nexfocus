from types import SimpleNamespace

from backend.sdxl_assembly.progress import SDXLAssemblyProgressCallback


_active_task = None


def get_active_task():
    return _active_task


def progressbar(task_state, number, text):
    task_state.progress_events.append((task_state, number, text))


def test_sampling_callback_shape_is_forwarded_unchanged() -> None:
    forwarded = []

    def sampler_callback(step, x0, x, total_steps, y):
        forwarded.append((step, x0, x, total_steps, y))

    callback = SDXLAssemblyProgressCallback(SimpleNamespace(), sampler_callback)
    payload = (object(), object(), object())
    callback(2, payload[0], payload[1], 6, payload[2])

    assert forwarded == [(2, payload[0], payload[1], 6, payload[2])]


def test_legacy_progressbar_shape_is_adapted_without_five_argument_error() -> None:
    global _active_task
    state = SimpleNamespace(current_progress=12)
    active_task = SimpleNamespace(state=state)
    state.progress_events = []
    _active_task = active_task

    callback = SDXLAssemblyProgressCallback(
        SimpleNamespace(image_index=0, image_count=1),
        progressbar,
    )
    callback(2, None, None, 6, None)

    assert state.progress_events == [
        (
            state,
            12,
            "Sampling step 3/6, image 1/1 ...",
        )
    ]


def test_explicit_progress_state_avoids_global_task_lookup() -> None:
    state = SimpleNamespace(current_progress=27, progress_events=[])
    callback = SDXLAssemblyProgressCallback(
        SimpleNamespace(image_index=1, image_count=3),
        progressbar,
        progress_state=state,
    )

    callback(0, None, None, 18, None)

    assert state.progress_events == [
        (state, 27, "Sampling step 1/18, image 2/3 ...")
    ]
