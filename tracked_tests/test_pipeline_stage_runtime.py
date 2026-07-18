import os
import sys
from types import SimpleNamespace

sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.pipeline.stage_runtime import (
    PipelineResourceRequirement,
    PipelineRoute,
    PipelineRouteContext,
    PipelineStage,
    PipelineStageResult,
    PipelineStageRunner,
)


class RecordingStage(PipelineStage):
    def __init__(self, stage_id, events, *, route_complete=False):
        self.stage_id = stage_id
        self.phase_name = 'task'
        self._events = events
        self._route_complete = route_complete

    def describe_resources(self, context):
        return (
            PipelineResourceRequirement(
                resource_id=f'{self.stage_id}_resource',
                description=f'Resource for {self.stage_id}',
                resource_type='artifact',
            ),
        )

    def execute(self, context):
        self._events.append(f'execute:{self.stage_id}')
        return PipelineStageResult(route_complete=self._route_complete, notes={'stage': self.stage_id})

    def finalize(self, context, *, result=None, error=None):
        self._events.append(f'finalize:{self.stage_id}')


def test_stage_runner_records_stage_metadata_and_stops_on_route_complete(capsys):
    events = []
    route = PipelineRoute(
        route_id='test',
        family='test',
        display_name='Test',
        stages=[
            RecordingStage('first', events, route_complete=True),
            RecordingStage('second', events),
        ],
    )
    context = PipelineRouteContext(
        async_task=SimpleNamespace(),
        task_state=SimpleNamespace(goals=[]),
        route_id='test',
        route_family='test',
    )

    PipelineStageRunner().run(route, context)
    output = capsys.readouterr().out

    assert events == ['execute:first', 'finalize:first']
    assert '[Residency]' in output
    assert 'required=' in output
    assert ' pinned=' not in output
    assert context.route_complete is True
    assert [record.stage_id for record in context.executed_stages] == ['first']
    assert context.executed_stages[0].resources[0].resource_id == 'first_resource'
    assert 'residency_required' in context.executed_stages[0].notes
