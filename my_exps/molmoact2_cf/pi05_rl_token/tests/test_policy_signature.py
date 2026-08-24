"""How MolmoSpaces constructs a policy, checked without MolmoSpaces.

`run_evaluation` with more than one worker builds the policy in each worker as
``policy_cls(exp_config, task)``. If anything but `task` owns the second
position, the task object is silently bound to it -- which is how a run failed
with ``'PickTask' object has no attribute 'act'`` after every single-process
smoke test had passed.
"""

from __future__ import annotations

import inspect


def test_task_owns_the_second_positional_parameter():
    from rlt.rl_policy import RLTokenPolicy

    parameters = list(inspect.signature(RLTokenPolicy.__init__).parameters.values())
    names = [p.name for p in parameters]
    assert names[:3] == ["self", "exp_config", "task"]


def test_the_injected_objects_are_keyword_only():
    from rlt.rl_policy import RLTokenPolicy

    parameters = inspect.signature(RLTokenPolicy.__init__).parameters
    for name in ("vla", "encoder", "agent", "chunk", "stride", "explore", "on_decision"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, name
