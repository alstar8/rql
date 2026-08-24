"""RL Token (arXiv 2604.23073) on MolmoAct2, in MolmoSpaces.

Phase 1 -- the RL token:

    token_ae.py       encoder/decoder, L_ro                Eq. 1, 2
    train_token_ae.py phase 1 training
    data.py           token-shard reader + memmap cache
    serve_vla.py      frozen-VLA HTTP server

Phases 2-3 -- online actor-critic (Algorithm 1):

    networks.py       actor and twin critic
    agent.py          the two loss functions                Eq. 3, 5
    replay.py         chunk transitions and the buffer
    vla.py            VLA client + frozen token encoder -> z_rl
    rl_policy.py      the MolmoSpaces policy that rolls out and records
    rollout.py        one benchmark episode at a time
    learner.py        Algorithm 1's inner update loop
    train_online.py   the training entry point
    evaluate.py       success rates, per episode or on the benchmark

Shared: config.py (every hyperparameter), cli.py, logging_utils.py,
mlspaces_env.py.
"""

__all__ = ["config", "token_ae", "data", "logging_utils"]
