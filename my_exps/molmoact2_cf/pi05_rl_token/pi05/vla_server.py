"""Frozen pi0.5 as an RL-Token state source: action chunk plus final-layer tokens.

RL Token needs two things from the frozen VLA on every step: the reference action chunk
the actor refines, and the token sequence the phase-1 encoder compresses into z_rl. The
existing server does this for MolmoAct2; this is the same HTTP contract backed by pi0.5,
so rlt/vla.py talks to it unchanged.

Where the tokens come from. pi0.5 runs PaliGemma once over the prefix -- images and
language -- to build a KV cache, then denoises the action chunk with a smaller expert
against that cache. The prefix pass is the VLA's reading of the scene, so its last hidden
state is what serves as the RL state. It is captured with a forward hook rather than by
editing openpi.

Two shapes differ from MolmoAct2 and matter downstream:

  token width  2048 here (gemma_2b) against 2560 there, so the phase-1 encoder has to be
               retrained rather than reused;
  actions      returned as joint *deltas*, the space the model was fine-tuned in. The
               conversion to the absolute targets MolmoSpaces executes belongs at the
               environment boundary, so the actor refines deltas and nothing has to undo
               a conversion first.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

log = logging.getLogger(__name__)

# gemma_2b width; asserted at runtime rather than trusted.
PI05_TOKEN_DIM = 2048


class FrozenPi05:
    """Wraps an openpi policy and exposes the prefix tokens alongside the actions."""

    def __init__(self, policy, *, token_dim: int = PI05_TOKEN_DIM) -> None:
        self._policy = policy
        self._token_dim = token_dim
        self._captured: dict[str, torch.Tensor] = {}
        self._handles: list = []
        self._attach()

    def _model(self):
        # openpi wraps the module; reach the PI0Pytorch underneath.
        for attr in ("_model", "model"):
            m = getattr(self._policy, attr, None)
            if m is not None and hasattr(m, "paligemma_with_expert"):
                return m
        raise RuntimeError("could not locate PI0Pytorch inside the policy")

    def _attach(self) -> None:
        model = self._model()
        language_model = model.paligemma_with_expert.paligemma.language_model

        # openpi calls language_model.forward(...) explicitly rather than calling the
        # module, and register_forward_hook only fires on __call__. A hook here would
        # never run and the RL state would be silently absent, so the bound method is
        # wrapped instead.
        original = language_model.forward

        def capturing_forward(*args, **kwargs):
            out = original(*args, **kwargs)
            hidden = getattr(out, "last_hidden_state", None)
            if torch.is_tensor(hidden) and hidden.ndim == 3:
                # The prefix pass is far longer than the denoising passes that follow,
                # so keeping the longest picks the scene reading.
                previous = self._captured.get("hidden")
                if previous is None or hidden.shape[1] >= previous.shape[1]:
                    self._captured["hidden"] = hidden.detach()
            return out

        language_model.forward = capturing_forward
        self._handles.append((language_model, original))
        log.info("token capture installed on PaliGemma language model")

    def predict(self, observation: dict, *, want_tokens: bool = True) -> dict:
        """Run one inference and return the actions, and the tokens if asked for.

        Actions come back in the space the model produces: joint DELTAS. Converting them
        into the absolute targets MolmoSpaces executes is the client's job, because only
        the client knows which arm state each action of a chunk resolves against.

        The token sequence is (968, 2048) float32, about 7.9 MB, and moving it off the
        GPU and onto the wire costs more than the inference on steps where nothing reads
        it. `want_tokens=False` skips that entirely; the forward hook still runs, it is
        just not paid for.
        """
        self._captured.clear()
        result = self._policy.infer(observation)

        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]

        if not want_tokens:
            return {"actions": actions}

        hidden = self._captured.get("hidden")
        if hidden is None:
            raise RuntimeError(
                "the forward hook captured no hidden states; pi0.5's internals changed "
                "and the RL state would silently be missing"
            )
        tokens = hidden[0].to(torch.float32).cpu().numpy()
        if tokens.shape[-1] != self._token_dim:
            raise RuntimeError(
                f"token width {tokens.shape[-1]} but {self._token_dim} was expected; "
                "the phase-1 encoder is built for a fixed width"
            )

        # pi0.5's prefix is dense -- every position is a real image or language token --
        # so the mask is all ones. It is still returned so the contract matches the
        # MolmoAct2 server, whose prefix is padded.
        mask = np.ones(tokens.shape[0], dtype=np.float32)
        return {"actions": actions, "token_features": tokens, "token_attention_mask": mask}

    def close(self) -> None:
        for module, original in self._handles:
            module.forward = original
        self._handles.clear()
