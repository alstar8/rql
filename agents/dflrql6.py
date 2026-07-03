import copy
from typing import Any

from functools import partial
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections as mlc
import optax
from einops import rearrange, repeat

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, Actor, Value


class GradientGuidance(nn.Module):
    """Flow-point-conditioned guidance head: (s, x, f) -> W in R^d.

    W(s, x, f) is trained to match the unit-normalized action gradient of
    the target critic ensemble mean at the same point, so at inference it
    provides an amortized, pointwise grad_x Q direction without backprop
    through the deep critic.
    """

    hidden_dims: Any
    action_dim: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, obs_actions_times):
        w = MLP(
            (*self.hidden_dims, self.action_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(obs_actions_times)
        return w


class DFLRQL6Agent(flax.struct.PyTreeNode):
    """DFL-RQL v6: ensemble-consensus guidance (schedule-free, transferable).

    Change vs v5: the hardcoded step ramp (500k/700k, tuned to this env's
    learning curve) and the unit-direction normalization (discards all
    confidence information) are both replaced by one mathematically grounded
    object — the ensemble consensus vector

        T(s, x, f) = (1/K) * sum_k  g_k / ||g_k||,   g_k = grad_x Q_k(s, x, f)

    over the K target-critic ensemble members. ||T|| <= 1 always; its
    direction is the ensemble's average improvement direction and its norm
    is the *directional agreement*:
      - early in training members are independently initialized, their
        gradient directions are near-independent -> E||T||^2 ~ 1/K, so
        guidance is automatically weak while the critic is untrustworthy;
      - as TD training aligns members, ||T|| -> 1 where they agree.
    This yields per-state, per-point, per-time maturity gating with no
    step-schedule hyperparameters, and a principled pessimism mechanism:
    where members disagree (OOD actions), guidance vanishes.

    W(s, x, f) distills T with a single backward pass per training step:
    each batch element regresses onto unit(g_k) for one uniformly sampled
    member k; the conditional mean of that target is exactly T, so the
    MSE-optimal head converges to the consensus vector (amortization does
    the ensemble averaging).

    At use, W is projected into the unit ball (the mathematical range of T
    — not a tuned cap):

        dx/df = v_theta(x, s, f) + guidance_coef * f * Proj_{||.||<=1} W(s, x, f)
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def guidance_direction(self, observations, actions, times):
        """Time-gated consensus guidance at the flow point.

        W approximates the ensemble consensus vector T (||T|| <= 1), so its
        norm carries the ensemble's directional agreement. Projection into
        the unit ball only enforces the range of the target.
        """
        w = self.network.select("guidance")(
            jnp.concatenate([observations, actions, times], -1)
        )
        w_norm = jnp.linalg.norm(w, axis=-1, keepdims=True)
        w = w * jnp.minimum(1.0, 1.0 / (w_norm + 1e-6))
        return self.config["guidance_coef"] * times * w

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        batch_size, action_dim = self.config["batch_size"], self.config["action_dim"]

        rng, n_rng, u_rng, r_rng = jax.random.split(rng, 4)

        next_state = jnp.concatenate([batch["observations"][-1], jax.random.normal(n_rng, (batch_size, action_dim)), jnp.zeros((batch_size, 1,))], axis=-1) # s', x_0', 0 
        next_qs = self.network.select("target_value")(next_state)
        next_q = next_qs.mean(axis=0) - self.config["rho"] * next_qs.std(axis=0)

        d = jnp.concatenate(
            [
                jax.random.uniform(u_rng, (batch_size // 2,)),
                jax.random.randint(r_rng, (batch_size // 2,), 0, self.config["flow_steps"] + 1)
                / self.config["flow_steps"],
            ],
            0,
        )
        d_b = d / self.config["flow_steps"]

        actions = rearrange(batch["actions"][: self.config["h"]], "h b d -> b (h d)")

        # Reversing flow (with the same guided dynamics used at inference).
        x_f = jnp.copy(actions)
        f = jnp.ones((batch_size, 1))
        for i in range(self.config["flow_steps"]):
            fm_actor = jnp.concatenate([batch["observations"][0], x_f, f], -1)
            out = self.network.select("actor")(fm_actor).mode()
            out = out + self.guidance_direction(batch["observations"][0], x_f, f)
            x_f = x_f - out * d_b[..., None]
            f = f - d_b[..., None]

        state = jnp.concatenate([batch["observations"][0], jax.lax.stop_gradient(x_f), f], axis=-1) # s, x_f, f

        q = self.network.select("value")(state, params=grad_params)

        rs_terminals = jnp.concatenate([jnp.zeros_like(batch["terminals"][:1]), batch["terminals"][:-1]], axis=0) # right shift terminals
        n_rews = (
            batch["rewards"]
            * self.config["discount_mul"][..., None]
            * (1 - rs_terminals)
        ).sum(0)
        tqt_q  = (
            n_rews
            + (self.config["discount"] ** (self.config["h"]))
            * next_q
            * batch["masks"][-2]
        )
        
        s = rs_terminals.sum(0)
        valids = (
            (s <= 1).astype(s.dtype) # 1 for term and shift removes second (h=1)
        )
        critic_loss = (self.expectile_loss(tqt_q - q, tqt_q - q, self.config["expectile"]) * valids).mean()

        # BC flow loss
        rng, x_rng, t_rng, k_rng = jax.random.split(rng, 4)
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = rearrange(batch["actions"][: self.config["h"]], "h b d -> b (h d)")
        t = jax.random.uniform(t_rng, (batch_size, 1))

        x_t = (1 - t) * x_0 + t * x_1
        tgt = x_1 - x_0
        fm_actor = jnp.concatenate([batch["observations"][0], x_t, t], axis=-1)
        pred = self.network.select("actor")(
            fm_actor, params=grad_params
        ).mode()
        # One-step lookahead under the guided dynamics. Guidance is
        # stop-gradded here: the actor loss shapes v_theta only, while W is
        # trained by the gradient distillation loss below.
        guided_pred = pred + jax.lax.stop_gradient(
            self.guidance_direction(batch["observations"][0], x_t, t)
        )
        q_pe = self.network.select("value")(
                jnp.concatenate([batch["observations"][0], 
                    x_t + guided_pred
                    * jnp.minimum(1 / self.config["flow_steps"], 1 - t), 
		    jnp.clip(t + 1 / self.config["flow_steps"], max=1)], axis=-1)
                )
        q_pe = q_pe.mean(axis=0)

        ac_mask = repeat(1 - rs_terminals[:-1], 'h b -> b (h r)', r=self.config["action_dim"] // self.config["h"]) # mask repeated actions at end of ep
        bc_loss = (jnp.square(pred - tgt) * ac_mask).mean()
        actor_loss = -(q_pe * valids).mean()

        # Consensus distillation loss: each batch element regresses
        # W(s, x_t, t) onto unit(grad_x Q_k(s, x_t, t)) for one uniformly
        # sampled target-critic ensemble member k. The conditional mean of
        # this target is the consensus vector T = (1/K) sum_k unit(g_k),
        # so the MSE-optimal W converges to T (direction = consensus,
        # norm = directional agreement in [0, 1]) with a single backward
        # pass. Critic backprop happens only here, never during flow
        # integration.
        member = jax.random.randint(
            k_rng, (batch_size,), 0, self.config["ensemble_ct"]
        )
        member_onehot = jax.nn.one_hot(member, self.config["ensemble_ct"])  # (B, K)

        def q_member_sum(x):
            q_in = jnp.concatenate([batch["observations"][0], x, t], axis=-1)
            qs = self.network.select("target_value")(q_in)  # (K, B)
            return (qs * member_onehot.T).sum()

        q_grad = jax.lax.stop_gradient(jax.grad(q_member_sum)(x_t))
        q_grad_dir = q_grad / (jnp.linalg.norm(q_grad, axis=-1, keepdims=True) + 1e-6)

        w = self.network.select("guidance")(
            jnp.concatenate([batch["observations"][0], x_t, t], -1), params=grad_params
        )
        distill_loss = (jnp.square(w - q_grad_dir).sum(-1) * valids).mean()

        total_loss = (
            actor_loss + bc_loss * self.config["alpha"] + critic_loss
            + distill_loss * self.config["distill_coef"]
        )

        return total_loss, {
            "total_loss": total_loss,
            "actor_loss": actor_loss,
            "bc_loss": bc_loss,
            "q": q.mean(),
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),

            "q_pe_mean": q_pe.mean(),
            "q_pe_max": q_pe.max(),
            "q_pe_min": q_pe.min(),

            "distill_loss": distill_loss,
            # ||W|| ~ ensemble directional agreement: the emergent guidance
            # strength (replaces the step ramp of v4/v5).
            "w_norm": jnp.linalg.norm(w, axis=-1).mean(),
            "w_norm_max": jnp.linalg.norm(w, axis=-1).max(),
            "w_grad_cos": (w * q_grad_dir).sum(-1).mean()
            / (jnp.linalg.norm(w, axis=-1).mean() + 1e-6),
        }

    def target_update(self, network, module_name, d):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * d + tp * (1 - d),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        self.target_update(new_network, "value", d=self.config["tau"])
        self.target_update(new_network, "actor", d=1-self.config["ema"])

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=("temperature",))
    def compute_flow_actions(
        self,
        observations,
        noise,
        seed=None,
        temperature=0.0,
    ):
        actions = noise
        for i in range(self.config["flow_steps"]):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config["flow_steps"])
            fm_actor = jnp.concatenate([observations, actions, t], axis=-1)
            out = self.network.select("actor" if temperature > 0 else "target_actor")(fm_actor).mode()
            out = out + self.guidance_direction(observations, actions, t)
            actions = actions + (out / self.config["flow_steps"])
        actions = jnp.clip(actions, -1, 1)
        return actions

    @partial(jax.jit, static_argnames=("temperature",))
    def sample_actions(
        self,
        obs,
        seed=None,
        temperature=0.0,
    ):
        action_rng, n_rng = jax.random.split(seed)

        obs = jnp.atleast_2d(obs)[-1:]
        noise = jax.random.normal(
            n_rng,
            (
                1,
                self.config["action_dim"],
            ),
        )
        actions = self.compute_flow_actions(obs, seed=action_rng, noise=noise, temperature=temperature)[0]
        actions = rearrange(actions, "(h d) -> h d", h=self.config["h"])
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_actions = jnp.concatenate([ex_actions] * config["h"], -1)
        ex_times = ex_actions[..., :1]
        ex_in = jnp.concatenate([ex_observations, ex_actions, ex_times], -1)
        # Guidance is conditioned on (s, x, f) — same input layout as the actor.
        ex_guidance_in = jnp.concatenate([ex_observations, ex_actions, ex_times], -1)
        action_dim = ex_actions.shape[-1]

        value_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["ensemble_ct"],
        )

        actor_def = Actor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            layer_norm=config["actor_layer_norm"],
            tanh_squash=False,
            state_dependent_std=True,
            const_std=False,
            final_fc_init_scale=1,
        )

        guidance_def = GradientGuidance(
            hidden_dims=config["guidance_hidden_dims"],
            action_dim=action_dim,
            layer_norm=config["layer_norm"],
        )

        network_info = dict(
            value=(value_def, (ex_in,)),
            target_value=(copy.deepcopy(value_def), (ex_in)),
            actor=(actor_def, (ex_in,)),
            target_actor=(
                copy.deepcopy(actor_def),
                (ex_in,)
            ),
            guidance=(guidance_def, (ex_guidance_in,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_value"] = params["modules_value"]
        params["modules_target_actor"] = params["modules_actor"]
        config["action_dim"] = action_dim

        config["discount_mul"] = jnp.array(
            config["discount"] ** jnp.array(list(range(config["h"])) + [jnp.inf])
        )

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = mlc.ConfigDict(
        dict(
            agent_name="dflrql6",
            h=3,
            alpha=1.0,
            expectile=0.5,
            ensemble_ct=10,
            rho=0.0, 
            lr=3e-4,
            discount=0.99,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            guidance_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            tau=0.005,
            ema=0.999,
            flow_steps=10,
            q_agg="mean",
            guidance_coef=0.5,
            distill_coef=1.0,
        )
    )
    return config
