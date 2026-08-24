"""The policy must talk to the server it was given.

`MolmoAct2_Policy.__init__` health-checks whatever address sits in the config
and refuses to start if nothing answers. Collectors each get their own client,
so the config has to follow the client -- otherwise every collector except the
one matching the default dies at startup, which is what happened the first time
the servers moved off port 8000.
"""

from __future__ import annotations

from rlt.rl_policy import point_config_at
from rlt.vla import VlaClient


class Config:
    """Stands in for the pydantic policy config: only this field is read."""

    remote_config = {"host": "localhost", "port": 8000}


def test_client_exposes_where_it_points():
    client = VlaClient("somehost", 8021)
    assert (client.host, client.port) == ("somehost", 8021)
    assert client.url == "http://somehost:8021/act"


def test_config_follows_the_client():
    config = Config()
    point_config_at(config, VlaClient("localhost", 8031))
    assert config.remote_config == {"host": "localhost", "port": 8031}
