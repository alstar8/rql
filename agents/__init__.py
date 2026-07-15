from agents.consensus_discrete_flow import ConsensusDiscreteFlowAgent
from agents.consensus_latent_flow import ConsensusLatentFlowAgent
from agents.dflrql import DFLRQLAgent
from agents.dflrql2 import DFLRQL2Agent
from agents.dflrql3 import DFLRQL3Agent
from agents.dflrql4 import DFLRQL4Agent
from agents.dflrql5 import DFLRQL5Agent
from agents.dflrql6 import DFLRQL6Agent
from agents.dflrql7 import DFLRQL7Agent
from agents.dflrql8 import DFLRQL8Agent
from agents.dflrql9 import DFLRQL9Agent
from agents.rql import RQLAgent

agents = dict(
    rql=RQLAgent,
    dflrql=DFLRQLAgent,
    dflrql2=DFLRQL2Agent,
    dflrql3=DFLRQL3Agent,
    dflrql4=DFLRQL4Agent,
    dflrql5=DFLRQL5Agent,
    dflrql6=DFLRQL6Agent,
    dflrql7=DFLRQL7Agent,
    dflrql8=DFLRQL8Agent,
    dflrql9=DFLRQL9Agent,
    consensus_discrete_flow=ConsensusDiscreteFlowAgent,
    consensus_latent_flow=ConsensusLatentFlowAgent,
)
