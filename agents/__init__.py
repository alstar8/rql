from agents.ar_qdfl_fast_sac import ARQDFLFastSACAgent
from agents.consensus_discrete_flow import ConsensusDiscreteFlowAgent
from agents.consensus_latent_flow import ConsensusLatentFlowAgent
from agents.discrete_ar_qdfl_distill import DiscreteARQdflDistillAgent
from agents.discrete_ar_iql import DiscreteARIQLAgent
from agents.discrete_coord_mask_iql import DiscreteCoordMaskIQLAgent
from agents.discrete_diffusion_qdfl_distill import (
    DiscreteDiffusionQdflDistillAgent,
)
from agents.dflrql import DFLRQLAgent
from agents.dflrql2 import DFLRQL2Agent
from agents.dflrql3 import DFLRQL3Agent
from agents.dflrql4 import DFLRQL4Agent
from agents.dflrql5 import DFLRQL5Agent
from agents.dflrql6 import DFLRQL6Agent
from agents.dflrql7 import DFLRQL7Agent
from agents.dflrql8 import DFLRQL8Agent
from agents.dflrql9 import DFLRQL9Agent
from agents.dflrql10 import DFLRQL10Agent
from agents.dflrql11 import DFLRQL11Agent
from agents.dflrql12 import DFLRQL12Agent
from agents.pi05_cf import Pi05CFAgent
from agents.quantized_dflrql9 import QuantizedDFLRQL9Agent
from agents.qflow import QFlowAgent
from agents.qflow_rql_warmstart import QFlowRQLWarmstartAgent
from agents.rql import RQLAgent
from agents.rql_qflow import RQLQFlowAgent
from agents.rql_qflow_terminal import RQLQFlowTerminalAgent

agents = dict(
    ar_qdfl_fast_sac=ARQDFLFastSACAgent,
    rql=RQLAgent,
    rql_qflow=RQLQFlowAgent,
    rql_qflow_terminal=RQLQFlowTerminalAgent,
    qflow=QFlowAgent,
    qflow_rql_warmstart=QFlowRQLWarmstartAgent,
    dflrql=DFLRQLAgent,
    dflrql2=DFLRQL2Agent,
    dflrql3=DFLRQL3Agent,
    dflrql4=DFLRQL4Agent,
    dflrql5=DFLRQL5Agent,
    dflrql6=DFLRQL6Agent,
    dflrql7=DFLRQL7Agent,
    dflrql8=DFLRQL8Agent,
    dflrql9=DFLRQL9Agent,
    dflrql10=DFLRQL10Agent,
    dflrql11=DFLRQL11Agent,
    dflrql12=DFLRQL12Agent,
    pi05_cf=Pi05CFAgent,
    quantized_dflrql9=QuantizedDFLRQL9Agent,
    consensus_discrete_flow=ConsensusDiscreteFlowAgent,
    consensus_latent_flow=ConsensusLatentFlowAgent,
    discrete_coord_mask_iql=DiscreteCoordMaskIQLAgent,
    discrete_ar_iql=DiscreteARIQLAgent,
    discrete_ar_qdfl_distill=DiscreteARQdflDistillAgent,
    discrete_diffusion_qdfl_distill=DiscreteDiffusionQdflDistillAgent,
)
