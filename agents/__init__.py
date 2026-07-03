from agents.dflrql import DFLRQLAgent
from agents.dflrql2 import DFLRQL2Agent
from agents.dflrql3 import DFLRQL3Agent
from agents.dflrql4 import DFLRQL4Agent
from agents.dflrql5 import DFLRQL5Agent
from agents.dflrql6 import DFLRQL6Agent
from agents.rql import RQLAgent

agents = dict(
    rql=RQLAgent,
    dflrql=DFLRQLAgent,
    dflrql2=DFLRQL2Agent,
    dflrql3=DFLRQL3Agent,
    dflrql4=DFLRQL4Agent,
    dflrql5=DFLRQL5Agent,
    dflrql6=DFLRQL6Agent,
)
