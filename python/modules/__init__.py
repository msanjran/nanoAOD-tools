import ROOT
import sys
#can only load this once
if (ROOT.gSystem.Load("libPhysicsToolsNanoAODTools.so")!=0):
    print "Cannot load 'libPhysicsToolsNanoAODTools'"
    sys.exit(1)

#muons
from SingleMuonTriggerSelection import SingleMuonTriggerSelection
from MuonSelection import MuonSelection
from MuonVeto import MuonVeto

#electrons
from SingleElectronTriggerSelection import SingleElectronTriggerSelection
from ElectronSelection import ElectronSelection
from ElectronVeto import ElectronVeto

#aux
from EventSkim import EventSkim
from MetFilter import MetFilter
from EventInfo import EventInfo

#jets
from JetMetUncertainties import JetMetUncertainties
from JetSelection import JetSelection
from BTagSelection import BTagSelection
from btagSFProducer import btagSFProducer

#met
from MetSelection import MetSelection

#event
from PUWeightProducer import puWeightProducer, PUWeightProducer_dict
from GenWeightProducer import GenWeightProducer
from TopPtWeightProducer import TopPtWeightProducer

#reco
from EventObservables import EventObservables
from TopNNReco import TopNNRecoInputs

#lhe weights
from LHEWeightProducer import LHEWeightProducer

#gen particles
from GenPartSelection import GenPartSelection
from GenPartSelection_h2t import GenPartSelection_h2t
from GenPartSelection_h3t import GenPartSelection_h3t
from GenPartSelection_h4t import GenPartSelection_h4t
from GenParticleModuleGabriele import GenParticleModule
from GenPartSelection_h4t_v2 import GenPartSelection_h4t_v2
from GenParticleModuleGabriele_v2 import GenParticleModule_v2
from GenTopFinder import GenTopFinder
from GenPartSelection_h2t_v2 import GenPartSelection_h2t_v2
