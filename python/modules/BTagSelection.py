import os
import sys
import math
import json
import ROOT
import random

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

from utils import deltaR, deltaPhi
from collections import OrderedDict


class BTagSelection(Module):
    #tight DeepFlav WP (https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation2016Legacy)
    LOOSE=0
    MEDIUM=1
    TIGHT=2
    
    def __init__(
        self,
        btaggingWP = {},
        inputCollection=lambda event: Collection(event, "Jet"),
        flagName = "isBTagged",
        outputName_list=[], #"btaggedJets",
        jetMinPt=30.,
        jetMaxEta=2.4,
        workingpoint = [],
        storeKinematics=['pt', 'eta'],
        storeTruthKeys=[]
    ):
        self.btaggingWP = btaggingWP
        self.inputCollection = inputCollection
        #self.flagName = flagName
        self.outputName_list = outputName_list
        self.jetMinPt = jetMinPt
        self.jetMaxEta = jetMaxEta
        self.storeKinematics = storeKinematics
        self.storeTruthKeys = storeTruthKeys
        self.workingpoint = workingpoint
            
    def beginJob(self):
        pass

    def endJob(self):
        pass

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.out = wrappedOutputTree
        
        for outputName in self.outputName_list:
            self.out.branch("n"+outputName, "I")
            for variable in self.storeKinematics:
                self.out.branch(outputName+"_"+variable, "F", lenVar="n"+outputName)

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""

        jets = self.inputCollection(event)
        
        bJets = OrderedDict([('tight',[]), ('medium',[]), ('loose',[])])
        lJets = []


        for jet in jets:
            if jet.pt<self.jetMinPt:
                lJets.append(jet)
                continue
        
            if math.fabs(jet.eta) > self.jetMaxEta:
                lJets.append(jet)
                continue
                
            if jet.btagDeepFlavB>self.btaggingWP[2]:
                bJets['tight'].append(jet)
                setattr(jet,"is_tightBTagged",True)
            
            if jet.btagDeepFlavB>self.btaggingWP[1]:
                bJets['medium'].append(jet)
                setattr(jet,"is_mediumBTagged",True)
                
            if jet.btagDeepFlavB>self.btaggingWP[0]:
                bJets['loose'].append(jet)
                setattr(jet,"is_looseBTagged",True)
               
            # bJets.append(jet)
            
        #for jet in bJets:
        #    setattr(jet,self.flagName,True)
        #for jet in lJets:
        #    setattr(jet,self.flagName,False)
        
        for outputName, bJet_type in zip(self.outputName_list, ['tight', 'medium', 'loose']):
        
            self.out.fillBranch("n"+outputName, len(bJets[bJet_type]))

            for variable in self.storeKinematics:
                self.out.fillBranch(outputName+"_"+variable, map(lambda jet: getattr(jet, variable), bJets[bJet_type]))

                setattr(event, outputName, bJets[bJet_type])
        
        self.out.fillBranch("n"+self.outputName, len(bJets))

        return True

