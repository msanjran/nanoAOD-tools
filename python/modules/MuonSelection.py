import os
import sys
import math
import json
import ROOT
import random

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

from utils import getGraph, getHist, combineHist2D, getSFXY, deltaR

class MuonSelection(Module):
    VERYTIGHT = 1
    TIGHT = 1
    MEDIUM = 2
    LOOSE = 3
    NONE = 4
    INV = 5

    def __init__(
        self,
        inputCollection=lambda event: Collection(event, "Muon"),
        #outputName="tightMuons",
        outputName_list=["tightMuons","mediumMuons","looseMuons"],
        triggerMatch=False,
        #muonID=TIGHT,
        #muonIso=TIGHT,
        muonMinPt=25.,
        muonMaxEta=2.4,
        storeKinematics=['pt','eta'],
        #storeWeights=False,
    ):
        
        self.inputCollection = inputCollection
        self.outputName = outputName
        self.muonMinPt = muonMinPt
        self.muonMaxEta = muonMaxEta
        self.storeKinematics = storeKinematics
        #self.storeWeights = storeWeights
        self.triggerMatch = triggerMatch
        self.triggerObjectCollection = lambda event: Collection(event, "TrigObj") if triggerMatch else lambda event: []
        

    def triggerMatched(self, muon, trigger_object):
        if self.triggerMatch:
            trig_deltaR = math.pi
            for trig_obj in trigger_object:
                if abs(trig_obj.id) != 13:
                    continue
                trig_deltaR = min(trig_deltaR, deltaR(trig_obj, muon))
            if trig_deltaR < 0.3:
                return True
            else:
                return False
        else:
            return True    
 
    def beginJob(self):
        pass
        
    def endJob(self):
        pass
        
    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.out = wrappedOutputTree
        
        for var in ['tightID', 'mediumID', 'looseID']:
            self.out.branch("muon_"+var,"F",lenVar="nMuon")
            
        for outputName in self.outputName_list:
            self.out.branch("n"+outputName, "I")
            
            for variable in self.storeKinematics:
                self.out.branch(outputName+"_"+variable,"F",lenVar="n"+outputName)
        
    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass
        
    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        muons = self.inputCollection(event)

        triggerObjects = self.triggerObjectCollection(event)

        selectedMuons = {'tight': [], 'medium': [], 'loose': []}
        unselectedMuons = []
        
        muonID = {'tight': [], 'medium': [], 'loose': []}
        nMuon = 0
        
        #https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2#Tight_Muon
        for muon in muons:
            if muon.pt>self.muonMinPt \
            and math.fabs(muon.eta)<self.muonMaxEta \
            and self.muonIdFct(muon) \
            and self.muonIsoFct(muon) \
            and self.triggerMatched(muon, triggerObjects):
            
                #saving relIso, cutBased Id
                nMuon+=1
                muonID['tight'].append(muon.tightId)
                muonID['medium'].append(muon.mediumId)
                muonID['loose'].append(muon.looseId)
                
                if muon.tightId==1:
                    selectedMuons['tight'].append(muon)
                    selectedMuons['medium'].append(muon)
                    selectedMuons['loose'].append(muon)
                elif muon.mediumId==1:
                    selectedMuons['medium'].append(muon)
                    selectedMuons['loose'].append(muon)
                elif muon.looseId==1:
                    selectedMuons['loose'].append(muon)
                else:
                    unselectedMuons.append(muon)
                    
                selectedMuons.append(muon)
            else:
                unselectedMuons.append(muon)
                
        self.out.fillBranch("nMuon",nMuon)
        for wp in muonID.keys():
            self.out.fillBranch("muon_"+wp+"ID", map(lambda id: id, muonID[wp]))
            
        for outputName, muon_ID in zip(self.outputName_list, ['tight','medium','loose']):
            self.out.fillBranch("n"+outputName,len(selectedMuons[muon_ID]))
        
            for variable in self.storeKinematics:
                self.out.fillBranch(outputName+"_"+variable,map(lambda muon: getattr(muon,variable),selectedMuons[muon_ID]))
                
            setattr(event,outputName,selectedMuons[muon_ID])
            
        setattr(event,"unselectedMuons",unselectedMuons)
        setattr(event,'nMuon',nMuon)

        return True

