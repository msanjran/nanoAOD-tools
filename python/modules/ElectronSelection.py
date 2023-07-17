import os
import sys
import math
import json
import ROOT
import random
import numpy as np

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

from utils import getHist, getSFXY, deltaR
from collections import OrderedDict

class ElectronSelection(Module):
    WP80 = 1
    WP90 = 2
    INV = 3
    NONE = 4

    def __init__(
        self,
        inputCollection = lambda event: Collection(event, "Electron"),
        # outputName = "tightElectrons",
        triggerMatch=False,
        iso_type=[],
        electronID = WP80,
        electronMinPt = 29.,
        electronMaxEta = 2.4,
        storeKinematics=['pt','eta'],
        storeWeights=False,
    ):

        self.inputCollection = inputCollection
        # self.outputName = outputName
        self.iso_type = []
        self.electronMinPt = electronMinPt
        self.electronMaxEta = electronMaxEta
        self.storeKinematics = storeKinematics
        self.storeWeights = storeWeights
        self.triggerMatch = triggerMatch
        
        self.triggerObjectCollection = lambda event: Collection(event, "TrigObj") if triggerMatch else lambda event: []
        
        self.outputName_dict = OrderedDict()
        for iso in iso_type:
            self.outputName_dict[id] = OrderedDict()
            for wp in ['tight', 'medium', 'loose']:
                self.outputName_dict[id][wp] = wp+'_'+iso+'_'+'Electrons'

    def triggerMatched(self, electron, trigger_object):
        if self.triggerMatch:
            trig_deltaR = math.pi
            for trig_obj in trigger_object:
                if abs(trig_obj.id) != 11:
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
        # self.out.branch("n"+self.outputName, "I")
        for wp in ['tight','medium','loose']:
            self.out.branch("electron_MVA_Iso_"+wp+"ID", "F", lenVar="nElectron")
            self.out.branch("electron_MVA_noIso_"+wp+"ID", "F", lenVar="nElectron")
        
        for iso_type in self.iso_type:
            for wp in self.outputName_dict[id_type].keys():
                self.out.branch("n"+self.outputName_dict[iso_type][wp], "I")
                for variable in self.storeKinematics:
                    self.out.branch(self.outputName_dict[id_type][wp]+"_"+variable,"F",lenVar="n"+self.outputName_dict[id_type][wp])


    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        electrons = self.inputCollection(event)
        muons = Collection(event, "Muon")
        triggerObjects = self.triggerObjectCollection(event)

        selectedElectrons = OrderedDict([("Iso", OrderedDict([("tight", []), ("medium",[]), ("loose",[])])), ("noIso", OrderedDict([("tight", []), ("medium",[]), ("loose",[])])) ])
        unselectedElectrons = OrderedDict([("Iso", []), ("noIso", []) ])
        
        electronIsoId = {'tight': [], 'medium': [], 'loose': []}
        electronNoIsoId = {'tight': [], 'medium': [], 'loose': []}

        for electron in electrons:
            # https://twiki.cern.ch/twiki/bin/view/CMS/CutBasedElectronIdentificationRun2
            if electron.pt>self.electronMinPt \
            and math.fabs(electron.eta)<self.electronMaxEta \
            and self.triggerMatched(electron, triggerObjects):

                dxy = math.fabs(electron.dxy)
                dz = math.fabs(electron.dz)
                
                if math.fabs(electron.eta) < 1.479 and (dxy>0.05 or dz>0.10):
                    # unselectedElectrons.append(electron)
                    continue
                elif dxy>0.10 or dz>0.20:
                    # unselectedElectrons.append(electron)
                    continue

                #reject electron if close-by muon
                if len(muons)>0:
                    mindr = min(map(lambda muon: deltaR(muon, electron), muons))
                    if mindr < 0.05:
                        # unselectedElectrons.append(electron)
                        continue
                        
                # save mvaFall17V2Iso and mvaFall17V2noIso
                nElectron += 1
                electronIsoId['tight'].append(electron.mvaFall17V2Iso_WP80)
                electronIsoId['medium'].append(electron.mvaFall17V2Iso_WP90)
                electronIsoId['loose'].append(electron.mvaFall17V2Iso_WPL)
                electronNoIsoId['tight'].append(electron.mvaFall17V2noIso_WP80)
                electronNoIsoId['medium'].append(electron.mvaFall17V2noIso_WP90)
                electronNoIsoId['loose'].append(electron.mvaFall17V2noIso_WPL)

                # selectedElectrons.append(electron)
                
                for iso_type in self.iso_type:
                    if iso_type=='Iso':
                        if electron.mvaFall17V2Iso_WP80==1:
                            selectedElectrons[id_type]['tight'].append(electron)
                            selectedElectrons[id_type]['medium'].append(electron)
                            selectedElectrons[id_type]['loose'].append(electron)
                        elif electron.mvaFall17V2Iso_WP90==1:
                            selectedElectrons[id_type]['medium'].append(electron)
                            selectedElectrons[id_type]['loose'].append(electron)
                        elif electron.mvaFall17V2Iso_WPL==1:
                            selectedElectrons[id_type]['loose'].append(electron)
                        else:
                            unselectedElectrons[id_type].append(electron)
                    elif id_type=='noIso':
                        if electron.mvaFall17V2noIso_WP80==1:
                            selectedElectrons[id_type]['tight'].append(electron)
                            selectedElectrons[id_type]['medium'].append(electron)
                            selectedElectrons[id_type]['loose'].append(electron)
                        elif electron.mvaFall17V2noIso_WP90==1:
                            selectedElectrons[id_type]['medium'].append(electron)
                            selectedElectrons[id_type]['loose'].append(electron)
                        elif electron.mvaFall17V2noIso_WPL==1:
                            selectedElectrons[id_type]['loose'].append(electron)
                        else:
                            unselectedElectrons[id_type].append(electron)

            else:
                continue

            self.out.branch("electron_MVA_Iso_"+wp+"ID", "F", lenVar="nElectron")
            self.out.branch("electron_MVA_noIso_"+wp+"ID", "F", lenVar="nElectron")
            
            
        self.out.fillBranch("nElectron",nElectron)
        for wp in ['tight', 'medium', 'loose']:
            self.out.fillBranch("electron_MVA_Iso_"+wp+"ID", map(lambda id: id, electronIsoId[wp]))
            self.out.fillBranch("electron_MVA_noIso_"+wp+"ID", map(lambda id: id, electronNoIsoId[wp]))
            
        for iso_type in self.iso_type:
            for wp in self.outputName_dict[id_type].keys():
                self.out.fillBranch("n"+self.outputName_dict[id_type][wp], len(selectedElectrons[id_type][wp]))
                
                for variable in self.storeKinematics:
                    self.out.fillBranch(self.outputName_dict[id_type][wp]+"_"+variable,map(lambda electron: getattr(electron,variable),selectedElectrons[id_type][wp]))
                    
                setattr(event,self.outputName_dict[id_type][wp],selectedElectrons[id_type][wp])
            setattr(event,"unselectedElectrons",unselectedElectrons[id_type])
        
        return True
