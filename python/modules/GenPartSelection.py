# WIP: proof of principle to show that i can run SPA-Net on it

import os
import sys
import math
import json
import ROOT
import random
import numpy as np

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

from gen_helper import isHardProcess, isLastCopy, isPrompt, fromHardProcess
from utils import deltaR,deltaPhi

class GenPartSelection(Module):

    def __init__(self,
                 inputCollection=lambda event: Collection(event, "GenPart"),
                 outputName="selectedGenParts",
                 storeKinematics=['phi', 'eta', 'pdgId'],
                 ):
        self.inputCollection = inputCollection
        self.outputName = outputName
        self.storeKinematics = storeKinematics

    def beginJob(self):
        pass

    def endJob(self):
        pass
    
    # create branches that we want to add to output file
    # branch(branchname, typecode, lenVar) 
    # lenVar = name of variable holding length of array branches e.g. nElectron
    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.out = wrappedOutputTree
        
        # value should really be 6 for both of these
        # self.out.branch("n"+self.outputName+"t1", "I")
        # self.out.branch("n"+self.outputName+"t2", "I")

        # for ttbar
        #for variable in self.storeKinematics:
        #    self.out.branch(self.outputName+"_t1_"+variable, "F", lenVar="n"+self.outputName)
        #    self.out.branch(self.outputName+"_t2_"+variable, "F", lenVar="n"+self.outputName)

        self.out.branch("n"+self.outputName+"_t1", "I")
        self.out.branch("n"+self.outputName+"_t2", "I")
        #self.out.branch("n"+self.outputName+"_t3", "I")
        #self.out.branch("n"+self.outputName+"_t4", "I")
        

        # for four-top
        for variable in self.storeKinematics:
            self.out.branch(self.outputName+"_t1_"+variable, "F", lenVar="n"+self.outputName)
            self.out.branch(self.outputName+"_t2_"+variable, "F", lenVar="n"+self.outputName)
            #self.out.branch(self.outputName+"_t3_"+variable, "F", lenVar="n"+self.outputName)
            #self.out.branch(self.outputName+"_t4_"+variable, "F", lenVar="n"+self.outputName)
    
    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass

    def getGenTop(self, event):
        
        genParticles = self.inputCollection(event)
        
        def findTopIdx(p):
            motherIdx = p.genPartIdxMother
            while (motherIdx>=0):
                if abs(genParticles[motherIdx].pdgId)==6:
                    return motherIdx
                motherIdx = genParticles[motherIdx].genPartIdxMother
            return -1
            
        topDict = {}
        
        #NB: following will not handle hadronic taus cases
        # goes through all gen particles and gets index of top
        # top quark index corresponds to index in topDict
        for topIdx,genParticle in enumerate(genParticles):
            if abs(genParticle.pdgId)==6:
                topDict[topIdx] = {'top': genParticle, 'bquark':[], 'lepton': [], 'neutrino':[], 'quarks': []}
        
        # find associated parent top quark (if there is one)
        for genParticle in genParticles:
            if abs(genParticle.pdgId)==5 and isLastCopy(genParticle) and fromHardProcess(genParticle):
                topIdx = findTopIdx(genParticle)
                if topIdx>=0 and topIdx in topDict.keys():
                    topDict[topIdx]['bquark'].append(genParticle)
                    
            elif abs(genParticle.pdgId)<5 and abs(genParticle.pdgId)>0 and isLastCopy(genParticle) and fromHardProcess(genParticle):
                topIdx = findTopIdx(genParticle)
                if topIdx>=0 and topIdx in topDict.keys():
                    topDict[topIdx]['quarks'].append(genParticle)

            elif abs(genParticle.pdgId) in [11,13] and isPrompt(genParticle) and isLastCopy(genParticle):
                topIdx = findTopIdx(genParticle)
                if topIdx>=0 and topIdx in topDict.keys():
                    topDict[topIdx]['lepton'].append(genParticle)
                    
            elif abs(genParticle.pdgId) in [12,14] and isPrompt(genParticle) and isLastCopy(genParticle):
                topIdx = findTopIdx(genParticle)
                if topIdx>=0 and topIdx in topDict.keys():
                    topDict[topIdx]['neutrino'].append(genParticle)
            
        hadronicTops = []
        #leptonicTops = []
            
        for idx in topDict.keys():
            top = topDict[idx]
            if len(top['quarks'])==2 and len(top['lepton'])==0 and len(top['neutrino'])==0 and len(top['bquark'])==1:
                top['quarks'] = sorted(top['quarks'],key=lambda x: x.pt, reverse=True) #sort by pT like jets
                hadronicTops.append(top) # appending a dictionary
            #elif len(top['quarks'])==0 and len(top['lepton'])==1 and len(top['neutrino'])==1 and len(top['bquark'])==1:
            #    top['quarks'] = sorted(top['quarks'],key=lambda x: x.pt, reverse=True) #sort by pT like jets
            #    leptonicTops.append(top) # appending a dictionary
        
        # should put a test on whether the top quarks originate from ttbarttbar
        #return {"hadronic": hadronicTops, "leptonic": leptonicTops}
        return {"hadronic": hadronicTops}

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""

        # get hadronic and leptonic tops with the signal daughter particles
        genHadronicTops = self.getGenTop(event)["hadronic"]
        #genLeptonicTops = self.getGenTop(event)["leptonic"]

        # make sure this event is the signal
        #if len(genHadronicTops) != 3 and len(genLeptonicTops) != 1:
        #    return False
        if len(genHadronicTops) != 2:
            return False

        # in for loop in case we want more leptonic tops
        # but currently should only loop once
        #for i, genLeptonicTop_i in enumerate(genLeptonicTops): 
        #    genLeptonicParts_i = [genLeptonicTop_i['top'], genLeptonicTop_i['bquark'][0], genLeptonicTop_i['lepton'][0], genLeptonicTop_i['neutrino'][0]]
        #    i_Name = "_t"+str(i+1)+"_" # t1
        #    self.out.fillBranch("n"+self.outputName+"_t"+str(i+1), len(genLeptonicParts_i))
        #    # fillBranch( branchname, value )
        #    for genPart in genLeptonicParts_i:
        #        for variable in self.storeKinematics:
        #            self.out.fillBranch(
        #                self.outputName+i_Name+variable,
        #                map(lambda genPart: getattr(genPart, variable), genLeptonicParts_i)
        #            )
        
        for i, genHadronicTop_i in enumerate(genHadronicTops):
            genHadronicParts_i = [genHadronicTop_i['top'], genHadronicTop_i['bquark'][0], genHadronicTop_i['quarks'][0], genHadronicTop_i['quarks'][1]]
            i_Name = "_t"+str(i+1)+"_" # t1, t2
            self.out.fillBranch("n"+self.outputName+"_t"+str(i+1), len(genHadronicParts_i))
            # fillBranch( branchname, value )
            for genPart in genHadronicParts_i:
                for variable in self.storeKinematics:
                    self.out.fillBranch(
                        self.outputName+i_Name+variable,
                        map(lambda genPart: getattr(genPart, variable), genHadronicParts_i)
                    )
        
        return True