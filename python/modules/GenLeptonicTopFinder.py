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

from gen_helper import isHardProcess, isLastCopy, isPrompt, fromHardProcess, isFirstCopy
from gen_helper_shahzad import countTops

class GenLeptonicTopFinder(Module):

    def __init__(self,
                 inputCollection=lambda event: Collection(event, "GenPart"),
                 outputName="selectedGenLeptonicTops",
                 ):
        self.inputCollection = inputCollection
        self.outputName = outputName

    def beginJob(self):
        pass

    def endJob(self):
        pass
    
    # create branches that we want to add to output file
    # branch(branchname, typecode, lenVar) 
    # lenVar = name of variable holding length of array branches e.g. nElectron
    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.out = wrappedOutputTree

        # general GenPart information
        self.out.branch(self.outputName + "_nLeptonicTops", "I")
        self.out.branch(self.outputName + "_nHadronicTops", "I")

    
    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass

#    def countTops(self, event):
#
#        genParticles = self.inputCollection(event)
#
#        # method 1
#        # find n leptonic tops
#        # limitations:
#        # if there are multiple lepton copies pointing to same W boson --> might get more than we expect
#        lepTops = 0
#        hadTops = 0
#        topDict = {}
#        for genParticle in genParticles:
#
#            # if lepton
#            if (abs(genParticle.pdgId) in [11,13,15] or genParticle.pdgId in [1,2,3,4,5]) and fromHardProcess(genParticle) and isFirstCopy(genParticle):
#                mother_idx = genParticle.genPartIdxMother
#                if mother_idx < 0:
#                    continue
#
#                # select lepton if mother directly == W boson
#                mother_pdg = genParticles[mother_idx].pdgId
#                if abs(mother_pdg) != 24:
#                    continue
#
#                # find grandmother
#                grandmother_idx = genParticles[mother_idx].genPartIdxMother
#                grandmother_pdg = genParticles[grandmother_idx].pdgId
#
#                # make sure grandmother isn't just a W boson copy
#                while grandmother_pdg == mother_pdg:
#                    grandmother_idx = genParticles[grandmother_idx].genPartIdxMother
#                    grandmother_pdg = genParticles[grandmother_idx].pdgId
#
#                # make sure grandmother is a top quark
#                if abs(grandmother_pdg) == 6:
#                    topDict[genParticle._index] = {'index': genParticle._index,
#                                                   'pdg': genParticle.pdgId,
#                                                   'mother index':mother_idx,
#                                                   'mother pdg':mother_pdg}
#                    if abs(genParticle.pdgId) in [11,13,15]:
#                        lepTops += 1
#                    else:
#                        hadTops += 1
#
#        # could print for all, but printing for > 4 to exemplify the problem in less verbose output
#        if lepTops + hadTops != 4:
#            print("method 1 limitation, (lep tops, had tops)", lepTops, hadTops)
#            for key in topDict:
#                print("method 1 limitation, gen particle", topDict[key])
        
        # method 2
        # find n hadronic tops via W boson decays
        # limitation: if top quark has multiple W boson copies that each produce a quark
        # limitation: if W boson has multiple copies that each produce a quark
        # too many recursive loops needed
#        alternativeTops = 0
#        totalTops = 0
#        alternativeTopsLep = 0
#        for genParticle in genParticles:
#
#            # if W boson
#            if abs(genParticle.pdgId) in [24]: # and isFirstCopy(genParticle) and fromHardProcess(genParticle):
#                mother_idx = genParticle.genPartIdxMother
#                if mother_idx < 0:
#                    continue
#
#                # check the mother is top quark
#                mother_pdg = genParticles[mother_idx].pdgId
#                if abs(mother_pdg) != 6:
#                    totalTops += 1
#                    continue
#
#                # check decay of W boson
#                isHadDecay = False
#                isLepDecay = False
#                loop_copy = genParticle
#                for genParticle2 in genParticles:
#                    if genParticle._index == genParticle2.genPartIdxMother:
#                        #print(genParticle._index, genParticle2._index, genParticle.pdgId, genParticle2.pdgId)
#                        if genParticle2.pdgId == genParticle.pdgId:
#
#                        if abs(genParticle2.pdgId) < 6:
#                            isHadDecay = True
#                        elif abs(genParticle2.pdgId) in [11,12,13,14,15,16]:
#                            isLepDecay = True
#                        else:
#                            print(genParticle._index, genParticle2._index, genParticle.pdgId, genParticle2.pdgId)
#                if isHadDecay == True:
#                     alternativeTops += 1
#
#        # print(alternativeTops, nTops)
#        if alternativeTops + nTops != totalTops:
#            print("hadronic: ", alternativeTops, "leptonic: ", nTops, "total: ",  totalTops)
#        return lepTops


    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        
        genParticles = self.inputCollection(event)
        
        nLeptonicTops, nHadronicTops = countTops(genParticles)

        self.out.fillBranch(self.outputName + "_nLeptonicTops", nLeptonicTops) # n leptonic tops
        self.out.fillBranch(self.outputName + "_nHadronicTops", nHadronicTops) # n leptonic tops

        return True
