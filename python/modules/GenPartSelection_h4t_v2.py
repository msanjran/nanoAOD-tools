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
from utils import deltaR,deltaPhi

class GenPartSelection_h4t_v2(Module):

    def __init__(self,
                 inputCollection=lambda event: Collection(event, "GenPart"),
                 outputName="selectedGenParts",
                 storeKinematics=['phi', 'eta', 'pt', 'pdgId'],
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

        # general GenPart information
        self.out.branch("GenPart_pdgId", "I", lenVar="nGenPart")
        self.out.branch("GenPart_genPartIdxMother", "I", lenVar="nGenPart")
        self.out.branch("GenPart_statusFlags", "I", lenVar="nGenPart")
        self.out.branch("GenPart_phi", "F", lenVar="nGenPart")
        self.out.branch("GenPart_eta", "F", lenVar="nGenPart")
        self.out.branch("GenPart_pt", "F", lenVar="nGenPart")
        
        # for four-top
        self.out.branch(self.outputName + "_isL4t", "I") # n Leptonic tops
        self.out.branch(self.outputName + "_isH4t", "I") # n Hadronic tops
        self.out.branch(self.outputName + "_isO4t", "I") # n Other tops (should always be 4)
        self.out.branch("n"+self.outputName+"_t1", "I")
        self.out.branch("n"+self.outputName+"_t2", "I")
        self.out.branch("n"+self.outputName+"_t3", "I")
        self.out.branch("n"+self.outputName+"_t4", "I")

        for variable in self.storeKinematics:
            self.out.branch(self.outputName+"_t1_"+variable, "F", lenVar="n"+self.outputName+"_t1")
            self.out.branch(self.outputName+"_t2_"+variable, "F", lenVar="n"+self.outputName+"_t2")
            self.out.branch(self.outputName+"_t3_"+variable, "F", lenVar="n"+self.outputName+"_t3")
            self.out.branch(self.outputName+"_t4_"+variable, "F", lenVar="n"+self.outputName+"_t4")
    
    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass


    def getGenTop(self, event):
        ''' new and upgraded '''

        genParticles = self.inputCollection(event)

        def findTopIdx(p):
            motherIdx = p.genPartIdxMother
            while (motherIdx>=0):
                if abs(genParticles[motherIdx].pdgId)==6:
                    return motherIdx
                motherIdx = genParticles[motherIdx].genPartIdxMother
            return -1
        
        def checkMother(p, pdg):
            ''' check mother particle above p is pdg or not '''
            mother_idx = p.genPartIdxMother
            mother_pdg = genParticles[mother_idx].pdgId
            particle_pdg = p.pdgId

            # goes back up decay chain until it's something else
            while mother_pdg == particle_pdg:
                mother_idx = genParticles[mother_idx].genPartIdxMother
                mother_pdg = genParticles[mother_idx].pdgId

            # check if the something else is pdg we want (e.g., 24 for W boson)
            if abs(mother_pdg) == abs(pdg):
                return True
            else:
                return False
        
        def findLastCopy(p, pdg, depth):
            ''' recursive function to find last copy of a particle''' 
            lastCopy_remaining = False
            isLastCopy_flag = False

            if isLastCopy(p):
                lastCopy_remaining = True
                isLastCopy_flag = True
                finalCopy_dict[p] = {'depth':depth, 'isLastCopy':isLastCopy_flag, 'mother':p.genPartIdxMother}
                return lastCopy_remaining, p, depth, isLastCopy_flag
            
            remaining_genParticles = map(lambda part_i: part_i, genParticles)[p._index+1:]
            remaining_pdg_list_np = np.array([part_i.pdgId for part_i in remaining_genParticles])
            remaining_mother_list_np = np.array([part_i.genPartIdxMother for part_i in remaining_genParticles])
            remaining_index_list_np = np.array([part_i._index for part_i in remaining_genParticles])

            # 1. check if any daughters
            # 2. if not then last copy
            # 3. if there are, then check pdg id and hard process
            # 4. if 0, last copy
            # 5. if 1, repeat function for that one with that p._index
            # 6. if >1, repeat function for all of the potential daughters and choose one with maximum depth
            any_daughters = np.where(remaining_mother_list_np == p._index)[0]

            # > 0 daughters
            if len(any_daughters) > 0:

                # same pdg and fromHardProcess requirement
                any_daughters_pdg = remaining_pdg_list_np[any_daughters] == pdg
                any_daughters_list = remaining_index_list_np[any_daughters][any_daughters_pdg]
                any_daughters_list_hardprocess = [ fromHardProcess( genParticles[p_idx] ) for p_idx in any_daughters_list ]
                # if none meet requirements --> lastCopy
                if sum(any_daughters_list_hardprocess) == 0:
                    lastCopy_remaining = True
                    finalCopy_dict[p] = {'depth':depth, 'isLastCopy':isLastCopy_flag, 'mother':p.genPartIdxMother}
                    return lastCopy_remaining, p, depth, isLastCopy_flag
                
                # if one meets requirement --> recursively check if this is the last copy
                elif sum(any_daughters_list_hardprocess) == 1:
                    any_daughters_idx = int( any_daughters_list[any_daughters_list_hardprocess] )
                    return findLastCopy( genParticles[any_daughters_idx], pdg, depth + 1)
                
                # if > 1 meets requirement --> recursively check all possibilities and choose deepest one or the 'isLastCopy' one (priority)
                elif sum(any_daughters_list_hardprocess) > 1:
                    any_daughters_idx_list = any_daughters_list[any_daughters_list_hardprocess]
                    # print("Particle {particle_index}, pdg {particle_odg}".format(p._index, p.pdgId))
                    daughter_finalCopy_idx = []
                    daughter_finalCopy_depth = []
                    daughter_finalCopy_isLastCopy = []

                    # branching to handle multiple hard process copies
                    for i, weird_daughter in enumerate(any_daughters_idx_list):
                        finalCopy_bool, finalCopy_idx, finalCopy_depth, finalCopy_isLastCopy = findLastCopy(genParticles[weird_daughter], pdg, depth + 1 )
                        daughter_finalCopy_idx.append(finalCopy_idx._index)
                        daughter_finalCopy_depth.append(finalCopy_depth)
                        daughter_finalCopy_isLastCopy.append(finalCopy_isLastCopy)
                    
                    # check if any of the branched daughters are isLastCopy flagged
                    daughter_isLastCopy = np.where( np.array(daughter_finalCopy_isLastCopy) == True )
                    if len(daughter_isLastCopy[0]) == 1:
                        lastCopy_remaining = True
                        daughter_finalCopy = daughter_finalCopy_idx[int(daughter_isLastCopy[0])]
                        daughter_depth = daughter_finalCopy_depth[int(daughter_isLastCopy[0])]
                        daughter_isLastCopy_flag = True
                        return lastCopy_remaining, genParticles[daughter_finalCopy], daughter_depth, daughter_isLastCopy_flag
                    
                    # shouldn't really be possible
                    elif len(daughter_isLastCopy[0]) > 1:
                        lastCopy_remaining = True
                        daughter_finalCopy = daughter_finalCopy_idx[int(daughter_isLastCopy[0][0])]
                        daughter_depth = daughter_finalCopy_depth[int(daughter_isLastCopy[0][0])]
                        daughter_isLastCopy_flag = True
                        return lastCopy_remaining, genParticles[daughter_finalCopy], daughter_depth, daughter_isLastCopy_flag
                    
                    # if none isLastCopy flagged --> move onto 'deepest' one in the chain
                    else:
                        daughter_max = np.where( np.array(daughter_finalCopy_depth) == np.array(daughter_finalCopy_depth).max() )

                        # if one of these, cool
                        if len(daughter_max[0]) == 1:
                            lastCopy_remaining = True
                            daughter_finalCopy = daughter_finalCopy_idx[int( daughter_max[0] )]
                            daughter_depth = daughter_finalCopy_depth[int( daughter_max[0] )]
                            daughter_isLastCopy_flag = False
                            return lastCopy_remaining, genParticles[daughter_finalCopy], daughter_depth, daughter_isLastCopy_flag
                        
                        # if more than one, ffs choose first one
                        elif len(daughter_max[0]) > 1:
                            lastCopy_remaining = True
                            daughter_finalCopy = daughter_finalCopy_idx[int(daughter_max[0][0])]
                            daughter_depth = daughter_finalCopy_depth[int(daughter_max[0][0])]
                            daughter_isLastCopy_flag = False
                            return lastCopy_remaining, genParticles[daughter_finalCopy], daughter_depth, daughter_isLastCopy_flag
                        
                        # if 0 of these, doesn't make sense really
                        else:
                            print("What the shit, this shouldn't happen")
            
            # no daughters
            else:
                lastCopy_remaining = True
                finalCopy_dict[p] = {'depth':depth, 'isLastCopy':isLastCopy_flag, 'mother':p.genPartIdxMother} 
                return lastCopy_remaining, p, depth, isLastCopy_flag
            

        topDict = {}

        # find all tops
        for topIdx,genParticle in enumerate(genParticles):
            if abs(genParticle.pdgId)==6:
                topDict[topIdx] = {'top': genParticle, 'bquark':[], 'lepton': [], 'neutrino':[], 'quarks': []} # maybe add W boson too?

        # find associated particles for the top quarks
        # maybe turn the isFirstCopy into a separate gen filter?
        for genParticle in genParticles:
            if abs(genParticle.pdgId)==5 and isFirstCopy(genParticle) and fromHardProcess(genParticle):
                topIdx = findTopIdx(genParticle)
                check_mother = checkMother(genParticle, 6)
                if topIdx>=0 and topIdx in topDict.keys() and check_mother == True:
                    finalCopy_dict = {}
                    #finalCopy = findLastCopy(genParticle, genParticle.pdgId, 0)[1]
                    finalCopyBool, finalCopy, finalCopyDepth, finalCopyIsLastCopy = findLastCopy(genParticle, genParticle.pdgId, 0)
                    topDict[topIdx]['bquark'].append(finalCopy)
                    
            elif abs(genParticle.pdgId)<5 and abs(genParticle.pdgId)>0 and isFirstCopy(genParticle) and fromHardProcess(genParticle):
                topIdx = findTopIdx(genParticle)
                check_mother = checkMother(genParticle, 24)
                if topIdx>=0 and topIdx in topDict.keys() and check_mother == True:
                    finalCopy_dict = {}
                    # finalCopy = findLastCopy(genParticle, genParticle.pdgId, 0)[1]
                    finalCopyBool, finalCopy, finalCopyDepth, finalCopyIsLastCopy = findLastCopy(genParticle, genParticle.pdgId, 0)
                    topDict[topIdx]['quarks'].append(finalCopy)

            elif abs(genParticle.pdgId) in [11,13,15] and fromHardProcess(genParticle) and isFirstCopy(genParticle):
                topIdx = findTopIdx(genParticle)
                check_mother = checkMother(genParticle, 24)
                if topIdx>=0 and topIdx in topDict.keys() and check_mother == True:
                    finalCopy_dict = {}
                    # finalCopy = findLastCopy(genParticle, genParticle.pdgId, 0)[1]
                    finalCopyBool, finalCopy, finalCopyDepth, finalCopyIsLastCopy = findLastCopy(genParticle, genParticle.pdgId, 0)
                    topDict[topIdx]['lepton'].append(finalCopy)
                    
            elif abs(genParticle.pdgId) in [12,14,16] and fromHardProcess(genParticle) and isFirstCopy(genParticle):
                topIdx = findTopIdx(genParticle)
                check_mother = checkMother(genParticle, 24)
                if topIdx>=0 and topIdx in topDict.keys() and check_mother == True:
                    finalCopy_dict = {}
                    # finalCopy = findLastCopy(genParticle, genParticle.pdgId, 0)[1]
                    finalCopyBool, finalCopy, finalCopyDepth, finalCopyIsLastCopy = findLastCopy(genParticle, genParticle.pdgId, 0)
                    topDict[topIdx]['neutrino'].append(finalCopy)

        genTopList = []

        for idx in topDict.keys():
            top = topDict[idx]
            if len(top['quarks'])==2 and len(top['lepton'])==0 and len(top['neutrino'])==0 and len(top['bquark'])==1:
                genTopList.append({'top':top, 'type':'hadronic'})
            elif len(top['quarks'])==0 and len(top['lepton'])==1 and len(top['neutrino'])==1 and len(top['bquark'])==1:
                genTopList.append({'top':top, 'type':'leptonic'})
            else:
                genTopList.append({'top':top, 'type':'other'})

        return genTopList


    def saveGenInfo(self, event):
        """save all gen information, for later processing (looking at decay chains)"""
        genParticles = self.inputCollection(event)
        
        self.out.fillBranch("GenPart_pdgId", map(lambda genPart: getattr(genPart, "pdgId"), genParticles ))
        self.out.fillBranch("GenPart_genPartIdxMother", map(lambda genPart: getattr(genPart, "genPartIdxMother"), genParticles ))
        self.out.fillBranch("GenPart_statusFlags", map(lambda genPart: getattr(genPart, "statusFlags"), genParticles ))
        self.out.fillBranch("GenPart_phi", map(lambda genPart: getattr(genPart, "phi"), genParticles ))
        self.out.fillBranch("GenPart_eta", map(lambda genPart: getattr(genPart, "eta"), genParticles ))
        self.out.fillBranch("GenPart_pt", map(lambda genPart: getattr(genPart, "pt"), genParticles ))


    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""

        # get gen particle information (for later processing and sanity checking)
        self.saveGenInfo(event) 

        genTopList = self.getGenTop(event)

        isH4t = 0
        isL4t = 0
        isO4t = 0
        isHorL = 1
        for top in genTopList:

            # if we have more than 4 useful tops, we'll count these still but won't add it to the t1, t2, t3, t4 tops...
            # essentiall save as weird events
            if isHorL > 4:
                if top['type'] == 'hadronic':
                    isH4t += 1
                    isHorL += 1
                elif top['type'] == 'leptonic':
                    isL4t += 1
                    isHorL += 1
                continue

            if top['type'] == 'hadronic':
                isH4t += 1

                genHadronicParts_i = [top['top']['top'], top['top']['bquark'][0], top['top']['quarks'][0], top['top']['quarks'][1]]
                i_Name = "_t"+str(isHorL)+"_"
                self.out.fillBranch("n"+self.outputName+"_t"+str(isHorL), len(genHadronicParts_i))

                for variable in self.storeKinematics:
                    self.out.fillBranch( self.outputName+i_Name+variable, map(lambda genPart: getattr(genPart, variable), genHadronicParts_i) )

                isHorL += 1
                
            elif top['type'] == 'leptonic':
                isL4t += 1

                genLeptonicParts_i = [top['top']['top'], top['top']['bquark'][0], top['top']['lepton'][0], top['top']['neutrino'][0]]
                i_Name = "_t"+str(isHorL)+"_"
                self.out.fillBranch("n"+self.outputName+"_t"+str(isHorL), len(genLeptonicParts_i))

                for variable in self.storeKinematics:
                    self.out.fillBranch( self.outputName+i_Name+variable, map(lambda genPart: getattr(genPart, variable), genLeptonicParts_i) )

                isHorL += 1

            elif top['type'] == 'other':
                isO4t += 1

        # if we end up having less than 4 useful tops
        # need to fill in remaining t1, t2, t3, t4
        if isHorL < 4:

            remaining_branches = 4 - isHorL
            null_list = [0.]

            for i in range(remaining_branches):
                self.out.fillBranch("n"+self.outputName+"_t"+str(isHorL+i+1), 0)
                i_Name = "_t"+str(isHorL+i+1)+"_"
                for variable in self.storeKinematics:
                    self.out.fillBranch(self.outputName+i_Name+variable, null_list)

        # fill in extra branches
        self.out.fillBranch(self.outputName + "_isL4t", isL4t) # n Leptonic tops
        self.out.fillBranch(self.outputName + "_isH4t", isH4t) # n Hadronic tops
        self.out.fillBranch(self.outputName + "_isO4t", isO4t) # n Other tops

        return True
