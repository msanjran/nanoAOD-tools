import heapq
import json
import math
import os
import random
import sys
from collections import OrderedDict

import ROOT
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import \
    Collection
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

from gen_helper import *
from utils import PhysicsObject, deltaR


class GenParticleModule(Module):

    def __init__(
        self,
        inputGenCollection=lambda event: Collection(event, "GenPart"),
        #inputFatGenJetCollection=lambda event: Collection(event, "GenJetAK8"),
        #inputGenJetCollection=lambda event: Collection(event, "GenJet"),
        #inputFatJetCollection=lambda event: Collection(event, "FatJet"),
        #inputJetCollection=lambda event: Collection(event, "Jet"),
        # inputHOTVRJetCollection = lambda event: Collection(event, "HOTVRJet"),
        # inputSubHOTVRJetCollection = lambda event: Collection(event, "SubHOTVRJet"),
        #inputMuonCollection=lambda event: Collection(event, "Muon"),
        #inputElectronCollection=lambda event: Collection(event, "Electron"),
        outputName="genPart",
        storeKinematics= ['pt','eta','phi','mass'],
    ):
        
        self.inputGenCollection = inputGenCollection
        # self.inputFatGenJetCollection = inputFatGenJetCollection
        # self.inputGenJetCollection = inputGenJetCollection
        # self.inputFatJetCollection = inputFatJetCollection
        # self.inputJetCollection = inputJetCollection
        # self.inputHOTVRJetCollection = inputHOTVRJetCollection
        # self.inputSubHOTVRJetCollection = inputSubHOTVRJetCollection
        # self.inputMuonCollection = inputMuonCollection
        # self.inputElectronCollection = inputElectronCollection
        self.outputName = outputName
        self.storeKinematics = storeKinematics

        self.print_out = False

        self.genP_lastcopies, self.genP_lastcopies_pdgid, self.genP_lastcopies_id, self.genP_lastcopies_mother_idx = [], [], [], []
        self.oldToNew, self.new_motherIdx_list = OrderedDict(), []

        # self.genTopKeys = ['has_hadronically_decay', 'is_inside_ak8', 'is_inside_ak8_top_tagged', 'is_inside_hotvr', 'is_inside_hotvr_top_tagged', 'inside_nak8', 'inside_nhotvr', 'min_deltaR_ak8', 'min_deltaR_hotvr', 'rho_over_pt_hotvr', 'first_daughter', 'second_daughter', 'third_daughter', 'all_decays_inside_hotvr', 'all_decays_inside_ak8', 'max_deltaR_q_ak8', 'max_deltaR_q_hotvr']
        self.genTopKeys = ['has_hadronically_decay', 'first_daughter', 'second_daughter', 'third_daughter']
        if Module.globalOptions['isSignal']: self.genTopKeys.append('from_resonance')

    def is_genP_inside_genFJet(self, genFJet, genP):
        if deltaR(genFJet, genP)<0.8:
            return True
        else: return False

    def is_genP_inside_HOTVRJet(self, hotvr, genP):
        rho = 600   # parameter defined in the paper: https://arxiv.org/abs/1606.04961
        if deltaR(hotvr, genP)< (rho/hotvr.pt):
            return True
        else: return False
    
    def genTop_from_resonance(self, genParticle_idx, event):
        # recursive check if the genTop comes from the resonance; 
        mother_idx = self.inputGenCollection(event)[genParticle_idx].genPartIdxMother
        if mother_idx != -1:
            mother_pdg = self.inputGenCollection(event)[mother_idx].pdgId
            if abs(mother_pdg)==6000055: return True
            else:
                genParticle_idx = mother_idx
                return self.genTop_from_resonance(genParticle_idx, event)
        else: return False

    def genP_from_particle_of_interest(self, genParticle_idx, particle_of_interest, particle_of_interest_idx, event):
        # recursive check if the genP has the particle of interest as mother; if it reaches the origin without finding the particle of interest, return False; else True

        mother_idx = self.inputGenCollection(event)[genParticle_idx].genPartIdxMother
        if mother_idx != -1:
            mother_pdg = self.inputGenCollection(event)[mother_idx].pdgId
            if isLastCopy(self.inputGenCollection(event)[mother_idx]) and abs(mother_pdg)==particle_of_interest and mother_idx==particle_of_interest_idx:
                # if self.print_out: print('The particle inside the jet comes from the decay of {}'.format(particle_of_interest))
                return True
            else:
                # if self.print_out: print('The mother {} of the particle {}[pos. {}] is not a last copy/ not the particle of interest {}... looping back into the decay chain...'.format(mother_pdg, self.inputGenCollection(event)[genParticle_idx].pdgId, genParticle_idx, particle_of_interest))
                genParticle_idx = mother_idx
                return self.genP_from_particle_of_interest(genParticle_idx, particle_of_interest, particle_of_interest_idx, event)
        else: 
            # if self.print_out: print('...back to the origin! The particle inside the jet does not come from any {}'.format(particle_of_interest))
            return False

    def has_X_as_daughter(self, genP, daughter_of_interest, event):
        genparticles = map(lambda genP: genP, self.inputGenCollection(event))[genP._index+1:]
        if len(genparticles)==0:
            # if self.print_out: print('No daughter of pdg id {} found for this genP'.format(daughter_of_interest))
            return False, None
        else:
            for igenP, genparticle in enumerate(genparticles):
                if igenP+1==len(genparticles): return False, None # need a review...not the best option
                if genP._index==genparticle.genPartIdxMother and abs(genparticle.pdgId)==daughter_of_interest:
                    # if self.print_out: print('The {} in pos {} has a daughter {} in pos {} '.format(genP.pdgId, genP._index, genparticle.pdgId, genparticle._index))
                    if isLastCopy(genparticle) or genparticle._index not in list(map(lambda genp: genp.genPartIdxMother, genparticles)): #last argument due to wierd events (check pdf)
                        # if self.print_out: print('which is the last copy')
                        return True, genparticle
                    else: 
                        # if self.print_out: print('which is not the last copy.')
                        genP=genparticle
                        return self.has_X_as_daughter(genP, daughter_of_interest, event)
                else: continue

    def minimum_pairwise_mass(self, subjets_in_hotvr):
        min_pair_sum_mass = float('inf')

        for i, sjet_i in enumerate(subjets_in_hotvr[:2]):  # minimum pairwise mass of the three leading subjets
            for j, sjet_j in enumerate(subjets_in_hotvr[i+1:3]):
                pair_sum = sjet_i.p4() + sjet_j.p4()
                if pair_sum.M() < min_pair_sum_mass:
                    min_pair_sum_mass = pair_sum.M()
        # if self.print_out: print("The minimum sum pair mass is :", min_pair_sum_mass)
        return min_pair_sum_mass

    def hotvr_top_tagged(self, hotvr, subjets_in_hotvr):
        mass_flag = hotvr.mass>140 and hotvr.mass<220
        n_subjet_flag = len(subjets_in_hotvr) > 2
        tau3_over_tau2 = (hotvr.tau3/hotvr.tau2) < 0.56

        if mass_flag and n_subjet_flag and tau3_over_tau2 and self.minimum_pairwise_mass(subjets_in_hotvr)>50 and (subjets_in_hotvr[0].pt/hotvr.pt)<0.8:
            if self.print_out: print('mass, nsubj, tau3/tau2 flag, min pairwise, relPt flag TRUE') 
            return True
        else: return False

    def deltaPhi(self,phi1, phi2):
        res = phi1-phi2
        while (res > math.pi):
            res -= 2 * math.pi
        while (res <= -math.pi):
            res += 2 * math.pi

        return res

    def relDeltaPt(self,pt1, pt2):
        if pt2 != 0.0:
            return abs((pt1-pt2)/pt2)
        return 1.
 
    def beginJob(self):
        pass
        
    def endJob(self):
        pass
        
    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.out = wrappedOutputTree

        self.out.branch("ngenTop","I")
        for genTopKey in self.genTopKeys:
            if 'deltaR' in genTopKey or 'rho_over_pt' in genTopKey: self.out.branch("genTop_"+genTopKey, "F", lenVar="ngenTop")
            else: self.out.branch("genTop_"+genTopKey, "I", lenVar="ngenTop")
        for variable in self.storeKinematics:
            self.out.branch("genTop_"+variable, "F", lenVar="ngenTop")
        # self.out.branch("ngenTop_daughters","I")
        self.out.branch("genTop_daughters_pdgId", "I", lenVar="ngenTop")


    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass
        
    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        
        genParticles = self.inputGenCollection(event)
        # fatGenJets = self.inputFatGenJetCollection(event)
        # fatGenJets = sorted(fatGenJets,key=lambda x: x.pt, reverse=True)
        # genJets = self.inputGenJetCollection(event)
        # genJets = sorted(genJets,key=lambda x: x.pt, reverse=True)
        
        # muons = self.inputMuonCollection(event)
        # electrons = self.inputElectronCollection(event)
        
        # fatjets = self.inputFatJetCollection(event)
        # jets = self.inputJetCollection(event)
        # hotvrjets = self.inputHOTVRJetCollection(event)
        # subhotvrjets = self.inputSubHOTVRJetCollection(event)

        if self.print_out: 
            print('########## EVENT ############')
            print(map(lambda genp: getattr(genp, "pdgId") , genParticles))
            print(map(lambda genp: getattr(genp, "genPartIdxMother") , genParticles))
            cacca, pupu = {},{}
            for pdg, id, num in zip(list(map(lambda genp: genp.pdgId, genParticles)), list(map(lambda genp: genp.genPartIdxMother, genParticles)), range(0, len(genParticles))):
                cacca[num]=pdg
                pupu[num]=id
            print(cacca)
            print(pupu)

        # ---- creating new list of last copies of t, W, b, resonance
        # these list are used to calculated the fraction of energy of t, W, b are inside jets
        quarks_from_W = [1, 2, 3, 4]
        gentops = []
        gentops_daughters = []

        for genParticle in genParticles:
            if isLastCopy(genParticle):
                if abs(genParticle.pdgId)==6:
    
                    if self.print_out: print('GenTop last copy in pos {}'.format(genParticle._index))

                    # ---- check the origin of the genTop (if comes from resonance)
                    if Module.globalOptions['isSignal']:
                        is_top_from_resonance = False
                        setattr(genParticle, 'from_resonance', False)
                        if self.genTop_from_resonance(genParticle._index, event):  
                            is_top_from_resonance = True
                            if self.print_out: print('which comes from resonance!')
                        if is_top_from_resonance: setattr(genParticle, 'from_resonance', True)  

                        if self.print_out: print(genParticle.from_resonance)
                    # ----
                    
                    # ---- genTop daughter search
                    gentop_daughters = [] #{'from_resonance': [], 'not_from_resonance': []}
                    if self.has_X_as_daughter(genParticle, 5, event)[0]: 
                        b_daughter = self.has_X_as_daughter(genParticle, 5, event)[1]
                        if self.print_out: print('{} is a daughter in pos {}'.format(5, self.has_X_as_daughter(genParticle, 5, event)[1]._index))
                        gentop_daughters.append(b_daughter)
                        # if is_top_from_resonance: gentop_daughters['from_resonance'].append(b_daughter)
                        # else: gentop_daughters['not_from_resonance'].append(b_daughter)
                    if self.has_X_as_daughter(genParticle, 24, event)[0]:
                        W_daughter = self.has_X_as_daughter(genParticle, 24, event)[1]
                        if (self.has_X_as_daughter(W_daughter, 1, event)[0] and self.has_X_as_daughter(W_daughter, 2, event)[0]):
                            if self.print_out: print('The quarks u,d are daughters of the W in pos {}'.format(W_daughter._index))
                            d_daughter, u_daughter = self.has_X_as_daughter(W_daughter, 1, event)[1], self.has_X_as_daughter(W_daughter, 2, event)[1]
                            gentop_daughters.extend([d_daughter, u_daughter])
                            # if is_top_from_resonance: gentop_daughters['from_resonance'].extend([d_daughter, u_daughter])
                            # else: gentop_daughters['not_from_resonance'].extend([d_daughter, u_daughter])
                        elif (self.has_X_as_daughter(W_daughter, 3, event)[0] and self.has_X_as_daughter(W_daughter, 4, event)[0]): 
                            if self.print_out: print('The quarks s,c are daughters of the W in pos {}'.format(W_daughter._index))
                            s_daughter, c_daughter = self.has_X_as_daughter(W_daughter, 3, event)[1], self.has_X_as_daughter(W_daughter, 4, event)[1]
                            gentop_daughters.extend([s_daughter, c_daughter])
                            # if is_top_from_resonance: gentop_daughters['from_resonance'].extend([s_daughter, c_daughter])
                            # else: gentop_daughters['not_from_resonance'].extend([s_daughter, c_daughter])
                        elif (self.has_X_as_daughter(W_daughter, 11, event)[0] and self.has_X_as_daughter(W_daughter, 12, event)[0]): 
                            if self.print_out: print('The electron is a daughter of the W in pos {}'.format(W_daughter._index))
                            e_daughter, nu_e_daughter = self.has_X_as_daughter(W_daughter, 11, event)[1], self.has_X_as_daughter(W_daughter, 12, event)[1]
                            gentop_daughters.extend([e_daughter,nu_e_daughter])
                        elif (self.has_X_as_daughter(W_daughter, 13, event)[0] and self.has_X_as_daughter(W_daughter, 14, event)[0]): 
                            if self.print_out: print('The muon is a daughter of the W in pos {}'.format(W_daughter._index))
                            mu_daughter, nu_mu_daughter = self.has_X_as_daughter(W_daughter, 13, event)[1], self.has_X_as_daughter(W_daughter, 14, event)[1]
                            gentop_daughters.extend([mu_daughter,nu_mu_daughter])
                        elif (self.has_X_as_daughter(W_daughter, 15, event)[0]and self.has_X_as_daughter(W_daughter, 16, event)[0]): 
                            if self.print_out: print('The tau is a daughter of the W in pos {}'.format(W_daughter._index))
                            tau_daughter, nu_tau_daughter = self.has_X_as_daughter(W_daughter, 15, event)[1], self.has_X_as_daughter(W_daughter, 16, event)[1]
                            gentop_daughters.extend([tau_daughter,nu_tau_daughter])
                     
                    setattr(genParticle, 'daughters', gentop_daughters)
                    
                    for n_daugther in ['first_daughter','second_daughter','third_daughter']:
                        setattr(genParticle, n_daugther, -99)
                    if len(gentop_daughters)==len(['first_daughter','second_daughter','third_daughter']):
                        for i, daughter in zip(['first_daughter','second_daughter','third_daughter'], gentop_daughters):
                            setattr(genParticle, i, daughter.pdgId)
                        gentops_daughters.append(gentop_daughters)
                    else: 
                        if self.print_out: print("Wierd Event...")


                    if self.print_out: 
                        print('Top pos.[{}] has {} in pos.[{}] as daughters'.format(genParticle._index, list(map(lambda daughter: daughter.pdgId, gentop_daughters)), list(map(lambda daughter: daughter._index, gentop_daughters))))

                    setattr(genParticle, 'has_hadronically_decay', False)
                    if any(top_daughter in list(map(lambda daughter: daughter.pdgId, gentop_daughters)) for top_daughter in quarks_from_W):
                        if self.print_out: print('Top pos.[{}] decays hadronically')
                        setattr(genParticle, 'has_hadronically_decay', True)
                    # ----
   
                    # ---- genStudies with AK8
                    # setattr(genParticle, 'is_inside_ak8', False), setattr(genParticle, 'is_inside_ak8_top_tagged', False)
                    # setattr(genParticle, 'inside_nak8', 0) #, setattr(genParticle, 'min_deltaR_ak8', -99.)
                    # setattr(genParticle,'all_decays_inside_ak8', False), setattr(genParticle,'max_deltaR_q_ak8', -99)

                    # min_deltaR_ak8_top  = float('inf')
                    # for ak8 in fatjets:    
                    #     if self.print_out: print('------------ Beginning loop over reco ak8 jets (that have a link to genAK8Jet) ------------')    
                    #     if self.print_out: print('---- AK8 Number {}'.format((ak8._index)+1))

                    #     associated_genFJet = None
                    #     for genFJet in fatGenJets:
                    #         if ak8.genJetAK8Idx == genFJet._index:
                    #             associated_genFJet = genFJet
                    #         else: continue
                    #     if associated_genFJet == None: 
                    #         if self.print_out: print('No associated genJet')
                    #         continue

                    #     min_deltaR_ak8_top = min(min_deltaR_ak8_top, deltaR(genFJet, genParticle))
                        
                    #     if self.is_genP_inside_genFJet(associated_genFJet, genParticle):
                    #         if self.print_out: print('The genTop is inside an ak8!')
                    #         setattr(genParticle, 'is_inside_ak8', True)
                    #         setattr(genParticle, 'inside_nak8', genParticle.inside_nak8+1)

                    #         max_deltaR_ak8_q = 0.
                    #         for daughter in gentop_daughters:
                    #             max_deltaR_ak8_q = max(max_deltaR_ak8_q, deltaR(associated_genFJet, daughter))
                    #         setattr(genParticle, 'max_deltaR_q_ak8', max_deltaR_ak8_q)

                    #         if max_deltaR_ak8_q < 0.8:
                    #             if self.print_out: print('All the decays are inside the ak8. The greater dR is {}'.format(max_deltaR_ak8_q))
                    #             setattr(genParticle, 'all_decays_inside_ak8', True)
                            

                    #         if ak8.particleNet_TvsQCD>0.58: 
                    #             if self.print_out: print('which is top tagged!')
                    #             setattr(genParticle, 'is_inside_ak8_top_tagged', True)                            
                    #     else: continue

                    # setattr(genParticle, 'min_deltaR_ak8', min_deltaR_ak8_top)
                    # ----

                    # ---- genStudies with HOTVR
                    # setattr(genParticle, 'is_inside_hotvr', False), setattr(genParticle, 'is_inside_hotvr_top_tagged', False)
                    # setattr(genParticle, 'inside_nhotvr', 0), setattr(genParticle, 'rho_over_pt_hotvr', -99)
                    # setattr(genParticle,'all_decays_inside_hotvr', False), setattr(genParticle,'max_deltaR_q_hotvr', -99)
                    # min_deltaR_hotvr_top = float('inf')
                    # for hotvr in hotvrjets:
                    #     if self.print_out: print('------------ Beginning loop over reco hotvr jets ------------')    
                    #     if self.print_out: print('---- HOTVR Number {}'.format((hotvr._index)+1))
                    #     subjets_in_hotvr = []
                    #     for hotvr_subjet in subhotvrjets:
                    #         if hotvr.subJetIdx1==hotvr_subjet._index:
                    #             subjets_in_hotvr.insert(0, hotvr_subjet)
                    #         if hotvr.subJetIdx2==hotvr_subjet._index:  
                    #             subjets_in_hotvr.insert(1, hotvr_subjet)
                    #         if hotvr.subJetIdx3==hotvr_subjet._index:  
                    #             subjets_in_hotvr.insert(2, hotvr_subjet)
                            
                    #     subjets_in_hotvr = sorted(subjets_in_hotvr,key=lambda x: x.pt, reverse=True)

                    #     min_deltaR_hotvr_top = min(min_deltaR_hotvr_top, deltaR(hotvr, genParticle))

                    #     if self.is_genP_inside_HOTVRJet(hotvr, genParticle):
                    #         if self.print_out: print('The genTop is inside an hotvr!')
                    #         setattr(genParticle, 'is_inside_hotvr', True)
                    #         setattr(genParticle, 'inside_nhotvr', genParticle.inside_nhotvr+1)
                    #         setattr(genParticle, 'rho_over_pt_hotvr', 600./hotvr.pt)

                    #         max_deltaR_hotvr_q = 0.
                    #         for daughter in gentop_daughters:
                    #             max_deltaR_hotvr_q = max(max_deltaR_hotvr_q, deltaR(hotvr, daughter))

                    #         if max_deltaR_hotvr_q < 600./hotvr.pt:
                    #             if self.print_out: print('All the decays are inside the hotvr. The greater dR is {}'.format(max_deltaR_hotvr_q))
                    #             setattr(genParticle, 'all_decays_inside_hotvr', True)
                    #         setattr(genParticle, 'max_deltaR_q_hotvr', max_deltaR_hotvr_q)

                    #         if self.hotvr_top_tagged(hotvr, subjets_in_hotvr): 
                    #             if self.print_out: print('which is top tagged!')
                    #             setattr(genParticle, 'is_inside_hotvr_top_tagged', True)
                    #     else: continue
                    # setattr(genParticle, 'min_deltaR_hotvr', min_deltaR_hotvr_top)
                    # ----

                    gentops.append(genParticle)
            else: continue
        # ----

        self.out.fillBranch("ngenTop", len(gentops))
        for genTopKey in self.genTopKeys:
            self.out.fillBranch("genTop_"+genTopKey, map(lambda gentop: getattr(gentop,genTopKey), gentops))
        for variable in self.storeKinematics:
            self.out.fillBranch("genTop_"+variable, map(lambda gentop: getattr(gentop,variable), gentops))    

        return True

