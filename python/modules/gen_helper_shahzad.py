import numpy as np
from gen_helper import *

def findTopIdx(p, genParticles):
    ''' find first top quark in a particle's mother decay chain '''
    motherIdx = p.genPartIdxMother
    while (motherIdx>=0):
        if abs(genParticles[motherIdx].pdgId)==6:
            return motherIdx
        motherIdx = genParticles[motherIdx].genPartIdxMother
    return -1
    
    # end of findTopIdx() function --------------------
    
def checkMother(p, pdg, genParticles):
    ''' check mother particle above p is abs(pdgId) == abs(pdg) or not '''
    mother_idx = p.genPartIdxMother
    mother_pdg = genParticles[mother_idx].pdgId
    particle_pdg = p.pdgId

    # goes back up decay chain until it's something else
    while mother_pdg == particle_pdg:
        mother_idx = genParticles[mother_idx].genPartIdxMother
        mother_pdg = genParticles[mother_idx].pdgId

    # check if the something else is pdg we want (e.g., 24 for W boson)
    # limitation in dealing with absolute values
    if mother_pdg == pdg:
        return True
    else:
        return False
    
    # end of checkMother() function --------------------
        
def countTops(genParticles):
    
    # method 1
    # find n leptonic and m hadronic tops in the event
    # loops through all particles
    lepTops = 0
    hadTops = 0
    # topDict = {} # dict for debugging purposes
    
    for genParticle in genParticles:
        # if lepton or quark
        if (abs(genParticle.pdgId) in [11,13,15] or genParticle.pdgId in [1,2,3,4,5]) and fromHardProcess(genParticle) and isFirstCopy(genParticle):
            mother_idx = genParticle.genPartIdxMother
            if mother_idx < 0:
                continue

            # select lepton if mother directly == W boson
            mother_pdg = genParticles[mother_idx].pdgId
            if abs(mother_pdg) != 24:
                continue
                
            # find grandmother
            grandmother_idx = genParticles[mother_idx].genPartIdxMother
            grandmother_pdg = genParticles[grandmother_idx].pdgId

            # make sure grandmother isn't just a W boson copy
            while grandmother_pdg == mother_pdg:
                grandmother_idx = genParticles[grandmother_idx].genPartIdxMother
                grandmother_pdg = genParticles[grandmother_idx].pdgId
                
            # make sure grandmother is a top quark
            if abs(grandmother_pdg) == 6:
            
                # debugging purposes
#                topDict[genParticle._index] = {'index': genParticle._index, 'pdg': genParticle.pdgId, 'mother index':mother_idx, 'mother pdg':mother_pdg}
                if abs(genParticle.pdgId) in [11,13,15]:
                    lepTops += 1
                else:
                    hadTops += 1
                
            # could print for all, but printing for > 4 to exemplify the problem in less verbose output
            # debugging purposes
#            if lepTops + hadTops != 4:
#                print("method 1 limitation, (lep tops, had tops)", lepTops, hadTops)
#                for key in topDict:
#                    print("method 1 limitation, gen particle", topDict[key])
            
    return lepTops, hadTops
    
    # end of countTops() function --------------------

def findLastCopy(p, pdg, depth, genParticles):
    ''' recursive function to find the last copy '''
    isLastCopy_flag = False
    
    if isLastCopy(p):
        isLastCopy_flag = True
        return p, depth, isLastCopy_flag
        
    remaining_genParticles = map(lambda part_i: part_i, genParticles)[p._index+1:]
    remaining_pdg_list_np = np.array([part_i.pdgId for part_i in remaining_genParticles])
    remaining_mother_list_np = np.array([part_i.genPartIdxMother for part_i in remaining_genParticles])
    remaining_index_list_np = np.array([part_i._index for part_i in remaining_genParticles])
    
    # 1. check if any daughters
    # 2. if not then last copy
    # 3. if there are, then check pdg id and hard process
    # 4. if 0, last copy
    # 5. if 1, repeat function for that one with that p._index
    # 6. if > 1, repeat function for all of the potential daughters and choose one with maximum depth or if last copy flag
    any_daughters = np.where(remaining_mother_list_np == p._index)[0]
    
    if len(any_daughters) == 0:
        return p, depth, isLastCopy_flag
    
    # same pdg and fromHardProcess requirement
    any_daughters_pdg = remaining_pdg_list_np[any_daughters] == pdg
    any_daughters_list = remaining_index_list_np[any_daughters][any_daughters_pdg]
    any_daughters_list_hardprocess = [ fromHardProcess( genParticles[p_idx] ) for p_idx in any_daughters_list ]
    
    # if no requirements met, it is the last copy
    if sum(any_daughters_list_hardprocess) == 0:
        return p, depth, isLastCopy_flag
    
    # if one meets requirement --> recursively check if this is the last copy
    elif sum(any_daughters_list_hardprocess) == 1:
        any_daughters_idx = int( any_daughters_list[any_daughters_list_hardprocess] )
        return findLastCopy( genParticles[any_daughters_idx], pdg, depth + 1, genParticles)
        
    # if > 1 meets requirement --> recursively check all possibilities and choose deepest one or the 'isLastCopy' one (priority)
    else:
        any_daughters_idx_list = any_daughters_list[any_daughters_list_hardprocess]
        
        daughter_finalCopy_idx = []
        daughter_finalCopy_depth = []
        daughter_finalCopy_isLastCopy = []
        
        # branching to handle multiple hard process copies
        for i, weird_daughter in enumerate(any_daughters_idx_list):
            finalCopy_idx, finalCopy_depth, finalCopy_isLastCopy = findLastCopy(genParticles[weird_daughter], pdg, depth + 1, genParticles)
            daughter_finalCopy_idx.append(finalCopy_idx._index)
            daughter_finalCopy_depth.append(finalCopy_depth)
            daughter_finalCopy_isLastCopy.append(finalCopy_isLastCopy)
            
        # check if any branched daughters are isLastCopy flagged
        daughter_isLastCopy = np.where( np.array(daughter_finalCopy_isLastCopy) == True )
        if len(daughter_isLastCopy[0]) == 1:
            daughter_finalCopy = daughter_finalCopy_idx[int(daughter_isLastCopy[0])]
            daughter_depth = daughter_finalCopy_depth[int(daughter_isLastCopy[0])]
            daughter_isLastCopy_flag = True
            return genParticles[daughter_finalCopy], daughter_depth, daughter_isLastCopy_flag

        # shouldn't really be possible but would choose first one
        elif len(daughter_isLastCopy[0]) > 1:
            print("multiple last copy flagged daughters")
            daughter_finalCopy = daughter_finalCopy_idx[int(daughter_isLastCopy[0][0])]
            daughter_depth = daughter_finalCopy_depth[int(daughter_isLastCopy[0][0])]
            daughter_isLastCopy_flag = True
            return genParticles[daughter_finalCopy], daughter_depth, daughter_isLastCopy_flag
            
        # if none isLastCopy flagged --> move onto 'deepest' one in the chain
        else:
            daughter_max = np.where( np.array(daughter_finalCopy_depth) == np.array(daughter_finalCopy_depth).max() )

            # if one of these, cool
            if len(daughter_max[0]) == 1:
                daughter_finalCopy = daughter_finalCopy_idx[int( daughter_max[0] )]
                daughter_depth = daughter_finalCopy_depth[int( daughter_max[0] )]
                daughter_isLastCopy_flag = True
                return genParticles[daughter_finalCopy], daughter_depth, daughter_isLastCopy_flag

            # if more than one, ffs, choose first one
            else:
                daughter_finalCopy = daughter_finalCopy_idx[int( daughter_max[0][0] )]
                daughter_depth = daughter_finalCopy_depth[int( daughter_max[0][0] )]
                daughter_isLastCopy_flag = True
                return genParticles[daughter_finalCopy], daughter_depth, daughter_isLastCopy_flag
                
            # can't really have len(0) here ...
            
    # end of findLastCopy() function ---------------
    

    
    
                
            
                
            
                
            
                
        
            

    
    
