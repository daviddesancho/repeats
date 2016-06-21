#!/usr/env python

import sys
import os
import copy
import random
import numpy as np
import multiprocessing as mp
#get_ipython().magic(u'pylab inline')

from modeller import *
from modeller.optimizers import molecular_dynamics, conjugate_gradients
from modeller.automodel import autosched

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", color_codes=True, font_scale=1.5)
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})


# # Modelling mutants of a TPR repeat with Modeller
# We adapt the [`mutate_model.py`](http://salilab.org/modeller/wiki/Mutate%20model
# ), from which we get the main code for generating and optimizing the mutants.

def optimize(atmsel, sched):
    """ Conjugate gradient """
    for step in sched:
        step.optimize(atmsel, max_iterations=200, min_atom_shift=0.001)
    #md
    refine(atmsel)
    cg = conjugate_gradients()
    cg.optimize(atmsel, max_iterations=200, min_atom_shift=0.001)


def refine(atmsel):
    """ Molecular dynamics """
    # at T=1000, max_atom_shift for 4fs is cca 0.15 A.
    md = molecular_dynamics(cap_atom_shift=0.39, md_time_step=4.0,
                            md_return='FINAL')
    init_vel = True
    for (its, equil, temps) in ((200, 20, (150.0, 250.0, 400.0, 700.0, 1000.0)),
                                (200, 600,
                                 (1000.0, 800.0, 600.0, 500.0, 400.0, 300.0))):
        for temp in temps:
            md.optimize(atmsel, init_velocities=init_vel, temperature=temp,
                         max_iterations=its, equilibrate=equil)
            init_vel = False

def make_restraints(mdl, aln):
    """Use homologs and dihedral library for dihedral angle restraints """
    rsr = mdl.restraints
    rsr.clear()
    s = selection(mdl)
    for typ in ('stereo', 'phi-psi_binormal'):
        rsr.make(s, restraint_type=typ, aln=aln, spline_on_site=True)
    for typ in ('omega', 'chi1', 'chi2', 'chi3', 'chi4'):
        rsr.make(s, restraint_type=typ+'_dihedral', spline_range=4.0,
                spline_dx=0.3, spline_min_points = 5, aln=aln,
                spline_on_site=True)

def mutate_model(env, modelname, mdl1, rp, rt):
    """
    Mutates Modeller protein model
    
    Parameters
    ----------
    env : 
        Modeller environment

    mdl1 : Modeller model
        The model that will be mutated.
        
    rp : str
        Residue position.
    
    rt : str
    
    Returns
    -------
    mdl1 : Modeller model
        The model that will be mutated.
        
    """
    chain = "A"
    # Read the original PDB file and copy its sequence to the alignment array:
    ali = alignment(env)
    ali.append_model(mdl1, atom_files=modelname, align_codes=modelname)

    #set up the mutate residue selection segment
    s = selection(mdl1.chains[chain].residues[rp])

    #perform the mutate residue operation
    s.mutate(residue_type=rt)
    #get two copies of the sequence.  A modeller trick to get things set up
    ali.append_model(mdl1, align_codes=modelname)

    # Generate molecular topology for mutant
    mdl1.clear_topology()
    mdl1.generate_topology(ali[-1])

    # Transfer all the coordinates you can from the template native structure
    # to the mutant (this works even if the order of atoms in the native PDB
    # file is not standard):
    #here we are generating the model by reading the template coordinates
    mdl1.transfer_xyz(ali)

    # Build the remaining unknown coordinates
    mdl1.build(initialize_xyz=False, build_method='INTERNAL_COORDINATES')

    #yes model_copy is the same file as model.  It's a modeller trick.
    mdl2 = model(env, file=modelname)

    #required to do a transfer_res_numb
    #ali.append_model(mdl2, atom_files=modelname, align_codes=modelname)
    #transfers from "model 2" to "model 1"
    mdl1.res_num_from(mdl2,ali)

    #It is usually necessary to write the mutated sequence out and read it in
    #before proceeding, because not all sequence related information about MODEL
    #is changed by this command (e.g., internal coordinates, charges, and atom
    #types and radii are not updated).
    mdl1.write(file=modelname+rt+rp+'.tmp')
    mdl1.read(file=modelname+rt+rp+'.tmp')

    #set up restraints before computing energy
    #we do this a second time because the model has been written out and read in,
    #clearing the previously set restraints
    make_restraints(mdl1, ali)

    #a non-bonded pair has to have at least as many selected atoms
    mdl1.env.edat.nonbonded_sel_atoms=1
    sched = autosched.loop.make_for_model(mdl1)

    #only optimize the selected residue (in first pass, just atoms in selected
    #residue, in second pass, include nonbonded neighboring atoms)
    #set up the mutate residue selection segment
    s = selection(mdl1.chains[chain].residues[rp])

    mdl1.restraints.unpick_all()
    mdl1.restraints.pick(s)

    s.energy()
    s.randomize_xyz(deviation=4.0)
    mdl1.env.edat.nonbonded_sel_atoms=2
    optimize(s, sched)

    # feels environment (energy computed on pairs that have at least one member
    # in the selected)
    mdl1.env.edat.nonbonded_sel_atoms=1
    optimize(s, sched)
    energy = s.energy()

    #give a proper name
    #mdl1.write(file=modelname+rt+rp+'.pdb')

    #delete the temporary file
    try:
        os.remove(modelname+rt+rp+'.tmp')
    except OSError:
        pass
    return mdl1

def model_compete_worker(xxx):
    """ 
    Worker function for producing models in parallel

    Parameters
    ----------
    xxx : str
        Input parameters for parallel run. Includes:
        x[0] : int
            The run index.

    Returns
    -------

    """
    log.none()
    # Set a different value for rand_seed to get a different final model
    env = environ(rand_seed=-np.random.randint(0,100000))
    env.io.hetatm = True
    #soft sphere potential
    env.edat.dynamic_sphere=False
    #lennard-jones potential (more accurate)
    env.edat.dynamic_lennard=True
    env.edat.contact_shell = 4.0
    env.edat.update_dynamic = 0.39
    
    # Read customized topology file with phosphoserines (or standard one)
    env.libs.topology.read(file='$(LIB)/top_heav.lib')
    
    # Read customized CHARMM parameter library with phosphoserines (or standard one)
    env.libs.parameters.read(file='$(LIB)/par.lib')
    
    modelname = "3atb"
    repeatB0 = [25, 26, 28, 29, 32, 33, 35]
    repeatA0 = [42, 44, 45, 48, 49, 52, 53, 55, 56]
    respos0 = repeatB0 + repeatA0
    repeatB = [x + 34 for x in repeatB0]
    repeatA = [x + 34 for x in repeatA0]
    respos = repeatB + repeatA
    restyp = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", \
            "ILE", "LEU", "LYS",    "MET", "PHE", "PRO", "SER", "THR", "TRP", \
            "TYR", "VAL"]
    mdl_tpr = model(env, file=modelname)
    s = selection(mdl_tpr) 
    ener_tpr = s.assess_dope()
    mdl_tpr.write(file='initial%s.pdb'%xxx) # save initial state
    tpr_residues = {}
    for r in repeatA:
        tpr_residues[r] = mdl_tpr.residues["%s:A"%r].pdb_name
    
    # calculate energy interface BA'
    s = selection()
    [s.add(mdl_tpr.residues["%s:A"%x]) for x in respos]
    ener_ba_tpr = s.assess_dope()
    
    s = selection()
    [s.add(mdl_tpr.residues["%s:A"%x]) for x in respos0]

    #print s
    ener_ab_tpr = s.assess_dope()
    beta = 1./50 # inverse temperature
    len_mc = 10000 # length of MC run
    fwrite = 10
    w = 0.4
    dener_tpr = ener_ba_tpr - ener_ab_tpr
    ener_init = w*ener_tpr + (1.-w)*dener_tpr
    
    ener_cum = [] 
    nb_stdout = sys.stdout # redirect outputnb_stdout = sys.stdout # redirect output
    sys.stdout = open('mc%s.out'%xxx, 'w') # redirect output
    n=0
    naccept = 0
    ener_prev = ener_init
    contribs = [ener_init, ener_tpr, ener_ab_tpr, ener_ba_tpr]
    energy = [contribs]
    print "initial", ener_prev
    mdl = model(env, file='initial%s.pdb'%xxx)
    current = 'initial%s.pdb'%xxx
    while True:
        mdl = model(env, file=current)
        mdl.write(file='old%s.pdb'%xxx)
        rp = random.choice(respos) # randomly select position
        rt = random.choice(restyp) # randomly select residue
        print "\n Run: %g; Iteration : %i, %i, %s"%(xxx, n, rp, rt)
        
        # build model for actual mutation
        try:
            print " mutation "
            mdl = mutate_model(env, modelname, mdl, "%s"%rp, rt)
            mdl.write(file='mutant%s.pdb'%xxx)
            s = selection(mdl)
            ener_mut = s.assess_dope()
        
            # calculate energy interface BA'
            s = selection()
            [s.add(mdl.residues["%s:A"%x]) for x in respos]
            ener_ba = s.assess_dope()
            print ener_ba
    
            # build model for competing mutation    
            mutations = []
            for r in repeatA:
                #print r, mdl.residues["%s:A"%r].pdb_name, tpr_residues[r]
                if mdl.residues["%s:A"%r].pdb_name != tpr_residues[r]:
                    mutations.append((r, mdl.residues["%s:A"%r].pdb_name))
                    #print " divergent residue", r, mdl.residues["%s:A"%r].pdb_name, tpr_residues[r]
            mdl.read(file="initial%s.pdb"%xxx)
            print " competitor "
            for r, res in mutations:
                print r, res
                mdl = mutate_model(env, modelname, mdl, "%s"%(r-34), rt)
#            mdl.write(file="competitor.pdb")

            # calculate energy interface AB
            s = selection()
            [s.add(mdl.residues["%s:A"%x]) for x in respos0]
            ener_ab = s.assess_dope()
            dener = ener_ba - ener_ab
            ener = w*ener_mut + (1.-w)*dener
            if ener < ener_prev:
                print "### ACCEPT ###"
                ener_prev = ener #[ener_mut, ener_ab, ener_ba]
                current = 'mutant%s.pdb'%xxx
                naccept +=1
                contribs = [ener, ener_mut, ener_ab, ener_ba]
            else:
                dener = ener - ener_prev
                print dener
                if np.exp(-beta*dener) > np.random.random():
                    print "*** Boltzmann ACCEPT ***"
                    current = 'mutant%s.pdb'%xxx
                    naccept +=1
                    ener_prev = ener 
                    contribs = [ener, ener_mut, ener_ab, ener_ba]
                else:
                    current = 'old%s.pdb'%xxx
        except OverflowError:
            current = 'old%s.pdb'%xxx
    
        print " Current energy %g\n"%ener_prev
        if n%fwrite == 0:
            energy.append(contribs)
    
        n +=1
        print current
        if n >= len_mc:
            mdl = model(env, file=current)
            mdl.write(file=modelname + "_run%s"%xxx + ".pdb")
            break
    sys.stdout = nb_stdout # redirect output
    fout = open("optim%s.dat"%xxx, "w")
    for i,data in enumerate(energy):
        fout.write("%i %10.4f %10.4f %10.4f %10.4f %10.4f\n"%(i, data[0],data[1], data[2], data[3], data[3]-data[2]))
    fout.close()

    return energy, float(naccept)/len_mc

if __name__ == "__main__":
   
    results = []
    pool = mp.Pool(mp.cpu_count())
#    pool = mp.Pool(4)
    for x in range(4):
        results.append(pool.apply_async(model_compete_worker, [x]))
    pool.close()
    pool.join()
   
    ener_cum = []
    acceptance = []
    for r in results:
        ener_cum.append(r.get()[0])
        acceptance.append(r.get()[1])

    fig, ax = plt.subplots(5,1, sharex=True, figsize=(8,10))
    for energy in ener_cum:
        ax[0].plot([x[0] for x in energy])
        ax[1].plot([x[1] for x in energy])
        ax[2].plot([x[2] for x in energy])
        ax[3].plot([x[3] for x in energy])
        ax[4].plot([(x[3]-x[2]) for x in energy])
    
    ax[4].set_xlabel('# MC steps', fontsize=16)
    ax[0].set_ylabel('Weighted E', fontsize=16)
    ax[1].set_ylabel('DOPE', fontsize=16)
    ax[2].set_ylabel('E(BA)', fontsize=16)
    ax[3].set_ylabel('E(AB\')', fontsize=16)
    ax[4].set_ylabel('E(BA)-E(AB\')', fontsize=16)
    
    #fig, ax = plt.subplots()
    #for energy in ener_cum:
    #    ax.plot([x[1] for x in energy],[x[3]-x[2] for x in energy], 'o-')
    #ax.set_xlabel('DOPE Energy', fontsize=18)
    #ax.set_ylabel('E(BA\')-E(AB)', fontsize=18)
    #plt.savefig("optim.png")
    #plt.tight_layout()
