#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from meld.remd import ladder, adaptor, master_runner
from meld import system
from meld import comm, vault
from meld import parse


N_REPLICAS = 20
N_STEPS = 10000
BLOCK_SIZE = 100

def gen_state(s, index):
    pos = s._coordinates
    pos = pos - np.mean(pos, axis=0)
    vel = np.zeros_like(pos)
    alpha = index / (N_REPLICAS - 1.0)
    energy = 0
    return system.SystemState(pos, vel, alpha, energy)


def get_dist_restraints(filename, s, scaler):
    dists = []
    rest_group = []
    lines = open(filename).read().splitlines()
    lines = [line.strip() for line in lines]
    for line in lines:
        if not line:
            dists.append(s.restraints.create_restraint_group(rest_group, 1))
            rest_group = []
        else:
            cols = line.split()
            i = int(cols[0])
            name_i = cols[1]
            j = int(cols[2])
            name_j = cols[3]
            dist = float(cols[4]) / 10.

            rest = s.restraints.create_restraint('distance', scaler,
                                                 r1=0.0, r2=0.0, r3=dist, r4=dist+0.2, k=250,
                                                 atom_1_res_index=i, atom_2_res_index=j,
                                                 atom_1_name=name_i, atom_2_name=name_j)
            rest_group.append(rest)
    return dists


def setup_system():
    # load the sequence
    sequence = parse.get_sequence_from_AA1(filename='sequence.dat')
    n_res = len(sequence.split())

    # build the system
    p = system.ProteinMoleculeFromSequence(sequence)
    b = system.SystemBuilder()
    s = b.build_system_from_molecules([p])
    s.temperature_scaler = system.GeometricTemperatureScaler(0, 0.4, 300., 550.)

    #
    # Secondary Structure
    #
    ss_scaler = s.restraints.create_scaler('constant')
    ss_rests = parse.get_secondary_structure_restraints(filename='ss.dat', system=s, scaler=ss_scaler,
            torsion_force_constant=2.5, distance_force_constant=2.5)
    n_ss_keep = int(len(ss_rests) * 0.85)
    s.restraints.add_selectively_active_collection(ss_rests, n_ss_keep)

    #
    # Confinement Restraints
    #
    conf_scaler = s.restraints.create_scaler('constant')
    confinement_rests = []
    confinement_dist = (16.9*np.log(s.residue_numbers[-1])-15.8)/28
    for index in range(n_res):
        rest = s.restraints.create_restraint('confine', conf_scaler, res_index=index+1, atom_name='CA', radius=confinement_dist, force_const=250.0)
        confinement_rests.append(rest)
    s.restraints.add_as_always_active_list(confinement_rests)

    #
    # Distance Restraints
    #
    # High reliability
    #
    dist_scaler = s.restraints.create_scaler('nonlinear', alpha_min=0.4, alpha_max=1.0, factor=4.0)
    selected_dist = get_dist_restraints('contacts.dat', s, dist_scaler)
    n_high_keep = int(1.00 * len(selected_dist))
    s.restraints.add_selectively_active_collection(selected_dist, n_high_keep)

    #
    # create the options
    options = system.RunOptions()
    options.implicit_solvent_model = 'gbNeck2'
    options.use_big_timestep = True
    options.cutoff = 1.8

    options.use_amap = True
    options.amap_beta_bias = 1.0
    options.timesteps = 14286
    options.minimize_steps = 20000

    # create a store
    store = vault.DataStore(s.n_atoms, N_REPLICAS, s.get_pdb_writer(), block_size=BLOCK_SIZE)
    store.initialize(mode='w')
    store.save_system(s)
    store.save_run_options(options)

    # create and store the remd_runner
    l = ladder.NearestNeighborLadder(n_trials=48 * 48)
    policy = adaptor.AdaptationPolicy(2.0, 50, 50)
    a = adaptor.EqualAcceptanceAdaptor(n_replicas=N_REPLICAS, adaptation_policy=policy)

    remd_runner = master_runner.MasterReplicaExchangeRunner(N_REPLICAS, max_steps=N_STEPS, ladder=l, adaptor=a, ramp_steps=50)
    store.save_remd_runner(remd_runner)

    # create and store the communicator
    c = comm.MPICommunicator(s.n_atoms, N_REPLICAS)
    store.save_communicator(c)

    # create and save the initial states
    states = [gen_state(s, i) for i in range(N_REPLICAS)]
    store.save_states(states, 0)

    # save data_store
    store.save_data_store()

    return s.n_atoms


setup_system()
