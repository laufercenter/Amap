#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from meld.remd import ladder, adaptor, master_runner
from meld import system
from meld import comm, vault
from meld import parse


N_REPLICAS = 10
N_STEPS = 20000
BLOCK_SIZE = 100


def gen_state(s, index):
    pos = s._coordinates
    vel = np.zeros_like(pos)
    alpha = index / (N_REPLICAS - 1.0)
    energy = 0
    return system.SystemState(pos, vel, alpha, energy)


def setup_system():
    # create the system
    sequence = parse.get_sequence_from_AA1(filename='sequence.dat')
    p = system.ProteinMoleculeFromSequence(sequence)
    #p = system.ProteinMoleculeFromPdbFile('start.pdb')
    b = system.SystemBuilder()
    s = b.build_system_from_molecules([p])
    s.temperature_scaler = system.GeometricTemperatureScaler(0, 1.0, 300., 425.)


    # create the options
    options = system.RunOptions()
    options.implicit_solvent_model = 'gbNeck2'
    options.use_big_timestep = True
    options.cutoff = 1.8
    options.use_amap = True
    options.amap_beta_bias = 3.4
    options.timesteps = 50000
    options.minimize_steps = 20000

    options.sc_alpha_min = 0.15
    options.sc_alpha_max_coulomb = 0.45
    options.sc_alpha_max_lennard_jones = 0.9
    options.sc_alpha_max_lj = 0.
    options.softcore = False

    # create a store
    store = vault.DataStore(s.n_atoms, N_REPLICAS, s.get_pdb_writer(), block_size=BLOCK_SIZE)
    store.initialize(mode='w')
    store.save_system(s)
    store.save_run_options(options)

    # create and store the remd_runner
    l = ladder.NearestNeighborLadder(n_trials=128)
    policy = adaptor.AdaptationPolicy(2.0, 20, 20)
    a = adaptor.EqualAcceptanceAdaptor(n_replicas=N_REPLICAS, adaptation_policy=policy)
    remd_runner = master_runner.MasterReplicaExchangeRunner(N_REPLICAS, max_steps=N_STEPS, ladder=l, adaptor=a)
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
