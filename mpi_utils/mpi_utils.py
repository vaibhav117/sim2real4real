from mpi4py import MPI
import numpy as np
import torch

# sync_networks across the different cores
def sync_networks(network):
    """
    netowrk is the network you want to sync

    """
    comm = MPI.COMM_WORLD
    flat_params = _get_flat_params_or_grads(network, mode='params')
    comm.Bcast(flat_params, root=0)
    # set the flat params back to the network
    _set_flat_params_or_grads(network, flat_params, mode='params')

def sync_grads(network):
    flat_grads = _get_flat_params_or_grads(network, mode='grads')
    comm = MPI.COMM_WORLD
    global_grads = np.zeros_like(flat_grads)
    comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)
    _set_flat_params_or_grads(network, global_grads, mode='grads')

def check_for_no_grad(param, attr):
    if getattr(param, attr) == None:
        return

    return getattr(param, attr).cpu().numpy().flatten()

# get the flat grads or params
def _get_flat_params_or_grads(network, mode='params'):
    """
    include two kinds: grads and params

    """
    attr = 'data' if mode == 'params' else 'grad'
    final_list = []
    for param in network.parameters():
        if getattr(param, attr) != None:
            final_list.append(getattr(param, attr).cpu().numpy().flatten())
    return np.concatenate(final_list)

def _set_flat_params_or_grads(network, flat_params, mode='params'):
    """
    include two kinds: grads and params

    """
    attr = 'data' if mode == 'params' else 'grad'
    # the pointer
    pointer = 0
    for param in network.parameters():
        if getattr(param, attr) != None:
            getattr(param, attr).copy_(torch.tensor(flat_params[pointer:pointer + param.data.numel()]).view_as(param.data))
            pointer += param.data.numel()
