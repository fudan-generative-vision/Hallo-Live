#!/usr/bin/env bash

# ===============================================
# Distributed Environment Verification
# ===============================================
print_env_info() {
    echo "=================================================="
    echo " Environment Information"
    echo "=================================================="
    echo "Project Dir:     $(pwd)"
    echo "Python Version:  $(which python)"
    echo "--------------------------------------------------"
    echo "Distributed Config:"
    echo "  MASTER_ADDR:       ${MASTER_ADDR}"
    echo "  MASTER_PORT:       ${MASTER_PORT}"
    echo "  PET_NNODES:        ${PET_NNODES}"
    echo "  PET_NODE_RANK:     ${PET_NODE_RANK}"
    echo "  PET_NPROC_PER_NODE: ${PET_NPROC_PER_NODE}"
    echo "=================================================="
}
