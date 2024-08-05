# Copyright 2022-2024 The Ramble Authors
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

import os
import subprocess
from ramble.appkit import *
from ramble.expander import Expander

def find_executable(executable):
    try:
        path = subprocess.check_output(['which', executable], stderr=subprocess.STDOUT).decode().strip()
        return path
    except subprocess.CalledProcessError:
        return None

class Beatnik(ExecutableApplication):
    """Define BEATNIK application"""

    name = "BEATNIK"

    maintainers("jasonstewart", "patrickbridges")

    tags("proxy-app", "mini-app")

    define_compiler("gcc11", pkg_spec="gcc@11.4.0", package_manager="spack*")
    define_compiler("gcc8", pkg_spec="gcc@8.3.1", package_manager="spack*")
    define_compiler("cce16", pkg_spec="cce@16.0.1", package_manager="spack*")

    software_spec(
        "beatnik",
        pkg_spec="beatnik@develop +cuda cuda_arch=86",
        compiler="gcc11",
        package_manager="spack*",
        # external_env="~/spack_envs/beatnik_lassen"
    )
    
    software_spec(
        "beatnik",
        pkg_spec="beatnik@develop +cuda cuda_arch=86",
        compiler="gcc8",
        package_manager="spack*",
    )

    required_package("beatnik", package_manager="spack*")

    executable("execute", "beatnik1.0 {flags}", use_mpi=True)

    workload("standard", executables=["execute"])

    mpi_executables = {
        'mpicc': find_executable('mpicc'),
        'mpicxx': find_executable('mpicxx'),
        # 'mpifc': find_executable('mpifc'),
        'mpirun': find_executable('mpirun')
    }

    if None in mpi_executables.values():
        raise EnvironmentError("One or more MPI executables could not be found.")

    def _make_experiments(self, workspace, app_inst=None):
        """
        BEATNIK requires the number of ranks to be a square number.

        Here we compute the closest integer equal to or larger than the target
        number of ranks.

        We also need to recompute the number of nodes, or the value of
        processes per node here too.
        """
        num_ranks = int(self.expander.expand_var_name("n_ranks"))

        square_root = int(num_ranks ** (1.0 / 2.0))

        self.variables["n_ranks"] = square_root**2

        super()._make_experiments(workspace)
