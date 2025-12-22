from dataclasses import dataclass
import os
import sys
import subprocess as sp
import time
from typing import Any

from scipy.sparse import coo_matrix
import urllib
import numpy as np
import msgpack
import msgpack_numpy
from cuopt_sh_client import CuOptServiceSelfHostClient

msgpack_numpy.patch()

from compass.models.MetabolicModel import MetabolicModel
from compass.opt.base import LinearProgramDelta, Optimizer, Solution


# Used to pass larger problems via file system instead of HTTP
CUOPT_DATA_PATH = "_cuopt_server_data"
CUOPT_RESULTS_PATH = "_cuopt_server_results"
CUOPT_LOG_PATH = "_cuopt_server_log"

# Keep same across client init and repoll calls
_REPOLL_INTERVAL = 1
_REPOLL_TIMEOUT = 6000

_CUOPT_CLIENT = None


def get_cuopt_client(ip: str, port: str) -> CuOptServiceSelfHostClient:
    """
    Lazy singleton to get or create the CuOpt client.
    Ensures one client per process.
    """
    global _CUOPT_CLIENT
    if _CUOPT_CLIENT is None:
        _CUOPT_CLIENT = CuOptServiceSelfHostClient(
            ip=ip,
            port=port,
            polling_interval=_REPOLL_INTERVAL,
            polling_timeout=_REPOLL_TIMEOUT,
        )
    return _CUOPT_CLIENT


def check_cuopt_server_status(ip: str, port: str) -> bool:
    """
    Docstring for check_cuopt_server_status

    :param ip: IP of cuopt server
    :type ip: str
    :param port: Port of cuopt server
    :type port: str
    :return: Returns whether the cuopt server reported as healthy
    :rtype: bool
    """
    # Note that CuOptServiceSelfHostClient does not seem to expose a health check method
    # So just using urllib here
    health_url = f"http://{ip}:{port}/cuopt/health"
    try:
        with urllib.request.urlopen(health_url, timeout=0.5) as response:
            if response.status == 200:
                return True
            else:
                return False
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return False


@dataclass
class CuoptServerParameters:
    ip: str
    port: str
    data_dir: str
    results_dir: str


class CuoptServerProcess:
    def __init__(self, output_dir: str, gpu_count: int, ip: str, port: str):
        # Spawn the cuopt server, must be done by the main process
        python_exe = sys.executable
        if python_exe is None or len(python_exe) == 0:
            raise Exception("sys.executable is None or empty; not possible to find current python interpreter")

        abs_path = os.path.abspath(output_dir)
        self.data_dir = os.path.join(abs_path, CUOPT_DATA_PATH)
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        self.results_dir = os.path.join(abs_path, CUOPT_RESULTS_PATH)
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)
        # TODO: max_result should be tuned for performance.
        self.max_result = 250
        self.log_file = os.path.join(abs_path, CUOPT_LOG_PATH)

        # TODO: Add logging.

        # We use a custom launcher to:
        # 1. Import logging.handlers to avoid
        # AttributeError: module 'logging' has no attribute 'handlers'. Did you mean: '_handlers'?
        # 2. Run by path to avoid runpy RuntimeWarning:
        # cuopt_server.cuopt_service' found in sys.modules after import of package 'cuopt_server', but prior to execution of 'cuopt_server.cuopt_service'; this may result in unpredictable behaviour
        launcher_code = (
            "import logging.handlers; "
            "import os, runpy, cuopt_server; "
            "pkg_path = os.path.dirname(cuopt_server.__file__); "
            "script_path = os.path.join(pkg_path, 'cuopt_service.py'); "
            "runpy.run_path(script_path, run_name='__main__')"
        )
        
        cmd = [
            python_exe,
            "-c",
            launcher_code,
            "--ip",
            ip,
            "--port",
            port,
            "--gpu-count",
            str(gpu_count),
            "--datadir",
            self.data_dir,
            "--resultdir",
            self.results_dir,
            "--maxresult",
            str(self.max_result),
            "--log-level",
            "warning",
            "--log-file",
            self.log_file,
        ]

        stdout_log = os.path.join(abs_path, "cuopt_server.stdout")
        stderr_log = os.path.join(abs_path, "cuopt_server.stderr")
        stdout_f = open(stdout_log, "w")
        stderr_f = open(stderr_log, "w")

        self.ip = ip
        self.port = port
        self.proc = sp.Popen(
            cmd,
            stdout=stdout_f,
            stderr=stderr_f,
        )

        # Arbitrarily chosen timeout of 10 seconds
        for _ in range(10):
            status = check_cuopt_server_status(ip, port)
            if status:
                return
            else:
                time.sleep(1)

        raise Exception("Could not verify status of cuopt server")

    def get_params(self) -> CuoptServerParameters:
        return CuoptServerParameters(ip=self.ip, port=self.port, data_dir=self.data_dir, results_dir=self.results_dir)

    def shutdown(self):
        self.proc.terminate()
        try:
            self.proc.wait(5)
            return
        except sp.TimeoutExpired:
            pass
        self.proc.kill()
        try:
            self.proc.wait(5)
        except sp.TimeoutExpired:
            print("Cuopt server failed to exit after terminate and kill", file=sys.stderr)


class CuoptServerOptimizer(Optimizer):
    """
    Implementation of Optimizer class that sends requests to a cuopt server
    """

    def __init__(self, model: MetabolicModel, params: CuoptServerParameters):
        super().__init__(model)

        self.data_dir = params.data_dir
        self.results_dir = params.results_dir
        self.ip = params.ip
        self.port = params.port

    def solve(self, delta: LinearProgramDelta) -> Solution:
        """
        Applies the delta, solves the model, and returns the solution.
        """

        rxn_ub = []
        rxn_lb = []
        rxn_map = {}
        for rxn_id, reaction in self.model.reactions.items():
            assert rxn_id not in rxn_map
            rxn_map[rxn_id] = len(rxn_map)
            rxn_lb.append(reaction.lower_bound)
            rxn_ub.append(reaction.upper_bound)
        assert len(rxn_map) == len(rxn_lb)

        # Construct as COO, so we can edit rows
        rows = []
        cols = []
        vals = []
        metab_map = {}
        for metab_id, stoichiometry in self.model.SMAT.items():
            # If there is no reaction associated with the given metabolite, then skip
            if len(stoichiometry) == 0:
                continue

            # Numbers metabolites in order of iteration
            ind = len(metab_map)
            for rxn_id, coeff in stoichiometry:
                rows.append(ind)
                cols.append(rxn_map[rxn_id])
                vals.append(coeff)

            assert metab_id not in metab_map
            metab_map[metab_id] = ind

        # metabolites must have total 0 flux
        metab_lb = [0.0 for _ in metab_map]
        metab_ub = [0.0 for _ in metab_map]

        # Secretion is consumption from the metabolite pool (-1), uptake is production (+1)
        added_reactions = [(delta.added_secretion.items(), -1.0), (delta.added_uptake.items(), 1.0)]

        for items, coeff in added_reactions:
            for met_id, rxn_id in items:
                rxn_index = len(rxn_map)
                assert rxn_id not in rxn_map
                rxn_map[rxn_id] = rxn_index
                metab_index = metab_map[met_id]

                rxn_ub.append(self.model.maximum_flux)
                rxn_lb.append(0.0)
                rows.append(metab_index)
                cols.append(rxn_index)
                vals.append(coeff)

        for rxn_id in delta.blocked_reactions:
            rxn_index = rxn_map[rxn_id]
            rxn_ub[rxn_index] = rxn_lb[rxn_index]

        for rxn_id, limit in delta.high_flux.items():
            rxn_index = rxn_map[rxn_id]
            rxn_lb[rxn_index] = limit

        objective_coeffs = [0.0 for _ in rxn_map]
        for rxn_id, coeff in delta.objective.items():
            objective_coeffs[rxn_map[rxn_id]] = coeff

        if delta.sense == "max":
            maximize = True
        else:
            maximize = False

        # Using scipy here, as the C code is likely faster than python for this step
        coo = coo_matrix((np.array(vals), (np.array(rows), np.array(cols))), shape=(len(metab_map), len(rxn_map)))
        csr = coo.tocsr()

        problem = {
            "variable_bounds": {
                "lower_bounds": rxn_lb,
                "upper_bounds": rxn_ub,
            },
            "constraint_bounds": {
                "lower_bounds": metab_lb,
                "upper_bounds": metab_ub,
            },
            "csr_constraint_matrix": {
                "indices": csr.indices,
                "offsets": csr.indptr,
                "values": csr.data,
            },
            "maximize": maximize,
            "objective_data": {
                "coefficients": objective_coeffs,
            },
            "solver_config": {
                "cudss_deterministic": True,
            },
        }

        # assert delta.name is not None
        # filename = f"{delta.name}.msgpack"
        # filepath = os.path.join(self.data_dir, filename)
        # with open(filepath, "wb") as f:
        #    msgpack.pack()

        client = get_cuopt_client(self.ip, self.port)
        solution = client.get_LP_solve(
            problem,
            response_type="dict",
        )

        poll_count = 0
        while "response" not in solution:
            if "reqId" not in solution:
                # This is fatal because we cannot repoll without a reqId
                print(solution)
                raise Exception(f"reqId missing from solution: keys are {solution.keys()}")
            poll_count += 1
            if poll_count > _REPOLL_TIMEOUT:
                return Solution(success=False, status="Repolling timeout", obj_value=None)
            time.sleep(_REPOLL_INTERVAL)
            solution = client.repoll(solution["reqId"], response_type="dict")

        resp = solution["response"]
        if resp["status"] == "Optimal":
            success = True
            obj_value = resp["solution"]["primal_objective"]
        else:
            success = False
            obj_value = None

        return Solution(success=success, status=resp["status"], obj_value=obj_value)
