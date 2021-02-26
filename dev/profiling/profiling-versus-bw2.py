from functools import partial
import psutil
import time


def run_curried_lca(func):
    lca = func()
    lca.lci()
    lca.lcia()
    print(lca.score)


def profile_func(func):
    # Logging code adapted from https://github.com/rasbt/pyprind/blob/master/pyprind/prog_class.py
    start = time.time()
    process = psutil.Process()

    func()

    cpu_total = process.cpu_percent()
    mem_total = process.memory_percent()
    print('Time: {}\nCPU %: {:.2f}\nMemory %: {:.2f}'.format(time.time() - start, cpu_total, mem_total))






# Code for bw2

import bw2data as bd, bw2calc as bc
bd.projects.set_current("ecoinvent 3.7.1 bw2")
bd.databases
a = bd.get_activity(('ecoinvent 3.7.1', 'f57568b2e553864152a6ac920595216f'))
a
ipcc = ('IPCC 2013', 'climate change', 'GWP 100a')

curry = partial(bc.LCA, demand={a: 1}, method=ipcc)
profile_func(partial(run_curried_lca, func=curry))


# Code for bw2.5

import bw2data as bd, bw2calc as bc
bd.projects.set_current("ecoinvent 3.7.1")
bd.databases
a = bd.get_activity(('ecoinvent 3.7.1', 'f57568b2e553864152a6ac920595216f'))
a
ipcc = ('IPCC 2013', 'climate change', 'GWP 100a')

fu, data_objs, _ = bd.prepare_lca_inputs({a: 1}, method=ipcc)
curry = partial(bc.LCA, demand=fu, data_objs=data_objs)
profile_func(partial(run_curried_lca, func=curry))


