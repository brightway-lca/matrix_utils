import bw2data as bd, bw2calc as bc
bd.projects.set_current("ecoinvent 3.7.1")
bd.databases
a = bd.get_activity(('ecoinvent 3.7.1', 'f57568b2e553864152a6ac920595216f'))
ipcc = ('IPCC 2013', 'climate change', 'GWP 100a')

fu, data_objs, _ = bd.prepare_lca_inputs({a: 1}, method=ipcc)
lca = bc.LCA(demand=fu, data_objs=data_objs)
lca.lci()
lca.lcia()


