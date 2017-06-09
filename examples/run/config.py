from astropy import units
aout = [0.5, 1.0]
stages = autostages(aout, astart=0.1, N=5)
unitary = True
nc = 64
#cosmology = Planck15.clone(Tcmb0=0)
cosmology = Planck15.clone(Neff=0)
