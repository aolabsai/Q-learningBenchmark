


import ao_arch as ar



description = "Basic Clam"
arch_i = [1, 1, 1, 1, 1, 1]     # 3 neurons, 1 in each of 3 channels, corresponding to Food, Chemical-A, Chemical-B (present=1/not=0)
arch_z = [2]           # corresponding to Open=1/Close=0
arch_c = [0]           # adding 1 control neuron which we'll define with the instinct control function below

connector_function = "full_conn"



# To maintain compatibility with our API, do not change the variable name "Arch" or the constructor class "ar.Arch" in the line below
Arch = ar.Arch(arch_i, arch_z, arch_c, connector_function, description=description)




