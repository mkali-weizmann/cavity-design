from cavity import *
params = params_dict['Sapphire, NA=0.2, L1=0.3, w=4.33mm - High NA axis']
cavity = Cavity.from_params(params=params, standing_wave=True,
                            lambda_laser=lambda_laser, names=names, t_is_trivial=True, p_is_trivial=True)
cavity.print_table()