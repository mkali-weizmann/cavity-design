from cavity import *
from matplotlib import ticker

lambda_0_laser = 1064e-9

# Original lens
# params = np.array([[-5.0000000000e-03+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j, -0.0000000000e+00-1.j,  5.0000632553e-03+0.j,   np.nan+0.j,  0.0000000000e+00+0.j,   np.nan+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  7.5000000000e-08+0.j,  1.0000000000e-06+0.j,  1.3100000000e+00+0.j,   np.nan+0.j,  1.7000000000e-01+0.j,   np.nan+0.j,  9.9988900000e-01+0.j,  1.0000000000e-04+0.j,   np.nan+0.j,  0.0000000000e+00+0.j],
#           [ 6.3393015143e-03+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  2.3479617901e-02+0.j,  6.2124542986e-03+0.j,  1.0000000000e+00+0.j,  2.6786030286e-03+0.j,  1.7600000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,          np.nan+0.j,   np.nan+0.j,   np.nan+0.j,   np.nan+0.j,   np.nan+0.j,   np.nan+0.j,          np.nan+0.j,   np.nan+0.j,   np.nan+0.j,  1.0000000000e+00+0.j],
#           [ 3.0767860303e-01+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.5038941544e-01+0.j,   np.nan+0.j,  0.0000000000e+00+0.j,   np.nan+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  7.5000000000e-08+0.j,  1.0000000000e-06+0.j,  1.3100000000e+00+0.j,   np.nan+0.j,  1.7000000000e-01+0.j,   np.nan+0.j,  9.9988900000e-01+0.j,  1.0000000000e-04+0.j,   np.nan+0.j,  0.0000000000e+00+0.j]])

params = np.array([[-5.0000000000e-03+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j, -0.0000000000e+00-1.j,  5.0000262120e-03+0.j,   np.nan+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  1.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  7.5000000000e-08+0.j,  1.0000000000e-06+0.j,  1.3100000000e+00+0.j,   np.nan+0.j,  1.7000000000e-01+0.j,   np.nan+0.j,  9.9988900000e-01+0.j,  1.0000000000e-04+0.j,   np.nan+0.j,  0.0000000000e+00+0.j],
          [ 6.7870819720e-03+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.0596029000e-02+0.j,  5.0000000000e-03+0.j,  1.0000000000e+00+0.j,  3.5741639441e-03+0.j,  1.7600000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,  1.7600000000e+00+0.j,  5.5000000000e-06+0.j,  1.0000000000e-06+0.j,  4.6060000000e+01+0.j,  1.1700000000e-05+0.j,  3.0000000000e-01+0.j,  1.0000000000e-02+0.j,  1.0000000000e-04+0.j,  9.9989900000e-01+0.j,   np.nan+0.j,  1.0000000000e+00+0.j],
          [ 3.0857416394e-01+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  2.8020442921e-01+0.j,   np.nan+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  1.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  7.5000000000e-08+0.j,  1.0000000000e-06+0.j,  1.3100000000e+00+0.j,   np.nan+0.j,  1.7000000000e-01+0.j,   np.nan+0.j,  9.9988900000e-01+0.j,  1.0000000000e-04+0.j,   np.nan+0.j,  0.0000000000e+00+0.j]])

cavity = Cavity.from_params(params=params,
                            standing_wave=True,
                            lambda_0_laser=lambda_0_laser,
                            names=['left mirror', 'lens-left', 'lens_right',  'right mirror'],
                            set_central_line=True,
                            set_mode_parameters=True,
                            set_initial_surface=False,
                            t_is_trivial=True,
                            p_is_trivial=True,
                            power=2e4)

cavity.arms[0].mode_parameters_on_surface_1.spot_size
cavity.arms[1].mode_parameters_on_surface_0.spot_size

plot_mirror_lens_mirror_cavity_analysis(cavity)

NA = 0.05461415  # 0.138
beginning_ray = Ray(origin=np.array([0, 0, 0]), k_vector=np.array([np.cos(np.arcsin(NA)), NA, 0]))
second_ray = cavity.physical_surfaces[1].reflect_ray(beginning_ray)
third_ray = cavity.physical_surfaces[2].reflect_ray(second_ray)

ax = plt.gca()
beginning_ray.plot(ax=ax, color='r')
second_ray.plot(ax=ax, color='r')
third_ray.plot(ax=ax, color='r')


plt.xlim(-0.001, 0.01)
plt.ylim(-0.001, 0.002)


ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=20))

ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=20))
plt.title("Mines")
plt.tight_layout()
plt.show()
