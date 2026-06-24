# %%
from cavity_design import *
eksma_lens_params = generate_aspheric_lens_params(back_focal_length=17.001e-3, T_c=4.35e-3, forward_normal=LEFT, flat_faces_center=ORIGIN+15e-3*LEFT, n=PHYSICAL_SIZES_DICT['material_properties_fused_silica'].refractive_index, diameter=INCH / 2, polynomial_degree=10, n_outside=1, name="Eksma 20mm")
EKSMA_LENS_20mm_ASPHERIC = OpticalSystem.from_params(params=eksma_lens_params)
focal_length_of_lens(R_1=np.inf, R_2=-EKSMA_LENS_20mm_ASPHERIC.surfaces[1].radius, n=1.45, T_c=4.35e-3)


