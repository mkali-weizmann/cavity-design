from sympy.parsing.mathematica import parse_mathematica



mathematica_expr = "(1/(kx^2 + ky^2 + kz^2))(-kx x0 + kx xc - ky y0 + ky yc - kz z0 - 1/2 Sqrt[4 (kx (x0 - xc) + ky (y0 - yc) + kz (z0 - zc))^2 - 4 (kx^2 + ky^2 + kz^2) (-R^2 + (x0 - xc)^2 + (y0 - yc)^2 + (z0 - zc)^2)] + kz zc)"
sympy_expr = str(parse_mathematica(mathematica_expr))


sympy_expr = sympy_expr.replace("kx", "ray.k_vector[..., 0]").replace("ky", "ray.k_vector[..., 1]").replace("kz", "ray.k_vector[..., 2]").replace("xc", "self.origin[0]").replace("yc", "self.origin[1]").replace("zc", "self.origin[2]").replace("x0", "ray.origin[..., 0]").replace("y0", "ray.origin[..., 1]").replace("z0", "ray.origin[..., 2]").replace("R", "self.radius").replace("sqrt", "np.sqrt")

print(str(sympy_expr))
