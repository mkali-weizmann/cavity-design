from sympy.parsing.mathematica import parse_mathematica



mathematica_expr = "(1/(2 (kx^2 + ky^2 + kz^2)))(-2 kx x0 + 2 kx xc - 2 ky y0 + 2 ky yc - 2 kz z0 + 2 kz zc - Sqrt[(2 kx x0 - 2 kx xc + 2 ky y0 - 2 ky yc + 2 kz z0 - 2 kz zc)^2 - 4 (kx^2 + ky^2 + kz^2) (-R^2 + x0^2 - 2 x0 xc + xc^2 + y0^2 - 2 y0 yc + yc^2 + z0^2 - 2 z0 zc + zc^2)])"

sympy_expr = parse_mathematica(mathematica_expr)

print(str(sympy_expr))
