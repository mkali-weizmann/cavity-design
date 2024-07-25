from cavity import *
import ipywidgets as widgets


def f(ray: str, surface: str):
    surfaces = {'convex_right': CurvedSurface(radius=2, outwards_normal=np.array([1, 0, 0]), center=np.array([1, 0, 0]),
                                              curvature_sign=CurvatureSigns.concave),
                'concave_right': CurvedSurface(radius=2, outwards_normal=np.array([-1, 0, 0]),
                                               center=np.array([1, 0, 0]), curvature_sign=CurvatureSigns.convex),
                'convex_left': CurvedSurface(radius=2, outwards_normal=np.array([-1, 0, 0]),
                                             center=np.array([-1, 0, 0]), curvature_sign=CurvatureSigns.concave),
                'concave_left': CurvedSurface(radius=2, outwards_normal=np.array([1, 0, 0]),
                                              center=np.array([-1, 0, 0]), curvature_sign=CurvatureSigns.convex)}

    rays = {'right': Ray(origin=np.array([0, 0, 0]), k_vector=np.array([1, 0, 0])),
            'left': Ray(origin=np.array([0, 0, 0]), k_vector=np.array([-1, 0, 0]))}

    ray = rays[ray]
    surface = surfaces[surface]

    a = surface.find_intersection_with_ray_exact(ray)
    fig, ax = plt.subplots(figsize=(8, 8))
    ray.plot(ax, label='ray', color='r')
    surface.plot(ax, label='surface', color='g', length=4 * np.pi, linewidth=0.5)
    surface.plot(ax, label='surface', color='b', linewidth=2.5)
    ax.scatter(a[0], a[1], label='Intersection')
    ax.scatter(ray.origin[0], ray.origin[1], label='origin')
    ax.quiver(ray.origin[0], ray.origin[1], ray.k_vector[0], ray.k_vector[1], label='k_vector')
    plt.legend()
    plt.xlim(-5.1, 5.1)
    plt.ylim(-5.1, 5.1)
    plt.show()


widgets.interact(f,
                 surface=widgets.Dropdown(options=['convex_right', 'concave_right', 'convex_left', 'concave_left'],
                                          value='convex_right', description='surface'),
                 ray=widgets.Dropdown(options=['right', 'left', ], value='right', description='ray')
                 )

