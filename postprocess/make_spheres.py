from probes_stats import WritePolyData
import vtk
import os

def write_spheres(points, radius=None, name="sphere%s.vtp", base=0.0005):
    radius = [base]*len(points) if radius is None else radius
    for counter, point in enumerate(points):
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(point)
        sphere.SetPhiResolution(500)
        sphere.SetThetaResolution(500)
        sphere.SetRadius(radius[counter])

        WritePolyData(sphere.GetOutput(), "spheres/" + name % counter)


if __name__ == "__main__":
    """
    data = {'bif': {'i_div': 899, 'end_point': (52.080604553222656,
        48.686641693115234, 45.14276885986328), 'r_div': 1.4397806781365265,
        'r_end': 1.4918430524643953, 'ID_div': 312L, 'div_point':
        (52.683834075927734, 49.727909088134766, 44.91619110107422)}, 0:
        {'end_point': (51.84696578979492, 52.263038635253906,
            43.87771987915039), 'r_end': 0.8699254769222771, 'ID_end': 556L,
            'r_div': 0.9671943002873735, 'ID_div': 538L, 'div_point':
            (52.41809844970703, 51.54286193847656, 44.2702522277832)}, 1:
        {'end_point': (55.00656509399414, 51.79118728637695,
            44.51828384399414), 'r_end': 1.158268137236508, 'ID_end': 495L,
            'r_div': 1.2362120276839503, 'ID_div': 515L, 'div_point':
            (53.917728424072266, 51.206634521484375, 44.67728805541992)}}
    """
    data = {'bif': {'i_div': 904, 'end_point': (52.31532287597656,
        49.077362060546875, 45.10821533203125), 'r_div': 1.3880181350244833,
        'r_end': 1.4775190508983251, 'ID_div': 307L, 'div_point':
        (52.79876708984375, 50.09815216064453, 44.77281951904297)}, 0:
        {'end_point': (51.99351501464844, 52.153709411621094,
            43.95381546020508), 'r_end': 0.8808746601508601, 'ID_end': 552L,
            'r_div': 1.0202088418579196, 'ID_div': 535L, 'div_point':
            (52.52396011352539, 51.370853424072266, 44.33856201171875)}, 1:
        {'end_point': (54.9153938293457, 51.752349853515625,
            44.53167724609375), 'r_end': 1.1484698483588718, 'ID_end': 496L,
            'r_div': 1.2514631524760735, 'ID_div': 519L, 'div_point':
            (53.83111572265625, 51.14166259765625, 44.67278289794922)}}

    points = [data[k]["end_point"] for k in data.keys()]
    write_spheres(points, name="sphere_anu_end%s.vtp", base=0.15)

    points = [data[k]["div_point"] for k in data.keys()]
    radius = [data[k]["r_div"] for k in data.keys()]
    write_spheres(points, radius=radius, name="sphere_P0252_anu%s.vtp")
    write_spheres(points, name="sphere_P0252_anu_small%s.vtp", base=0.15)
    
    center = [[(points[0][i] + points[1][i] + points[2][i]) / 3 for i in range(3)]]
    write_spheres(center, name="sphere_P0252_center%s.vtp", base=0.15)
