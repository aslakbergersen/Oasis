from os import path
from compare import map_filenames

def compute_validation_matrix(results, data, filepath, legend):
    error = 0
    eval = 0
    tol = 1e-5

    if 0 in results.keys():
        comp_list = results.keys()
    else:
        comp_list = [0]
        results = {0: results}

    error_list = [0 for i in range(len(results.keys()))]
    eval_list = [0 for i in range(len(results.keys()))]

    for key, item in data.items():
        key_re, element = map_filenames(key)
        if key_re is not None and "slice_u" in key_re and "slice_u_r" not in key_re:

            u = data[key]
            x_piv = u[-1]
            u_piv = u[0]

            for k in comp_list:
                U = results[k]["array"][key_re][:,2]
                x = results[k]["points"]["_".join(key_re.split("_")[::2])][:,0]
           
                counter = 0
                for x_punkt in x_piv:
                    for j in range(len(x)-1):
                        if x_punkt > 0:
                            b = x_punkt + tol >= x[j] and x_punkt + tol <= x[j+1]
                        else: 
                            b = abs(x_punkt + tol) <= abs(x[j]) and abs(x_punkt + tol) >= abs(x[j+1])
                        if b:    
                            vekt_rel = abs(x[j] - x[j+1])
                            vekt_right = abs(x[j] - x_punkt) / vekt_rel
                            vekt_left = abs(x[j+1] - x_punkt) / vekt_rel
                            u_this = U[j]*vekt_left + vekt_right*U[j+1]
                            break

                    if abs(u_piv[counter]) > 0.01:
                        error_list[k] += abs((u_piv[counter] - u_this) / u_piv[counter])
                        counter += 1

                eval_list[k] += counter


    if legend is None and len(comp_list) == 1:
        for i in comp_list:
            print "Validation matric: %s" % (error_list[i] / eval_list[i])
    if legend is not None:
        for i in comp_list:
            print "Validation matric for %s: %s" % (legend[i], error_list[i] / eval_list[i])


#if __name__ == "__main__":
#    pass
