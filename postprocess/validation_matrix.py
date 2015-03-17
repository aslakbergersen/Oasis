#map = get_data()
#data = get_variance(map)

def compute_validation_matrix(results, data, filepath)
    error = 0
    eval = 0
    tol = 1e-5

    for key, item in results.items():
        if file.endswith("UMean.xy") and not "X1_UMean" in file and not \
        "lineX14" in file and not "lineX15" in file:
            x, U = readfile(path.join(folder,file))
            x = x[1]
            U = U[0]
            u = data[line_name_map[file.split("_")[0]]]
            
            x_piv = u[-1]
            u_piv = u[0]
            counter = 0
            for x_punkt in x_piv:
                for j in range(len(x)-1):
                    if x_punkt + tol <= x[j] and x_punkt + tol >= x[j+1]:
                        vekt_rel = abs(x[j] - x[j+1])
                        vekt_right = abs(x[j] - x_punkt) / vekt_rel
                        vekt_left = abs(x[j+1] - x_punkt) / vekt_rel
                        u_this = U[j]*vekt_left + vekt_right*U[j+1]
                        break

                if "omega" in folder:
                    error_omega += abs((u_piv[counter] - u_this) / u_piv[counter])
                else:
                    error_epsilon += abs((u_piv[counter] - u_this) / u_piv[counter])
                counter += 1

            eval += len(u[-1])

error_omega = error_omega / eval
error_epsilon = error_epsilon / eval

print "Omega:", error_omega, "Epsilon", error_epsilon
