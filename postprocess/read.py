text = """5M
Length on data set names: 92
Hmin: 6.73784175135e-05
Hmax: 0.000727102871211
Hmean: 0.000353532559934
l+ mean: 3.47801568109
l+ max: 24.4349994426
l+ min: 0.219140969065
t+ mean: 0.000144823576499
t+ max: 0.00220866195923
t+ min: 1.16751576881e-05
Length scale mean: 0.000121781626322
Length scale max: 0.00185725592402
Length scale min: 9.8175982474e-06
Time scale mean: 0.00599400822607
Time scale max: 1.04073314191
Time scale min: 2.90808024363e-05
Velocity scale mean: 0.040667175242
Velocity scale max: 0.360329237959
Velocity scale min: 0.00178419548738
CFL mean: 0.00998115530342
CFL max: 0.173672334592
CFL min: 7.80020855511e-05
SSV mean: 52049.1388493
SSV max: 2237002.36648
SSV min: 62.5076929703
dissipation mean: 11.7090787414
dissipation max: 5086.20530766
dissipation min: 3.05749404812e-06

10M
Length on data set names: 100
Hmin: 2.25969042023e-05
Hmax: 0.000509601396455
Hmean: 0.000277560712898
l+ mean: 2.64958852956
l+ max: 19.8321870114
l+ min: 0.134962174243
t+ mean: 0.000150653936557
t+ max: 0.002531227582
t+ min: 1.07487353582e-05
Length scale mean: 0.000126684355195
Length scale max: 0.00212850019989
Length scale min: 9.03857303121e-06
Time scale mean: 0.00647599231914
Time scale max: 1.36692052428
Time scale min: 2.46487906792e-05
Velocity scale mean: 0.0389747411489
Velocity scale max: 0.366694372089
Velocity scale min: 0.00155717362116
CFL mean: 0.0126801212871
CFL max: 0.207353866151
CFL min: 1.08260989357e-05
SSV mean: 55309.9708354
SSV max: 2868728.08646
SSV min: 51.7299117878
dissipation mean: 11.4224882266
dissipation max: 5455.22731778
dissipation min: 1.77396082991e-06

17M
Length on data set names: 100
Hmin: 6.18530978264e-05
Hmax: 0.000404674502094
Hmean: 0.000235959545016
l+ mean: 2.38190474266
l+ max: 18.1663595803
l+ min: 0.0676468346921
t+ mean: 0.000156399519872
t+ max: 0.00295019454415
t+ min: 1.03078509327e-05
Length scale mean: 0.000131515795608
Length scale max: 0.00248080801648
Length scale min: 8.66783489827e-06
Time scale mean: 0.00810352339058
Time scale max: 1.94553972907
Time scale min: 2.26682051674e-05
Velocity scale mean: 0.0403393719743
Velocity scale max: 0.382378518293
Velocity scale min: 0.000998681639427
CFL mean: 0.0160814145975
CFL max: 0.276801259198
CFL min: 9.05036788564e-07
SSV mean: 58460.0275768
SSV max: 3119377.01272
SSV min: 38.0805152034
dissipation mean: 11.7968187915
dissipation max: 6450.15005055
dissipation min: 3.00126356487e-07

28M
Length on data set names: 100
Hmin: 4.24378561977e-05
Hmax: 0.000394723731755
Hmean: 0.000202765079445
l+ mean: 1.96782834582
l+ max: 14.98970287
l+ min: 0.0236237817474
t+ mean: 0.000159911439005
t+ max: 0.0050566249039
t+ min: 9.82247086238e-06
Length scale mean: 0.000134468955817
Length scale max: 0.00425209775498
Length scale min: 8.25968053711e-06
Time scale mean: 0.00802048554496
Time scale max: 5.59993120004
Time scale min: 2.05836493255e-05
Velocity scale mean: 0.0397615924379
Velocity scale max: 0.401273937269
Velocity scale min: 0.000662797413182
CFL mean: 0.0184474246016
CFL max: 0.304266854018
CFL min: 1.58632676192e-07
SSV mean: 55883.0675942
SSV max: 3435283.85081
SSV min: 13.7119182446
dissipation mean: 11.5017372921
dissipation max: 7822.75424257
dissipation min: 5.82262864837e-08

10M P2-P1
Length on data set names: 100
Hmin: 2.25969042023e-05
Hmax: 0.000509601396455
Hmean: 0.000277560712898
l+ mean: 2.64366061123
l+ max: 22.0506634943
l+ min: 0.0199645406134
t+ mean: 0.000166635847149
t+ max: 0.00620042863259
t+ min: 9.67224713636e-06
Length scale mean: 0.000139768514252
Length scale max: 0.00504069868761
Length scale min: 8.05744054924e-06
Time scale mean: 0.00951436378885
Time scale max: 8.09702652892
Time scale min: 1.95879999154e-05
Velocity scale mean: 0.0387823174348
Velocity scale max: 0.411345751652
Velocity scale min: 0.000580932496507
CFL mean: 0.013299186352
CFL max: 0.207750080839
CFL min: 1.51965836751e-07
SSV mean: 53438.2750838
SSV max: 3542822.19504
SSV min: 10.0501846071
dissipation mean: 10.8564703727
dissipation max: 8638.21360554
dissipation min: 3.43635922879e-08"""

lines = text.split("\n")
data = {}
new = True
for line in lines:
    if line == "":
        new = True
        continue
    if new:
        key = line
        new = False
        data[key] = {}
        continue

    split_line = line.split(": ")
    data[key][split_line[0]] = "%03.02g" % float(split_line[-1])


def sort(list):
    list.sort(key=lambda x: float(x.split("M")[0]))
    return list

cases = sort(data.keys())
cases.remove("10M P2-P1")
cases.insert(len(cases), "10M P2-P1")
text = " & ".join(cases) + " \\\\\n"
categories = data[cases[0]].keys()
categories.sort()

for cat in categories:
    tmp = [cat]
    for case in cases:
        tmp.append(data[case][cat])

    text += " & ".join(tmp) + " \\\\\n"

print text
