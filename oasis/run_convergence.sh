#python NSfracStep.py problem=MMS N=50 dt=$1 T=0$(expr 20*$1 | bc) solver=IPCS

python NSfracStep.py problem=MMS N=20 dt=0.0001 T=0.0003 solver=IPCS
python NSfracStep.py problem=MMS N=50 dt=0.0001 T=0.0003 solver=IPCS
python NSfracStep.py problem=MMS N=100 dt=0.0001 T=0.0003 solver=IPCS
#python NSfracStep.py problem=MMS N=200 dt=0.001 T=0.003 solver=IPCS

#python NSfracStep.py problem=MMS N=20 dt=0.001 T=0.003 solver=BDFPC
#python NSfracStep.py problem=MMS N=50 dt=0.001 T=0.003 solver=BDFPC
#python NSfracStep.py problem=MMS N=100 dt=0.001 T=0.003 solver=BDFPC
#python NSfracStep.py problem=MMS N=200 dt=0.001 T=0.003 solver=BDFPC
