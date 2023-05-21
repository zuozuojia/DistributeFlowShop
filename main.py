from schedule import Utils, Objective, DFsp
from schedule.ga import GaDFsp
from schedule.name import GaName, DataName
import pandas as pd
import numpy as np
import copy
import os
import matplotlib.pyplot as plt

pd_tmp = pd.read_table('D:\\603\\DATASET_\\DPFSP\\DPFSP_Small\\4\\I_4_6_5_1.txt',sep=' ',header=None,engine='python') # 309
JOB_NUMBER = int(pd_tmp.values[0][0])
MACHINE_NUMBER_IN_A_FACTORY = int(pd_tmp.values[0][2])
FACTORY_NUMBER = int(pd_tmp.values[1][0])

C_max_calcu = []
process_time = np.array(np.zeros((JOB_NUMBER,MACHINE_NUMBER_IN_A_FACTORY)))

for i in range(2,JOB_NUMBER+2):
    index_i_tmp = pd_tmp.values[i][0]
    index_i_tmp = index_i_tmp.split()
    for j in range(MACHINE_NUMBER_IN_A_FACTORY):
        process_time[i-2][j] = index_i_tmp[j*2+1]
X_ = np.arange(0, MACHINE_NUMBER_IN_A_FACTORY, 1)
X_ = X_.tolist()
X__ = [X_] * JOB_NUMBER
X__ = np.array(X__)
X__ = [X__] * FACTORY_NUMBER
Y_ = copy.deepcopy(X__)
for i in range(len(Y_)): # factory
    for j in range(len(Y_[0])): # job
        for k in range(len(Y_[0][0])): # machine
            Y_[i][j][k] = process_time[j][k]
C_max_calcu = []
def main_dfsp():
# =============================================================================
#     w, n, m, low, high = 3, 10, 4, 5, 30
#     ops, prt = Utils.crt_data_dfsp(w, n, m, low, high)
#     print(ops)
#     print(prt)
# =============================================================================
    w = FACTORY_NUMBER
    n = JOB_NUMBER
    m = MACHINE_NUMBER_IN_A_FACTORY
    ops = X__
    prt = Y_

    # dfsp = DFsp(w, n, m, ops, prt)
    # job = dfsp.code_job_dfsp()
    # wkc = dfsp.code_wkc_dfsp()
    # info = dfsp.decode_dfsp(job, wkc)
    # info.ganttChart_png(file_name="GanttChart-dfsp-1")
    para = {
        GaName.pop_size: 40,
        GaName.rate_crossover: 0.65,
        GaName.rate_mutation: 0.35,
        GaName.operator_crossover: "pmx",
        GaName.operator_mutation: "tpe",
        GaName.operator_selection: "roullete",
        GaName.stop_max_generation: 50,
        GaName.stop_max_stay: 30,
        GaName.function_objective: Objective.makespan
    }
    data = {
        DataName.w: w,
        DataName.n: n,
        DataName.m: m,
        DataName.ops: ops,
        DataName.prt: prt
    }
    ga_fsp = GaDFsp(para, data)
    read_calcu = ga_fsp.start_generation()
    ga_fsp.global_best_info.ganttChart_png(file_name="GanttChart-dfsp-ga-1")
    ga_fsp.objective_png(file_name="Objective-dfsp-ga-1")
    ga_fsp.runtime_png(file_name="Runtime-dfsp-ga-1")
    
    C_max_calcu.append(read_calcu)

# =============================================================================
# main_dfsp()
# =============================================================================

# =============================================================================
# def run():
#     main_dfsp()
# 
# 
# if __name__ == "__main__":
#     run()
# =============================================================================

# multi file start
ifparametersweep = 0
ifshowdetails = 0
C_max_real = []
answer_file = pd.read_excel('D:\\603\\DATASET_\\DPFSP\\DPFSP_New_Best_2017.xls')

path_tmp = 'D:/603/DATASET_/InstancesAndBounds/VRF_Instances/Small'
path_tmp = 'D:\\603\\DATASET_\\DPFSP\\DPFSP_Small\\4'
files = os.listdir(path_tmp)
for file in files:
    print(file)
    pd_tmp = pd.read_table(path_tmp+'/'+file,sep=' ',header=None,engine='python') # 309
    index_ = answer_file[answer_file.Instance==file]
    answer_now = int(index_.Best)
    C_max_real.append(answer_now)
    JOB_NUMBER = int(pd_tmp.values[0][0])
    MACHINE_NUMBER_IN_A_FACTORY = int(pd_tmp.values[0][2])
    FACTORY_NUMBER = int(pd_tmp.values[1][0])
    STATE_NUMBER = 2**(JOB_NUMBER+MACHINE_NUMBER_IN_A_FACTORY)

    process_time = np.array(np.zeros((JOB_NUMBER,MACHINE_NUMBER_IN_A_FACTORY)))
    for i in range(2,JOB_NUMBER+2):
        index_i_tmp = pd_tmp.values[i][0]
        index_i_tmp = index_i_tmp.split()
        for j in range(MACHINE_NUMBER_IN_A_FACTORY):
            process_time[i-2][j] = index_i_tmp[j*2+1]
    X_ = np.arange(0, MACHINE_NUMBER_IN_A_FACTORY, 1)
    X_ = X_.tolist()
    X__ = [X_] * JOB_NUMBER
    X__ = np.array(X__)
    X__ = [X__] * FACTORY_NUMBER
    Y_ = copy.deepcopy(X__)
    for i in range(len(Y_)): # factory
        for j in range(len(Y_[0])): # job
            for k in range(len(Y_[0][0])): # machine
                Y_[i][j][k] = process_time[j][k]
    
    main_dfsp()

plt.figure(figsize=(20, 10), dpi=100)
x_line = list(range(len(C_max_calcu)))
plt.ylabel(u'Makespan') 
plt.xlabel(u'Number of file')
pp1, = plt.plot(x_line, C_max_calcu, markersize=4, label=u"Makespan_Calculate")
pp2, = plt.plot(x_line, C_max_real, markersize=10, label=u"Makespan_Answer")
plt.legend(handles=[pp1,pp2], labels=['Makespan_Calculate','Makespan_Answer'], loc='lower right', prop={'size':20})
plt.show()
# multi files end











