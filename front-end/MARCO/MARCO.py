import numpy as np
import json
from libs import init,Get_Neighborhood,Get_mapping_exe_time,Get_detailed_data,Get_rand_computation_ability2,CVB_method,read_NoC,init_from_json,Get_weight
import copy
from routing import improved_routeCompute
import time
import sys
import getopt
Best_reward=999999999
PEs_task_best_solution = {}
Tasks_position_best_solution = []
best_task_graph ={}


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

#MapResults里task的编号从0开始，包含路由的task_graph的编号从1开始
def handle_my_output(tg_file_name,MapResults,task_graph,computation_ability,num_of_rows,output_file_name):
    hyperperiod,num_of_tasks,edges,comp_cost=init(tg_file_name)
    adj_matrix,total_needSend,total_needReceive,execution=Get_detailed_data(num_of_tasks,edges,comp_cost)
    ret_task_graph={}#任务编号从0开始，包括key和out_links
    for i in task_graph.keys():
        ret_task_graph.update({})
        task_graph[i].update({'input_links':[]})
        task_graph[i].update({'start_time':0})
        task_graph[i].update({'visited':0})
        task_graph[i].update({'total_needSend':total_needSend[int(i)]})
        task_graph[i].update({'end_time':0})
        task_graph[i].update({'total_needReceive':total_needReceive[int(i)]})
        mapto=MapResults[int(i)-1]
        task_graph[i].update({'mapto':mapto})
        task_graph[i].update({'exe_time':computation_ability[int(i)-1][mapto]})
        #处理out_link
        for j in range(len(task_graph[i]['out_links'])):
            task_graph[i]['out_links'][j][0]=int(task_graph[i]['out_links'][j][0])
            task_graph[i]['out_links'][j].insert(2,[])
            # task_graph[i]['out_links'][j][3]=[ task_graph[i]['out_links'][j][3] ]
            task_graph[i]['out_links'][j][-2]=mapto
            dest_position=MapResults[task_graph[i]['out_links'][j][0]-1]
            task_graph[i]['out_links'][j][-1]=dest_position
            task_graph[i]['out_links'][j]=[ task_graph[i]['out_links'][j] ]
            #task_graph[i]['out_links'][j].append(0)
    
    #将task的编号改成从0开始，包括key和out_link里的task
    for i in task_graph.keys():
        cur_key=str(int(i)-1)
        ret_task_graph.update({cur_key:task_graph[i]})
        for j in range(len(ret_task_graph[cur_key]['out_links'])):
            ret_task_graph[cur_key]['out_links'][j][0][0]-=1
    
    with open(output_file_name,"w") as f:
        f.write(json.dumps(ret_task_graph,cls=MyEncoder))
    print("write done")

def read_NoC(NoC_file_name):
    ret=[]
    f=open(NoC_file_name)
    for line in f:
        tmp=[]
        for i in line[1:-2].split(','):
            tmp.append(int(i))
        ret.append(tmp)
    return ret

def Initialization(num_of_tasks,PEs_task_current_solution,Tasks_position_current_solution,computation_ability,M,N): 
    for i in range(0,M*N):
        PEs_task_current_solution.append([])
    for i in range(0,num_of_tasks):
        """
        rand_tmp=np.random.randint(0,M*N)
        PEs_task_current_solution[rand_tmp].append(i)
        Tasks_position_current_solution.update({i:rand_tmp})
        """
        tmp_exe_time=99999999
        ret=-1
        for j in range(0,len(computation_ability[i])):
            if(computation_ability[i][j]<tmp_exe_time):
                ret=j
                tmp_exe_time=computation_ability[i][j]
        PEs_task_current_solution[ret].append(i)
        Tasks_position_current_solution.update({i:ret})
    
    print(Tasks_position_current_solution)

def Update_tabu_list(choosen_position, target_position,Tabu_list,Tabu_length):
    #delete front element in queue
    if(len(Tabu_list)==Tabu_length): 
        Tabu_list.pop(0)

    #insert new tabu element
    s=str(choosen_position)+" "+str(target_position)
    Tabu_list.append(s)




def Iteration(num_of_tasks,radius,num_of_rows,Tasks_position_current_solution,M,N,Tabu_list,PEs_task_current_solution,execution,computation_ability,adj_matrix,weight_a,weight_b,Tabu_length): #expand neighborhood, find the fittest, update best solution and tabu list
    randomly_selected_task=np.random.randint(0,num_of_tasks) #randomly choose a task
    current_position=Tasks_position_current_solution[randomly_selected_task]
    neighborhood=Get_Neighborhood(current_position,radius,M,N)
    target_position=-1
    tmp_reward=999999999
    cur_task_graph={}
    for i in neighborhood: #visit all neighborhood and find the fittest one in these neighborhoods
        print("neighborhood:",i)
        flag=False
        for j in Tabu_list:
            source=j.split()[0]
            destination=j.split()[1]
            if(source==current_position and destination==i):
                flag=True
                break
            if(source==i and destination==current_position):
                flag=True
                break
        if(flag):
            #print(i," is baned")
            #print("------------")
            continue
        tmp_solution=copy.deepcopy(PEs_task_current_solution)
        tmp_solution[current_position].remove(randomly_selected_task)
        tmp_solution[i].append(randomly_selected_task)

        tmp_mapresults=copy.deepcopy(Tasks_position_current_solution)
        tmp_mapresults.update({randomly_selected_task:i})

        mapping_exe_time=Get_mapping_exe_time(PEs_task_current_solution=tmp_solution,Tasks_position_current_solution=tmp_mapresults,computation_ability=computation_ability,num_of_rows=num_of_rows,execution=execution)


        tmp_mapresults1=[-1]#把task编号改成从1开始，然后传给routing部分
        for j in range(0,num_of_tasks):
            tmp_mapresults1.append(tmp_mapresults[j])
        #通过RL计算最佳routing，然后获得routing的reward
        #首先根据MapResult更新execution
        execution_to_routing=copy.deepcopy(execution)
        for j in range(1,num_of_tasks+1):
            #execution_to_routing[j]=int(execution_to_routing[j]/computation_ability[int(tmp_mapresults1[j]/num_of_rows)][tmp_mapresults1[j]%num_of_rows])
            execution_to_routing[j]=computation_ability[j-1][tmp_mapresults1[j]]
        #计算路由
        #pendTimes,ret_task_graph=routeCompute(adj_matrix,num_of_tasks,execution_to_routing,num_of_rows,tmp_mapresults1)
        pendTimes,ret_task_graph=improved_routeCompute(adj_matrix,num_of_tasks,execution_to_routing,num_of_rows,tmp_mapresults1)
        total=weight_a*mapping_exe_time+weight_b*pendTimes#这个值应该是越小越好，第一个值小表示mapping结果匹配到的PE计算能力强，第二个值小表示routing中争用少

        if(total<tmp_reward):
            tmp_reward=total
            target_position=i
            cur_task_graph=copy.deepcopy(ret_task_graph)
    
    #update the current solution
    PEs_task_current_solution[current_position].remove(randomly_selected_task)
    PEs_task_current_solution[target_position].append(randomly_selected_task)
    Tasks_position_current_solution.update({randomly_selected_task:target_position})
    print("reward in this iteration is ",tmp_reward)
  
    #update the best solution
    global Best_reward
    if(tmp_reward<Best_reward):#越小越好，详见total=mapping_exe_time+pendTimes处的注释
        #print("Update Best sol")
        Best_reward=tmp_reward
        global PEs_task_best_solution
        PEs_task_best_solution=copy.deepcopy(PEs_task_current_solution)
        global Tasks_position_best_solution
        Tasks_position_best_solution=copy.deepcopy(Tasks_position_current_solution)
        global best_task_graph
        best_task_graph=copy.deepcopy(cur_task_graph)

    Update_tabu_list(current_position, target_position,Tabu_list,Tabu_length)
    #Update_tabu_list(current_position,target_position)
        

def computingMARCO(inputfile, rowNum, confile,vmachine):
    #M=computation_ability.shape[0]
    #N=computation_ability.shape[1]
    M=int(rowNum)#num_of_rows
    N=int(rowNum)#num_of_rows

    #init input
    hyperperiod,num_of_tasks,edges,comp_cost=init(inputfile)
    adj_matrix,total_needSend,total_needReceive,execution=Get_detailed_data(num_of_tasks,edges,comp_cost)
    weight_a,weight_b=Get_weight(total_needSend,execution)
    if confile == "":
        computation_ability=CVB_method(execution=execution[1:],V_machine=vmachine,num_of_rows=N)
    else:
        computation_ability=read_NoC(confile)


    #M*N mesh network, task placed on (i,j) PE
    #current solution
    #task编号都是从0开始的，跟routing部分不一样，需要适配一下
    PEs_task_current_solution=[] #PEs_task_current_solution[i] is a list, means pending tasks of PE(i/N,i%N)
    Tasks_position_current_solution={}#key-value: key=task, value=position in mesh


    #best solution


    Tabu_list=[]
    Tabu_length=2
    Search_radius=1


    total_start_time=time.time()
    Initialization(num_of_tasks,PEs_task_current_solution,Tasks_position_current_solution,computation_ability,M,N)

    #print(Tasks_position_current_solution)
    f = open("./log-MARCO.txt", 'w+')

    for i in range(0,150):
        print("Iteration ",i,":")
        iteration_start_time=time.time()
        Iteration(num_of_tasks,Search_radius,N,Tasks_position_current_solution,M,N,Tabu_list,PEs_task_current_solution,execution,computation_ability,adj_matrix,weight_a,weight_b,Tabu_length)
        iteration_end_time=time.time()
        print("This iteration costs ",(iteration_end_time-iteration_start_time)/60,"mins")
        if((i+1) % 10 ==0):
            print("write result")
            print("---------------",file=f)
            print(Tasks_position_best_solution,file=f)
            print(best_task_graph,file=f)
            print("---------------",file=f)
    #print(PEs_task_current_solution)
    #print(Get_reward(PEs_task_current_solution))
    total_end_time=time.time()
    print("It costs ",(total_end_time-total_start_time)/60,"mins")
    print("Best reward=",Best_reward)
    print("Now is best result:")
    print(Tasks_position_best_solution)
    print(best_task_graph)
    #print(computation_ability,file=f)
    print("[",file=f)
    for i in computation_ability:
        print("[",end="",file=f)
        for j in i:
            print(j,end=",",file=f)
        print("]",file=f)
    print("]",file=f)
    f.close()
    print("done")
    return Tasks_position_best_solution,best_task_graph,computation_ability

def main(argv):
    inputfile = ""
    rowNum = ""
    outputfile = ""
    confile = ""
    vmachine = 0.25
    try:
        opts, args = getopt.getopt(argv,"hi:r:o:c:v:",["ifile=","row=","ofile=","cfile=","vmachine="])
    except getopt.GetoptError:
        print('Error MARCO.py -i <inputfile> -r <row> -o <outputfile> -c <configurefile> -v <v_machine>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('MARCO.py -i <inputfile> -r <row> -o <outputfile> -c <configurefile> -v <v_machine>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            print("read input file")
            inputfile = arg
        elif opt in ("-r", "--row"):
            rowNum = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-c", "--cfile"):
            confile = arg
        elif opt in ("-v", "--vmachine"):
            vmachine = float(arg)
    Tasks_position_best_solution,best_task_graph,computation_ability = computingMARCO(inputfile, rowNum, confile, vmachine)
    handle_my_output(inputfile,Tasks_position_best_solution,best_task_graph,computation_ability,num_of_rows=int(rowNum),output_file_name=outputfile)
   




if __name__ == "__main__":
    main(sys.argv[1:])









    

    

        
    


