from routing import splitRoute,getMaxOne,link_item
import math
import random
import numpy as np
from queue import Queue
import copy
import sys
import getopt

import json
from libs import init,Get_Neighborhood,Get_mapping_exe_time,Get_detailed_data,Get_rand_computation_ability2,CVB_method,read_NoC,init_from_json,Get_weight,MyEncoder,Update_link_set
import time
import os
from re import findall


#adj_matrix里task的编号是从1开始的，execution和MapResult也是从1开始编号
def splitted_routeCompute(adj_matrix,num_of_tasks,execution,num_of_rows,MapResult,log_file):
    #初始化
    task_graph={}
    link_set=[]
    receiveMatrix = [-1]
    total_link_num=(num_of_rows-1+num_of_rows)*(num_of_rows-1)+num_of_rows-1
    for i in range(0,total_link_num):
        tmp=link_item()
        link_set.append(tmp)
    for i in range(1,num_of_tasks+1):#初始化receive_matrix，这里遍历的是每一列
        total_receive_for_i=0
        for j in range(1,num_of_tasks+1):
            total_receive_for_i+=adj_matrix[j][i]
        receiveMatrix.append(total_receive_for_i)

    edge_queue=[]#每个item为( 'task_source,task_dest' , end time of task_source )，如('1,2',20)，task的编号从1开始
    #添加一开始就能执行的边
    for i in range(1,num_of_tasks+1):
        if(receiveMatrix[i]==0):#这个task可以立刻执行，然后开始传输
            for j in range(1,num_of_tasks+1):
                if(adj_matrix[i][j]!=0):
                    tmp=(str(i)+','+str(j),execution[i])
                    edge_queue.append(tmp)
    edge_queue.sort(key=lambda x: x[1])#按照task_source的结束时间排序

    while(len(edge_queue)!=0):
        current_edge=edge_queue[0]
        edge_queue.pop(0)
        current_source_task=int(current_edge[0].split(',')[0])
        current_dest_task=int(current_edge[0].split(',')[1])
        start_time=current_edge[1]#这条边的预计传输开始时间，也就是source_task的预计结束时间
        current_transmission=adj_matrix[current_source_task][current_dest_task]

        #向task_graph中添加这条边以及两个任务节点
        if(str(current_source_task) not in task_graph.keys()):#task_source不在task graph中
            task_graph.update({str(current_source_task):{'out_links':[[str(current_dest_task),current_transmission,[],0,0,-1]]}})
        else:#task_source在task graph中，仅需要更新出边及相应参数
            task_graph[str(current_source_task)]['out_links'].append([str(current_dest_task),current_transmission,[],0,0,-1])
        if(str(current_dest_task) not in task_graph.keys()):#task_dest不在task graph中
            task_graph.update({str(current_dest_task):{'out_links':[]}})

        #开始为边current_source_task->current_dest_task计算路由
        print("handle edge",current_source_task,"to",current_dest_task,"now,","cur transmission is",current_transmission,", source is",MapResult[current_source_task],"des is",MapResult[current_dest_task],file=log_file)

        #处理完之后直接跳到下一条边
        if(MapResult[current_source_task]==MapResult[current_dest_task]):
            #传输结束后，需要计算是否有新的边可以加入队列
            receiveMatrix[current_dest_task]-=current_transmission
            if(receiveMatrix[current_dest_task]==0):#task_dest已经可以执行，那么将它的出边加入到队列中
                for i in range(1,num_of_tasks+1):
                    if(adj_matrix[current_dest_task][i]!=0):
                        edge_queue.append( (str(current_dest_task)+','+str(i) , start_time+cur_best_contention+execution[current_dest_task]) )#task_dest的结束时间，应当是source->dest的出边传输结束时间再加上task_dest的执行时间
                edge_queue.sort(key=lambda x: x[1])
            continue

        cur_best_contention=999999999
        cur_best_route=[]
        cur_best_sub_contention=[]
        cur_best_sub_msg_size=[]
        for split_number in range(1,5):
            #print("for split number=",split_number,":")
            sub_contention,sub_route,sub_msg_size=splitRoute(MapResult[current_source_task],MapResult[current_dest_task],num_of_rows,link_set,split_number,start_time,current_transmission)
            if(getMaxOne(sub_contention)<=cur_best_contention):
                #print("Update!")
                cur_best_contention=getMaxOne(sub_contention)
                cur_best_route=copy.deepcopy(sub_route)
                cur_best_sub_contention=copy.deepcopy(sub_contention)
                cur_best_sub_msg_size=copy.deepcopy(sub_msg_size)
        
        #print("sub msg size is",sub_msg_size)
        print("best split number is",len(cur_best_route),file=log_file)
        
        for i in range(0,len(cur_best_route)):
            Update_link_set(cur_best_route[i],link_set,num_of_rows,start_time+cur_best_sub_contention[i],start_time+cur_best_sub_msg_size[i]+cur_best_sub_contention[i])
            if(i==0):#直接更新task graph里的outlink，不需要额外添加
                for j in range(0,len(task_graph[str(current_source_task)]['out_links'])):
                    if(int(task_graph[str(current_source_task)]['out_links'][j][0])==current_dest_task):
                        task_graph[str(current_source_task)]['out_links'][j][2]=cur_best_route[i]
                        task_graph[str(current_source_task)]['out_links'][j][1]=cur_best_sub_msg_size[i]
            else:
                new_link=[str(current_dest_task),cur_best_sub_msg_size[i],cur_best_route[i],0,0,-1]
                task_graph[str(current_source_task)]['out_links'].append(new_link)

        #传输结束后，需要计算是否有新的边可以加入队列
        receiveMatrix[current_dest_task]-=current_transmission
        if(receiveMatrix[current_dest_task]==0):#task_dest已经可以执行，那么将它的出边加入到队列中
            for i in range(1,num_of_tasks+1):
                if(adj_matrix[current_dest_task][i]!=0):
                    edge_queue.append( (str(current_dest_task)+','+str(i) , start_time+cur_best_contention+execution[current_dest_task]) )#task_dest的结束时间，应当是source->dest的出边传输结束时间再加上task_dest的执行时间
            edge_queue.sort(key=lambda x: x[1])
    
    return task_graph


def main(argv):
    f = open("./log-LAMP.txt", 'w+')

    #NoC_description_dict={'audiobeam':'N22_audiobeam_Mesh','autocor':'N12_autocor_Mesh','fmradio':'N31_fmradio_Mesh','H264':'N51_H264_Mesh'}
    
    inputfile = ''
    rowNum = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:r:o:",["ifile=","row=","ofile="])
    except getopt.GetoptError:
        print('Error LAMP.py -i <inputfile> -r <row> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('LAMP.py -i <inputfile> -r <row> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            print("read input file")
            inputfile = arg
        elif opt in ("-r", "--row"):
            rowNum = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg

    # print('inputfile：', inputfile)
    # print('row: ', rowNum)
    # print('outputfile: ',outputfile)
    
    num_of_rows=int(rowNum)
    #NoC_description_file=NoC_description_dict[str_split[0]]+str(num_of_rows)+'x'+str(num_of_rows)+'_NoCdescription.txt'
    #computation_ability=read_NoC('./NoC description/'+NoC_description_file)
    adj_matrix,total_needSend,total_needReceive,execution,num_of_tasks=init_from_json(inputfile)
    task_graph={}
    with open(inputfile,'r') as f1:
        task_graph=json.load(f1)
    
    #for i in task_graph.keys():
    #    mapto=task_graph[i]['mapto']
    #    task_graph[i]['exe_time']=computation_ability[int(i)][mapto]

    map_results=[-1]#把task编号改成从1开始，然后传给routing部分
    for i in range(0,num_of_tasks):
        map_results.append(task_graph[str(i)]['mapto'])
    execution_to_routing=copy.deepcopy(execution)
    #for i in range(1,num_of_tasks+1):
    #    execution_to_routing[i]=computation_ability[i-1][map_results[i]]
    
    route=splitted_routeCompute(adj_matrix,num_of_tasks,execution_to_routing,num_of_rows,map_results,log_file=f)
    
    

    ret_route={}
    for i in route.keys():
        for j in range(len(route[i]['out_links'])):
            route[i]['out_links'][j][0]=int(route[i]['out_links'][j][0])
            route[i]['out_links'][j].insert(2,[])
            #route[i]['out_links'][j][3]=[ route[i]['out_links'][j][3] ]
            mapto=map_results[int(i)]
            route[i]['out_links'][j][-2]=mapto
            dest_position=map_results[1:][route[i]['out_links'][j][0]-1]
            route[i]['out_links'][j][-1]=dest_position
            route[i]['out_links'][j]=[ route[i]['out_links'][j] ]
    for i in route.keys():
        cur_key=str(int(i)-1)
        ret_route.update({cur_key:route[i]})
        for j in range(len(ret_route[cur_key]['out_links'])):
            ret_route[cur_key]['out_links'][j][0][0]-=1
    
    for i in task_graph.keys():
        task_graph[i]['out_links']=ret_route[i]['out_links']
    
    #print(task_graph)
   
    with open(outputfile,"w") as f2:
        print("file saved in "+outputfile)
        f2.write(json.dumps(task_graph,cls=MyEncoder))
    # print(output_file_name,'done')
    # print("-------------",file=f)
    
    f.close()
    print("LAMP completed!")


if __name__ == "__main__":
    main(sys.argv[1:])