
import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


from libs import init,Get_Neighborhood,Get_detailed_data,find_start_task,get_sorted_dict,ActorCritic,Get_full_route_by_XY,Get_reward_by_pendTimes,Actor,Critic,computeContention,Update_link_set,Check_if_Done_improved,Environment_improved,getMaxOne
from queue import Queue
import copy




class link_item():#可以根据在list中的下标索引到它连接的是哪两个PE
    def __init__(self):
        #记录这个link的timeline，list中的每个元素是list，形式为[task_source,task_dest,start_time,end_time]
        self.timeline=[]
    


        
#单独计算某个msg的route
def computeMsgRoute(source,destination,num_of_rows,link_set,start_time,msg_size):
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    #开始为PE source->PE destination计算路由
    #state_tensor的四个channel,从0-3以此为N,S,W,E
    state_tensor=torch.Tensor(np.zeros((1,4,num_of_rows*num_of_rows),dtype=np.int)).to(device)
    state=[state_tensor,source,[]]#state为[state_tensor,cur_position,partRoute]
    best_Route=[]#这条边的最佳路由
    #确保当前的state(map后的位置)不是end state，至少能执行一次action
    tmp_state,_,tmp_done=Check_if_Done_improved(state,source,destination,link_set,num_of_rows,start_time,start_time+msg_size)

    if(tmp_done):#这两个task的位置无需计算route，直接结束，将结果存储到best_Route
        best_Route=tmp_state[2]
    else:
        actor=Actor(4,num_of_rows*num_of_rows,2).to(device)
        critic=Critic(4,num_of_rows*num_of_rows).to(device)
        adam_actor=optim.Adam(actor.parameters(),lr=1e-3)
        adam_critic=optim.Adam(critic.parameters(), lr=1e-3)
        gamma=0.99
            
        best_reward=-9999999
        for _ in range(100):
            done=False
            total_reward=0
            state_tensor=torch.Tensor(np.zeros((1,4,num_of_rows*num_of_rows),dtype=np.int)).to(device)
            state=[state_tensor,source,[]]#state为[state_tensor,cur_position,partRoute]

            while not done:
                state[0]=state[0].to(device)
                probs=actor(state[0])
                dist = torch.distributions.Categorical(probs=probs)
                action = dist.sample()

                next_state,reward,done=Environment_improved(state,int(action),source,destination,link_set,num_of_rows,start_time,start_time+msg_size)
                d=0
                next_state[0]=next_state[0].to(device)
                if(done):
                    d=1
                advantage=reward+(1-d)*gamma*critic(next_state[0])-critic(state[0])

                total_reward+=reward
                state=next_state

                critic_loss = advantage.pow(2).mean()
                adam_critic.zero_grad()
                critic_loss.backward()
                adam_critic.step()

                actor_loss = -dist.log_prob(action)*advantage.detach()
                adam_actor.zero_grad()
                actor_loss.backward()
                adam_actor.step()

            if(total_reward>best_reward):
                best_reward=total_reward
                best_Route=state[2]
    
    return best_Route

def computeNewRoute(new_source,new_destination,start_route,end_route,num_of_rows,link_set,start_time,msg_size):
    route=computeMsgRoute(new_source,new_destination,num_of_rows,link_set,start_time,msg_size)
    route.insert(0,start_route)
    route.append(end_route)
    contention=computeContention(route,link_set,num_of_rows,start_time,msg_size)
    return contention,route

def generateStartEndRoute(source,destination,num_of_rows):
    if(destination==(source+1)):
        return [source,'E']
    elif(destination==(source-1)):
        return [source,'W']
    elif(destination==(source+num_of_rows)):
        return [source,'S']
    elif(destination==(source-num_of_rows)):
        return [source,'N']

def initPorts(position,num_of_rows,second_position,mode):#mode=0表示posi->second_posi, mode=1表示second_posi->posi
    ret=[]
    row=(int)(position/num_of_rows)
    col=position%num_of_rows

    second_row=int(second_position/num_of_rows)
    second_col=second_position%num_of_rows

    if(mode==0):#src为position，des为second position
        if(row!=0 and (col!=second_col or row>second_row) ):#check N
            ret.append(position-num_of_rows)
        if(row!=(num_of_rows-1) and (col!=second_col or row<second_row) ):#check S
            ret.append(position+num_of_rows)
        if(col!=0 and (row!=second_row or col>second_col) ):#check W
            ret.append(position-1)
        if(col!=(num_of_rows-1) and (row!=second_row or col<second_col) ):#check E
            ret.append(position+1)
    elif(mode==1):#src为second position, des为position
        if(row!=0 and (col!=second_col or row>second_row) ):#check N
            ret.append(position-num_of_rows)
        if(row!=(num_of_rows-1) and (col!=second_col or row<second_row) ):#check S
            ret.append(position+num_of_rows)
        if(col!=0 and (row!=second_row or col>second_col) ):#check W
            ret.append(position-1)
        if(col!=(num_of_rows-1) and (row!=second_row or col<second_col) ):#check E
            ret.append(position+1)

    return ret

def delPorts(ports,route,num_of_rows,mode):#mode=0表示在处理source的ports，mode=1表示在处理des的ports
    ret=copy.deepcopy(ports)
    target=-1
    if(mode==1):
        target=route[0]
    else:
        if(route[1]=='N'):
            target=route[0]-num_of_rows
        elif(route[1]=='S'):
            target=route[0]+num_of_rows
        elif(route[1]=='W'):
            target=route[0]-1
        elif(route[1]=='E'):
            target=route[0]+1
    if target in ret:
        ret.remove(target)
    return ret

def manhattanDis(source,destination,num_of_rows):
    source_row=(int)(source/num_of_rows)
    source_col=source%num_of_rows
    destination_row=(int)(destination/num_of_rows)
    destination_col=destination%num_of_rows

    return abs(source_row-destination_row)+abs(source_col-destination_col)

def delExtraRoute(route,des):
    flag=False
    for i in range(0,len(route)):
        if(route[i][0]==des):
            del(route[i:len(route)])
            flag=True
            break
    return flag


def computeSplit2Route(source,destination,num_of_rows,link_set,start_time,sub_msg_size):
    source_row=(int)(source/num_of_rows)
    source_col=source%num_of_rows
    destination_row=(int)(destination/num_of_rows)
    destination_col=destination%num_of_rows
    sub1_route=[]
    sub1_contention=0
    sub2_route=[]
    sub2_contention=0
    tmp_link_set=[]
    if(source_row==destination_row or source_col==destination_col):#在同一行或同一列，sub1直接沿X/Y轴传输过去
        sub1_route=computeMsgRoute(source,destination,num_of_rows,link_set,start_time,sub_msg_size[0])
        sub1_contention=computeContention(sub1_route,link_set,num_of_rows,start_time,sub_msg_size[0])
        tmp_link_set=copy.deepcopy(link_set)
        Update_link_set(sub1_route,tmp_link_set,num_of_rows,start_time+sub1_contention,start_time+sub_msg_size[0]+sub1_contention)
        
    if(source_row==destination_row):#在同一行，sub2往上或往下传输一跳，再传往目的地
        if(source_row==0 or source_row==(num_of_rows-1)):#sub2只有一个方向可以选
            new_source = source+num_of_rows if source_row==0 else source-num_of_rows
            new_destination = destination+num_of_rows if destination_row==0 else destination-num_of_rows
            sub2_start_route=[source,'S'] if source_row==0 else [source,'N']
            sub2_end_route=[destination+num_of_rows,'N'] if destination_row==0 else [destination-num_of_rows,'S']
            sub2_contention,sub2_route=computeNewRoute(new_source,new_destination,sub2_start_route,sub2_end_route,num_of_rows,tmp_link_set,start_time,sub_msg_size[1])
        else:#sub2有两个方向可以选，取contention最小的那个方向
            #sub2往S传输一跳
            sub2_contention_1,sub2_route_1=computeNewRoute(source+num_of_rows,destination+num_of_rows,[source,'S'],[destination+num_of_rows,'N'],num_of_rows,tmp_link_set,start_time,sub_msg_size[1])
            #sub2往N传输一跳
            sub2_contention_2,sub2_route_2=computeNewRoute(source-num_of_rows,destination-num_of_rows,[source,'N'],[destination-num_of_rows,'S'],num_of_rows,tmp_link_set,start_time,sub_msg_size[1])
            sub2_route=sub2_route_1 if sub2_contention_1<=sub2_contention_2 else sub2_route_2
            sub2_contention=sub2_contention_1 if sub2_contention_1<=sub2_contention_2 else sub2_contention_2
    elif (source_col==destination_col):#在同一列，sub2往左或往右传输一跳，再传往目的地
        if(source_col==0 or source_col==(num_of_rows-1)):#sub2只有一个方向可以选
            new_source = source+1 if source_col==0 else source-1
            new_destination = destination+1 if destination_col==0 else destination-1
            sub2_start_route=[source,'E'] if source_col==0 else [source,'W']
            sub2_end_route=[destination+1,'W'] if destination_col==0 else [destination-1,'E']
            sub2_contention,sub2_route=computeNewRoute(new_source,new_destination,sub2_start_route,sub2_end_route,num_of_rows,tmp_link_set,start_time,sub_msg_size[1])
        else:#sub2有两个方向可以选，取contention最小的那个方向
            #sub2往E传输一跳
            sub2_contention_1,sub2_route_1=computeNewRoute(source+1,destination+1,[source,'E'],[destination+1,'W'],num_of_rows,tmp_link_set,start_time,sub_msg_size[1])
                #sub2往W传输一跳
            sub2_contention_2,sub2_route_2=computeNewRoute(source-1,destination-1,[source,'W'],[destination-1,'E'],num_of_rows,tmp_link_set,start_time,sub_msg_size[1])
            sub2_route=sub2_route_1 if sub2_contention_1<=sub2_contention_2 else sub2_route_2
            sub2_contention=sub2_contention_1 if sub2_contention_1<=sub2_contention_2 else sub2_contention_2
    else:#既不在同一行也不在同一列，选择曼哈顿距离最近的port绑定，曼哈顿距离最近的情况下有两种绑定方案，选择最优的那一种
        #选出source和destination曼哈顿距离最近的port
        new_source_1 = source+1 if destination_col>source_col else source-1
        new_destination_1 = destination-1 if destination_col>source_col else destination+1
        new_source_2= source+num_of_rows if destination_row>source_row else source-num_of_rows
        new_destination_2= destination-num_of_rows if destination_row>source_row else destination+num_of_rows

        #solution 1，绑定new_source_1和new_destination_1
        sol1_sub1_contention,sol1_sub1_route=computeNewRoute(new_source_1,new_destination_1,generateStartEndRoute(source,new_source_1,num_of_rows),generateStartEndRoute(new_destination_1,destination,num_of_rows),num_of_rows,link_set,start_time,sub_msg_size[0])
        tmp_link_set=copy.deepcopy(link_set)
        Update_link_set(sol1_sub1_route,tmp_link_set,num_of_rows,start_time+sol1_sub1_contention,start_time+sub_msg_size[0]+sol1_sub1_contention)
        sol1_sub2_contention,sol1_sub2_route=computeNewRoute(new_source_2,new_destination_2,generateStartEndRoute(source,new_source_2,num_of_rows),generateStartEndRoute(new_destination_2,destination,num_of_rows),num_of_rows,tmp_link_set,start_time,sub_msg_size[1])

        #solution 2，绑定new_source_1和new_destination_2
        sol2_sub1_contention,sol2_sub1_route=computeNewRoute(new_source_1,new_destination_2,generateStartEndRoute(source,new_source_1,num_of_rows),generateStartEndRoute(new_destination_2,destination,num_of_rows),num_of_rows,link_set,start_time,sub_msg_size[0])
        tmp_link_set=copy.deepcopy(link_set)
        Update_link_set(sol2_sub1_route,tmp_link_set,num_of_rows,start_time+sol2_sub1_contention,start_time+sub_msg_size[0]+sol2_sub1_contention)
        sol2_sub2_contention,sol2_sub2_route=computeNewRoute(new_source_2,new_destination_1,generateStartEndRoute(source,new_source_2,num_of_rows),generateStartEndRoute(new_destination_1,destination,num_of_rows),num_of_rows,tmp_link_set,start_time,sub_msg_size[1])

        #选择contention最小的方案
        sol1_contention = sol1_sub1_contention if sol1_sub1_contention>=sol1_sub2_contention else sol1_sub2_contention
        sol2_contention = sol2_sub1_contention if sol2_sub1_contention>=sol2_sub2_contention else sol2_sub2_contention
            
        sub1_route=sol1_sub1_route if sol1_contention<=sol2_contention else sol2_sub1_route
        sub1_contention=sol1_sub1_contention if sol1_contention<=sol2_contention else sol2_sub1_contention
        sub2_route=sol1_sub2_route if sol1_contention<=sol2_contention else sol2_sub2_route
        sub2_contention=sol1_sub2_contention if sol1_contention<=sol2_contention else sol2_sub2_contention
        
    #contention_times=sub1_contention if sub1_contention>=sub2_contention else sub2_contention
    flag1=delExtraRoute(sub1_route,destination)
    flag2=delExtraRoute(sub2_route,destination)
    if(flag1):
        print("info:delete extra route in sub1")
        sub1_contention=computeContention(sub1_route,link_set,num_of_rows,start_time,sub_msg_size[0])
    if(flag2):
        print("info:delete extra route in sub2")
        tmp_link_set=copy.deepcopy(link_set)
        Update_link_set(sub1_route,tmp_link_set,num_of_rows,start_time+sub1_contention,start_time+sub_msg_size[0]+sub1_contention)
        sub2_contention=computeContention(sub2_route,tmp_link_set,num_of_rows,start_time,sub_msg_size[1])
    return [sub1_contention,sub2_contention],[sub1_route,sub2_route]

def computeSplit3Route(source,destination,num_of_rows,link_set,start_time,sub_msg_size):#src和des均至少有三个port可用
    source_row=(int)(source/num_of_rows)
    source_col=source%num_of_rows
    destination_row=(int)(destination/num_of_rows)
    destination_col=destination%num_of_rows
    sub1_route=[]
    sub1_contention=0
    sub2_route=[]
    sub2_contention=0
    sub3_route=[]
    sub3_contention=999999999
    tmp_link_set=[]

    if(source_row==destination_row or source_col==destination_col):#在同一行或同一列，sub1直接沿X/Y轴传输过去
        sub1_route=computeMsgRoute(source,destination,num_of_rows,link_set,start_time,sub_msg_size[0])
        sub1_contention=computeContention(sub1_route,link_set,num_of_rows,start_time,sub_msg_size[0])
        tmp_link_set=copy.deepcopy(link_set)
        Update_link_set(sub1_route,tmp_link_set,num_of_rows,start_time+sub1_contention,start_time+sub_msg_size[0]+sub1_contention)
        new_src2=0
        new_des2=0
        new_src3=0
        new_des3=0
        if(source_row==destination_row):#在同一行
            new_src2=source-num_of_rows
            new_des2=destination-num_of_rows
            new_src3=source+num_of_rows
            new_des3=destination+num_of_rows
        else:#在同一列
            new_src2=source+1
            new_des2=destination+1
            new_src3=source-1
            new_des3=destination-1
        sub2_contention,sub2_route=computeNewRoute(new_src2,new_des2,generateStartEndRoute(source,new_src2,num_of_rows),generateStartEndRoute(new_des2,destination,num_of_rows),num_of_rows,tmp_link_set,start_time,sub_msg_size[1])
        Update_link_set(sub2_route,tmp_link_set,num_of_rows,start_time+sub2_contention,start_time+sub_msg_size[1]+sub2_contention)
        sub3_contention,sub3_route=computeNewRoute(new_src3,new_des3,generateStartEndRoute(source,new_src3,num_of_rows),generateStartEndRoute(new_des3,destination,num_of_rows),num_of_rows,tmp_link_set,start_time,sub_msg_size[2])

        flag1=delExtraRoute(sub1_route,destination)
        flag2=delExtraRoute(sub2_route,destination)
        flag3=delExtraRoute(sub3_route,destination)
        if(flag1):
            print("info:delete extra route in sub1")
            sub1_contention=computeContention(sub1_route,link_set,num_of_rows,start_time,sub_msg_size[0])
        if(flag2):
            print("info:delete extra route in sub2")
            tmp_link_set=copy.deepcopy(link_set)
            Update_link_set(sub1_route,tmp_link_set,num_of_rows,start_time+sub1_contention,start_time+sub_msg_size[0]+sub1_contention)
            sub2_contention=computeContention(sub2_route,tmp_link_set,num_of_rows,start_time,sub_msg_size[1])
        if(flag3):
            print("info:delete extra route in sub2")
            tmp_link_set=copy.deepcopy(link_set)
            Update_link_set(sub1_route,tmp_link_set,num_of_rows,start_time+sub1_contention,start_time+sub_msg_size[0]+sub1_contention)
            Update_link_set(sub2_route,tmp_link_set,num_of_rows,start_time+sub2_contention,start_time+sub_msg_size[1]+sub2_contention)
            sub3_contention=computeContention(sub3_route,tmp_link_set,num_of_rows,start_time,sub_msg_size[2])
        
        return [sub1_contention,sub2_contention,sub3_contention],[sub1_route,sub2_route,sub3_route]
    else:#既不在同一行也不在同一列
        split2contention,split2route=computeSplit2Route(source,destination,num_of_rows,link_set,start_time,sub_msg_size[0:2])
        #print("info:",split2route)
        sub1_route=split2route[0]
        sub1_contention=split2contention[0]
        sub2_route=split2route[1]
        sub2_contention=split2contention[1]
        tmp_link_set=copy.deepcopy(link_set)
        Update_link_set(sub1_route,tmp_link_set,num_of_rows,start_time+split2contention[0],start_time+sub_msg_size[0]+split2contention[0])
        Update_link_set(sub2_route,tmp_link_set,num_of_rows,start_time+split2contention[1],start_time+sub_msg_size[1]+split2contention[1])
        #选出在out layer的两个port
        src_ports=initPorts(source,num_of_rows,destination,mode=0)
        des_ports=initPorts(destination,num_of_rows,source,mode=1)
        for i in split2route:
            #print("info:",i[0],i[-1])
            src_ports=delPorts(src_ports,i[0],num_of_rows,mode=0)
            des_ports=delPorts(des_ports,i[-1],num_of_rows,mode=1)
        solution=[]#item -> ('new_src,new_des',manDis)
        for i in src_ports:
            for j in des_ports:
                solution.append(( str(i)+','+str(j),manhattanDis(i,j,num_of_rows) ))
        solution.sort(key=lambda x: x[1])
        for i in solution:
            #if(i[1]!=solution[0][1]):#只取曼哈顿距离最小的solution
                #break
            new_src=int(i[0].split(',')[0])
            new_des=int(i[0].split(',')[1])
            tmp_sub3_contention,tmp_sub3_route=computeNewRoute(new_src,new_des,generateStartEndRoute(source,new_src,num_of_rows),generateStartEndRoute(new_des,destination,num_of_rows),num_of_rows,tmp_link_set,start_time,sub_msg_size[2])
            if(tmp_sub3_contention<sub3_contention):
                sub3_route=copy.deepcopy(tmp_sub3_route)
                sub3_contention=tmp_sub3_contention
        
        flag1=delExtraRoute(sub1_route,destination)
        flag2=delExtraRoute(sub2_route,destination)
        flag3=delExtraRoute(sub3_route,destination)
        if(flag1):
            print("info:delete extra route in sub1")
            sub1_contention=computeContention(sub1_route,link_set,num_of_rows,start_time,sub_msg_size[0])
        if(flag2):
            print("info:delete extra route in sub2")
            tmp_link_set=copy.deepcopy(link_set)
            Update_link_set(sub1_route,tmp_link_set,num_of_rows,start_time+sub1_contention,start_time+sub_msg_size[0]+sub1_contention)
            sub2_contention=computeContention(sub2_route,tmp_link_set,num_of_rows,start_time,sub_msg_size[1])
        if(flag3):
            print("info:delete extra route in sub2")
            tmp_link_set=copy.deepcopy(link_set)
            Update_link_set(sub1_route,tmp_link_set,num_of_rows,start_time+sub1_contention,start_time+sub_msg_size[0]+sub1_contention)
            Update_link_set(sub2_route,tmp_link_set,num_of_rows,start_time+sub2_contention,start_time+sub_msg_size[1]+sub2_contention)
            sub3_contention=computeContention(sub3_route,tmp_link_set,num_of_rows,start_time,sub_msg_size[2])
        return [sub1_contention,sub2_contention,sub3_contention],[sub1_route,sub2_route,sub3_route]

def computeSplit4Route(source,destination,num_of_rows,link_set,start_time,sub_msg_size):#src和des均至少有四个port可用
    split3contention,split3route=computeSplit3Route(source,destination,num_of_rows,link_set,start_time,sub_msg_size[0:3])
    tmp_link_set=copy.deepcopy(link_set)
    for i in range(0,3):
        Update_link_set(split3route[i],tmp_link_set,num_of_rows,start_time+split3contention[i],start_time+sub_msg_size[i]+split3contention[i])
    src_ports=initPorts(source,num_of_rows,destination,mode=0)
    des_ports=initPorts(destination,num_of_rows,source,mode=1)
    for i in split3route:
        #print("info:",i)
        src_ports=delPorts(src_ports,i[0],num_of_rows,mode=0)
        des_ports=delPorts(des_ports,i[-1],num_of_rows,mode=1)
    new_src=src_ports[0]
    new_des=des_ports[0]
    sub4_contention,sub4_route=computeNewRoute(new_src,new_des,generateStartEndRoute(source,new_src,num_of_rows),generateStartEndRoute(new_des,destination,num_of_rows),num_of_rows,tmp_link_set,start_time,sub_msg_size[3])
    
    split3contention.append(sub4_contention)
    split3route.append(sub4_route)

    for i in range(0,len(split3route)):
        flag=delExtraRoute(split3route[i],destination)
        if(flag):
            print("info:delete extra route in sub",i+1)
            tmp_link_set=copy.deepcopy(link_set)
            for j in range(0,i):
                Update_link_set(split3route[j],tmp_link_set,num_of_rows,start_time+split3contention[j],start_time+sub_msg_size[j]+split3contention[j])
            split3contention[i]=computeContention(split3route[i],tmp_link_set,num_of_rows,start_time,sub_msg_size[i])
            
    return split3contention,split3route

        



            
def splitRoute(source,destination,num_of_rows,link_set,split_number,start_time,msg_size):
    if(source==destination):
        return [0],[[]],[msg_size]

    src_ports=initPorts(source,num_of_rows,destination,mode=0)
    des_ports=initPorts(destination,num_of_rows,source,mode=1)

    source_row=(int)(source/num_of_rows)
    source_col=source%num_of_rows
    destination_row=(int)(destination/num_of_rows)
    destination_col=destination%num_of_rows

    if(source_row==destination_row or source_col==destination_col):#src和des在同一行/列，不考虑分4组的情况
        if(split_number==4):
            split_number=3

    if(len(src_ports)<split_number or len(des_ports)<split_number):
        split_number=len(src_ports) if len(src_ports)<len(des_ports) else len(des_ports)
    #print("split number is",split_number)

    sub_msg_size=[]
    tmp_msg_size=msg_size
    if(tmp_msg_size<split_number):
        split_number=tmp_msg_size
        
    while(tmp_msg_size>0 and len(sub_msg_size)<split_number):
        sub_msg_size.append((int)(msg_size/split_number))
        tmp_msg_size-=(int)(msg_size/split_number)
    sub_msg_size[-1]+=tmp_msg_size
    #print(sub_msg_size)
    
    if(split_number==1):
        ret_route=computeMsgRoute(source,destination,num_of_rows,link_set,start_time,msg_size)
        contention_times=computeContention(ret_route,link_set,num_of_rows,start_time,msg_size)
        return [contention_times],[ret_route],sub_msg_size
    elif(split_number==2):
        contention,route=computeSplit2Route(source,destination,num_of_rows,link_set,start_time,sub_msg_size)
        return contention,route,sub_msg_size
    elif(split_number==3):
        contention,route=computeSplit3Route(source,destination,num_of_rows,link_set,start_time,sub_msg_size)
        return contention,route,sub_msg_size
    elif(split_number==4):
        contention,route=computeSplit4Route(source,destination,num_of_rows,link_set,start_time,sub_msg_size)
        return contention,route,sub_msg_size
    





                



 






if __name__ == '__main__':

    hyperperiod,num_of_tasks,edges,comp_cost=init('./task graph/N4_test.tgff')
    adj_matrix,total_needSend,total_needReceive,execution=Get_detailed_data(num_of_tasks,edges,comp_cost)
    #print(adj_matrix)
    num_of_rows=4
    MapResult=[-1,5,11,2,15]

    contention,task_graph=improved_routeCompute(adj_matrix,num_of_tasks,execution,num_of_rows,MapResult)
    print(contention)
    print(task_graph)






