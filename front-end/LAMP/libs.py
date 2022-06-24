import re
import numpy as np
import json

import copy


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical



def init(filename):
    f = open(filename, 'r')

    #Get hyperperiod
    hyperperiod=int(f.readline().split()[1])
    #print(hyperperiod)
    
    f.readline()
    f.readline()
    f.readline()
    f.readline()

    # Calculate the amount of tasks
    num_of_tasks = 0
    while f.readline().startswith('\tTASK'):
        num_of_tasks += 1
    #print('Number of tasks =',num_of_tasks)

    # Build a communication matrix
    data = [[-1 for i in range(num_of_tasks)] for i in range(num_of_tasks)]
    line = f.readline()
    while line.startswith('\tARC'):
        line = re.sub(r'\bt\d_', '', line)
        i, j, d = [int(s) for s in line.split() if s.isdigit()]
        data[i][j] = d
        line = f.readline()
    #for line in data:
    #    print(line)

    while not f.readline().startswith('# type'):
        pass


    # Build a computation matrix
    comp_cost = {}
    line = f.readline()
    while line.startswith('\t') or line.startswith('    '):
        comp_cost.update({int(line.split()[0]):int(line.split()[1])})
        line = f.readline()
    #for key in comp_cost.keys():
    #    print(key,comp_cost[key],type(key),type(comp_cost[key]))

    

    return [hyperperiod, num_of_tasks, data, comp_cost]


def Get_Neighborhood(position,radius,M,N): #return a list which consists of positions around input position with radius=r
    row=int(position/N)
    col=position%N
    neighborhood=[]
    for i in range(row-radius,row+radius+1):
        if(i>=0 and i<M):
            for j in range(col-radius,col+radius+1):
                if(j>=0 and (i!=row or j!=col) and j<N):
                    neighborhood.append(i*N+j)
    return neighborhood

def Get_mapping_exe_time(PEs_task_current_solution,Tasks_position_current_solution,computation_ability,num_of_rows,execution):#传进来的computation_ability是CVB_method生成的矩阵
    num_of_tasks=len(execution)-1
    ret_execution=copy.deepcopy(execution)
    for i in Tasks_position_current_solution.keys():#首先计算是否有task被匹配到同一个PE，有的话添加惩罚，但是如果是这同一个PE上有任务图中同一条边上的两个task，意思就是它们一定不同时运行，这个时候不计算惩罚
        position=Tasks_position_current_solution[i]
        if(len(PEs_task_current_solution[position])>1):
            ret_execution[i+1]=ret_execution[i+1]*len(PEs_task_current_solution[position])

    #然后根据运行能力倍减运行时间
    for i in Tasks_position_current_solution.keys():
        position=Tasks_position_current_solution[i]
        #ret_execution[i+1]=int(ret_execution[i+1]/computation_ability[int(position/num_of_rows)][position%num_of_rows])
        ret_execution[i+1]=computation_ability[i][position]

    ret=0
    for i in ret_execution:
        ret+=i
    return ret

def Get_detailed_data(num_of_tasks,edges,comp_cost):#输出邻接矩阵,total_needSend,total_needReceive,execution
    adj_matrix=np.zeros((num_of_tasks+1,num_of_tasks+1),dtype=np.int)#task从1开始算
    for i in range(0,len(edges)):
        for j in range(0,len(edges[i])):
            if(edges[i][j]!=-1):
                adj_matrix[i+1][j+1]=comp_cost[edges[i][j]]#adj_matrix[i][j]不为0，表示task i有到task j的出边，数组的值为待传输的量

    total_needSend=np.zeros(num_of_tasks+1,dtype=np.int)#task从1开始算的
    total_needReceive=np.zeros(num_of_tasks+1,dtype=np.int)#task从1开始算的
    for i in range(1,num_of_tasks+1):
        task_i_needSend=0
        for j in range(1,num_of_tasks+1):
            task_i_needSend+=adj_matrix[i][j]
        total_needSend[i]=task_i_needSend

    for j in range(1,num_of_tasks+1):
        task_j_needReceive=0
        for i in range(1,num_of_tasks+1):
            task_j_needReceive+=adj_matrix[i][j]
        total_needReceive[j]=task_j_needReceive

    execution=np.zeros(num_of_tasks+1,dtype=np.int)#task从1开始算的
    for i in range(1,num_of_tasks+1):
        execution[i]=comp_cost[i-1]
    
    return adj_matrix,total_needSend,total_needReceive,execution

def find_start_task(adj_matrix,num_of_tasks):#寻找入度为0的点
    ret=[]
    in_degree=np.zeros(num_of_tasks+1,dtype=np.int)
    for i in range(1,num_of_tasks+1):
        for j in range(1,num_of_tasks+1):
            if(adj_matrix[i][j]!=0):
                in_degree[j]+=1
    for i in range(1,num_of_tasks+1):
        if(in_degree[i]==0):
            ret.append(i)
    return ret

def get_sorted_dict(dict):#将task_graph按照task的序号排序，再传入online_compute
    ret={}
    l=[]
    for i in dict.keys():
        l.append(int(i))
    l.sort()
    for i in l:
        ret.update({str(i):dict[str(i)]})
    return ret
    


def Get_full_route_by_XY(part_route,source_position,dest_position,num_of_rows):
    ret=copy.deepcopy(part_route)
    dest_row=int(dest_position/num_of_rows)
    dest_col=dest_position%num_of_rows
    cur_row=-1
    cur_col=-1

    if(len(part_route)==0):
        cur_row=int(source_position/num_of_rows)
        cur_col=source_position%num_of_rows
    else:#计算出数据现在走到了哪个位置
        cur_position=part_route[-1][0]
        cur_row=int(cur_position/num_of_rows)
        cur_col=cur_position%num_of_rows
        if(part_route[-1][1]=='N'):
            cur_row-=1
        elif(part_route[-1][1]=='S'):
            cur_row+=1
        elif(part_route[-1][1]=='W'):
            cur_col-=1
        elif(part_route[-1][1]=='E'):
            cur_col+=1

    while(cur_col<dest_col):
        tmp=[]
        tmp.append(cur_row*num_of_rows+cur_col)
        tmp.append('E')
        ret.append(tmp)
        cur_col+=1
    while(cur_col>dest_col):
        tmp=[]
        tmp.append(cur_row*num_of_rows+cur_col)
        tmp.append('W')
        ret.append(tmp)
        cur_col-=1

    while(cur_row<dest_row):
        tmp=[]
        tmp.append(cur_row*num_of_rows+cur_col)
        tmp.append('S')
        ret.append(tmp)
        cur_row+=1
    while(cur_row>dest_row):
        tmp=[]
        tmp.append(cur_row*num_of_rows+cur_col)
        tmp.append('N')
        ret.append(tmp)
        cur_row-=1
    
    
    return ret
    
def Get_reward_by_pendTimes(pendTimes):
    return 0-pendTimes

"""
#state为[state_tensor,cur_position,partRouteFromRL]，传进来的partRoute的格式是直接的路由表，没有第一位第二位的task
def check_if_Done(state,source_position,dest_position,num_of_rows,task_graph,fullRouteFromRL,task_source,task_dest,MapResult):#检查当前的state是否已经结束，结束了的话直接把end_state,reward,done=True返回
    next_state_tensor=state[0]
    next_position=-1
    next_partRoute=state[2]

    cur_row=int(state[1]/num_of_rows)
    cur_col=state[1]%num_of_rows
    dest_row=int(dest_position/num_of_rows)
    dest_col=dest_position%num_of_rows

    flag=False

    if(cur_row==dest_row or cur_col==dest_col):#结束，先更新tensor
        flag=True
        if(cur_row==dest_row and cur_col==dest_col):#考虑到一开始两个task就被map到同一个PE的情况
            next_state_tensor=state[0]
        elif(cur_row==dest_row):
            if(cur_col<dest_col):#向East走
                for i in range(cur_col,dest_col):#更新tensor
                    next_state_tensor[0][3][cur_row*num_of_rows+i]=1
            else:#向West走
                for i in range(cur_col,dest_col,-1):#更新tensor
                    next_state_tensor[0][2][cur_row*num_of_rows+i]=1
        elif(cur_col==dest_col):
            if(cur_row<dest_row):#向South走
                for i in range(cur_row,dest_row):#更新tensor
                    next_state_tensor[0][1][i*num_of_rows+cur_col]=1
            else:#向North走
                for i in range(cur_row,dest_row,-1):
                    next_state_tensor[0][0][i*num_of_rows+cur_col]=1
    
    if(flag==False):#没有结束
        return [],0,False
    else:#结束
        #更新position
        next_position=dest_position
        #更新partRoute，此时的partRoute就是这一条链路全部的路由表，可以直接传进onlineCompute
        next_partRoute=Get_full_route_by_XY(state[2],source_position,dest_position,num_of_rows)
        #处理参数，传进onlineCompute计算pending次数
        #首先更新taskgraph里的这条链路的路由表
        for i in range(0,len(task_graph[str(task_source)]['out_links'])):
            if(int(task_graph[str(task_source)]['out_links'][i][0])==task_dest):
                task_graph[str(task_source)]['out_links'][i][2]=next_partRoute
        #处理partRoute
        partRoute_to_onlineCompute=[]
        partRoute_to_onlineCompute.append(task_source)
        partRoute_to_onlineCompute.append(task_dest)
        for i in next_partRoute:
            partRoute_to_onlineCompute.append(i)
        task=onlineTimeline("",num_of_rows)
        #print("C_fullRoute:",fullRouteFromRL)
        #print("C_partRoute:",partRoute_to_onlineCompute)
        #print("C_computing:",task_graph)
        task.loadGraphByDict(task_graph,MapResult,fullRouteFromRL,partRoute_to_onlineCompute,len(MapResult)-1)
        pendTimes=task.computeTime()
        #print("C_pendTimes",pendTimes)
        
        for j in task_graph.keys():
            for k in range(0,len(task_graph[j]['out_links'])):
                task_graph[j]['out_links'][k]=task_graph[j]['out_links'][k][0:6]
        
        #根据pendTimes计算reward
        return [next_state_tensor,next_position,next_partRoute],Get_reward_by_pendTimes(pendTimes),True


#state为[state_tensor,cur_position,partRouteFromRL]，传进来的partRoute的格式是直接的路由表，没有第一位第二位的task
def Environment(state,action,source_position,dest_position,num_of_rows,task_graph,fullRouteFromRL,task_source,task_dest,MapResult):#用于获得next_state，reward，done，除了state和action，剩下的参数都是为了传进onlineCompute
    next_state_tensor=torch.Tensor(np.zeros((1,4,num_of_rows*num_of_rows),dtype=np.int))
    next_state_tensor.copy_(state[0])
    next_position=-1
    next_partRoute=copy.deepcopy(state[2])

    #执行action前
    cur_row=int(state[1]/num_of_rows)
    cur_col=state[1]%num_of_rows
    dest_row=int(dest_position/num_of_rows)
    dest_col=dest_position%num_of_rows
    
    #RL学习之前check一次，就能确保起码能走一步

    #开始执行action
    #state_tensor的四个channel,从0-3以此为N,S,W,E
    if(action==0):#沿x轴走
        if(cur_col<dest_col):
            next_state_tensor[0][3][cur_row*num_of_rows+cur_col]=1#更新tensor
            next_partRoute.append([cur_row*num_of_rows+cur_col,'E'])#更新路由表
            cur_col+=1#向East走了一步
        elif(cur_col>dest_col):
            next_state_tensor[0][2][cur_row*num_of_rows+cur_col]=1#更新tensor
            next_partRoute.append([cur_row*num_of_rows+cur_col,'W'])#更新路由表
            cur_col-=1#向West走了一步
    elif(action==1):#沿y轴走
        if(cur_row<dest_row):
            next_state_tensor[0][1][cur_row*num_of_rows+cur_col]=1#更新tensor
            next_partRoute.append([cur_row*num_of_rows+cur_col,'S'])#更新路由表
            cur_row+=1#向South走了一步
        elif(cur_row>dest_row):
            next_state_tensor[0][0][cur_row*num_of_rows+cur_col]=1#更新tensor
            next_partRoute.append([cur_row*num_of_rows+cur_col,'N'])#更新路由表
            cur_row-=1#向North走了一步
    
    next_position=cur_row*num_of_rows+cur_col

    ret_state,ret_reward,done=check_if_Done([next_state_tensor,next_position,next_partRoute],source_position,dest_position,num_of_rows,task_graph,fullRouteFromRL,task_source,task_dest,MapResult)

    if(done==True):
        return ret_state,ret_reward,done
    else:#没有结束，需要计算reward
        #根据XY-routing补全路由表，然后传进onlineCompute
        fullRouteByXY=Get_full_route_by_XY(next_partRoute,source_position,dest_position,num_of_rows)
        #处理参数，传进onlineCompute计算pending次数
        #首先更新taskgraph里的这条链路的路由表
        for i in range(0,len(task_graph[str(task_source)]['out_links'])):
            if(int(task_graph[str(task_source)]['out_links'][i][0])==task_dest):
                task_graph[str(task_source)]['out_links'][i][2]=fullRouteByXY
        #处理partRoute
        partRoute_to_onlineCompute=[]
        partRoute_to_onlineCompute.append(task_source)
        partRoute_to_onlineCompute.append(task_dest)
        for i in next_partRoute:
            partRoute_to_onlineCompute.append(i)
        task=onlineTimeline("",num_of_rows)
        #print("fullRoute:",fullRouteFromRL)
        #print("partRoute:",partRoute_to_onlineCompute)
        #print("computing:",task_graph)
        task.loadGraphByDict(task_graph,MapResult,fullRouteFromRL,partRoute_to_onlineCompute,len(MapResult)-1)
        pendTimes=task.computeTime()
        #print("pendTimes",pendTimes)
        
        for j in task_graph.keys():
            for k in range(0,len(task_graph[j]['out_links'])):
                task_graph[j]['out_links'][k]=task_graph[j]['out_links'][k][0:6]
        
        #根据pendTimes计算reward
        return [next_state_tensor,next_position,next_partRoute],Get_reward_by_pendTimes(pendTimes),False
"""


class ActorCritic(nn.Module):#输入：channel*length,4是channel,N是PE 输出：1*action_space和1*1
    def __init__(self, input_channel,input_length, action_space):#input_length是PE的个数
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(#输入为1*input_channel*input_length
            #nn.Linear(num_inputs, hidden_size),
            #nn.ReLU(),
            #nn.Linear(hidden_size, 1)#1*1
            nn.Conv1d(input_channel,1,input_length,1)#输出为 1*1*1
        )
        
        self.actor = nn.Sequential(
            #nn.Linear(num_inputs, hidden_size),
            #nn.ReLU(),
            #nn.Linear(hidden_size, action_space),#1*action_space
            #nn.Softmax(dim=1),
            nn.Conv1d(input_channel,1,(input_length-action_space+1),1),
            nn.Softmax(dim=2)#输出为1*1*action_space
        )
        
    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)#.sample()得到的是1*1的二维矩阵tensor，例如[[1]]，再.numpy后可以得到正常的矩阵
        return dist, value

# Actor module, categorical actions only
class Actor(nn.Module):
    def __init__(self, input_channel, input_length, action_space):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(input_channel,1,(input_length-action_space+1),1),
            nn.Softmax(dim=2)#输出为1*1*action_space
        )
    
    def forward(self, X):
        return self.model(X).squeeze(0).squeeze(0)

# Critic module
class Critic(nn.Module):
    def __init__(self, input_channel, input_length):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(input_channel,1,input_length,1)#输出为 1*1*1
        )
    
    def forward(self, X):
        return self.model(X).squeeze(0).squeeze(0)

def Get_rand_computation_ability(num_of_rows):
    ret=np.random.randint(1,5,(num_of_rows,num_of_rows))
    return ret

def Get_rand_computation_ability2(num_of_rows):
    ret=np.full((num_of_rows,num_of_rows),0.5)
    for i in range(0,num_of_rows*num_of_rows):
        random_choose=np.random.randint(0,num_of_rows*num_of_rows)
        ret[int(random_choose/num_of_rows)][int(random_choose%num_of_rows)]+=0.5
    return ret



#improved版新添加的函数
#检查在used_link中，给定的时间区间是否有冲突
#返回是否有争用，哪一条link有争用，这条link在哪个时间段发生了争用
def Check_contention(used_link,link_set,start_time,end_time):
    for i in used_link:
        for j in link_set[i].timeline:
            if(end_time>j[0] and end_time<j[1]):
                return False,i,link_set[i].timeline.index(j)
            elif(start_time>j[0] and start_time<j[1]):
                return False,i,link_set[i].timeline.index(j)
            elif(start_time<j[0] and end_time>j[1]):
                return False,i,link_set[i].timeline.index(j)
            elif(start_time == j[0]):#目标区间起点与当前检测的区间起点相同，冲突
                return False,i,link_set[i].timeline.index(j)
            elif(end_time == j[1]):#目标区间终点与当前检测的区间终点相同，冲突
                return False,i,link_set[i].timeline.index(j)
    
    return True,-1,-1

def Get_link_index_by_route(route,num_of_rows):#根据route（如[6,"S"]）返回link的编号
    tmp_row=int(route[0]/num_of_rows)
    tmp_col=route[0]%num_of_rows
    if(route[1]=='N'):
        tmp_row-=1
        return (2*num_of_rows-1)*tmp_row+(num_of_rows-1)+tmp_col
    elif(route[1]=='S'):
        return (2*num_of_rows-1)*tmp_row+(num_of_rows-1)+tmp_col
    elif(route[1]=='W'):
        return (2*num_of_rows-1)*tmp_row+(tmp_col-1)
    elif(route[1]=='E'):
        return (2*num_of_rows-1)*tmp_row+tmp_col

        
def computeContention(partRoute,link_set,num_of_rows,start_time,transmission):
    used_link=[]
    for i in partRoute:
        used_link.append(Get_link_index_by_route(i,num_of_rows))
    contentious_link=[]
    contentious_timeline_index=[]
    end_time=start_time+transmission
    for i in used_link:
        for j in link_set[i].timeline:#检查这条link的时间轴
            if(end_time>j[0] and end_time<j[1]):#目标区间右侧落在了当前检测到的区间里，冲突
                contentious_link.append(i)
                contentious_timeline_index.append(link_set[i].timeline.index(j))
            elif(start_time>j[0] and start_time<j[1]):#目标区间左侧落在了当前检测到的区间里，冲突
                contentious_link.append(i)
                contentious_timeline_index.append(link_set[i].timeline.index(j))
            elif(start_time<j[0] and end_time>j[1]):#目标区间包括了当前检测到的区间，冲突
                contentious_link.append(i)
                contentious_timeline_index.append(link_set[i].timeline.index(j))
            elif(start_time == j[0]):#目标区间起点与当前检测的区间起点相同，冲突
                contentious_link.append(i)
                contentious_timeline_index.append(link_set[i].timeline.index(j))
            elif(end_time == j[1]):#目标区间终点与当前检测的区间终点相同，冲突
                contentious_link.append(i)
                contentious_timeline_index.append(link_set[i].timeline.index(j))
    
    if(len(contentious_link)==0):#这个route没有任何争用
        return 0
    else:#这个route有争用，需要计算参数T
        T_=0
        max_end_time=0
        for i in range(0,len(contentious_link)):#寻找争用过的link的最大结束时间
            if(link_set[ contentious_link[i] ].timeline[ contentious_timeline_index[i] ][1] > max_end_time):
                max_end_time=link_set[ contentious_link[i] ].timeline[ contentious_timeline_index[i] ][1]
        T_=max_end_time-start_time#设置T值的初始值，然后还需要检测这个T是否可以使得这条边传输
        flag_contention,link_index,timeline_index=Check_contention(used_link,link_set,start_time+T_,end_time+T_)
        while(flag_contention==False):#增加T后依然发生争用，还需要继续增大T
            T_=link_set[link_index].timeline[timeline_index][1]-start_time
            flag_contention,link_index,timeline_index=Check_contention(used_link,link_set,start_time+T_,end_time+T_)
        return T_

def Update_link_set(partRoute,link_set,num_of_rows,start_time,end_time):#在link_set上预约[start_time,end_time]的时间
    used_link=[]
    for i in partRoute:
        used_link.append(Get_link_index_by_route(i,num_of_rows))

    
    for i in used_link:#遍历要使用的link
        if(len(link_set[i].timeline)==0):#这条link的时间轴为空
            link_set[i].timeline.append([start_time,end_time])
        for j in range(0,len(link_set[i].timeline)):#遍历这条link的timeline
            if(end_time <= link_set[i].timeline[j][0]):#放在这个区间的左边
                link_set[i].timeline.insert(j,[start_time,end_time])
                break
            elif(start_time >= link_set[i].timeline[j][1]):#放在这个区间的右边
                if( j == (len(link_set[i].timeline)-1) ):#最后一位，append即可
                    link_set[i].timeline.append([start_time,end_time])
                else:
                    link_set[i].timeline.insert(j+1,[start_time,end_time])
                break


#这里reward返回的是-contention，需要再考虑一下
def Check_if_Done_improved(state,source_position,dest_position,link_set,num_of_rows,start_time,end_time):#state为[state_tensor,cur_position,partRoute],这里的partRoute是直接的路由表，如[[0,'S'],[4,'E']]
    next_state_tensor=torch.Tensor(np.zeros((1,4,num_of_rows*num_of_rows),dtype=np.int))
    next_state_tensor.copy_(state[0])
    cur_row=int(state[1]/num_of_rows)
    cur_col=state[1]%num_of_rows
    dest_row=int(dest_position/num_of_rows)
    dest_col=dest_position%num_of_rows

    if(cur_row==dest_row or cur_col==dest_col):#结束，需要使用XY routing补全路由，更新tensor，并且计算contention
        #更新tensor
        if(cur_row==dest_row and cur_col==dest_col):#考虑到一开始两个task就被map到同一个PE的情况
            pass
        elif(cur_row==dest_row):
            if(cur_col<dest_col):#向East走
                for i in range(cur_col,dest_col):#更新tensor
                    next_state_tensor[0][3][cur_row*num_of_rows+i]=1
            else:#向West走
                for i in range(cur_col,dest_col,-1):#更新tensor
                    next_state_tensor[0][2][cur_row*num_of_rows+i]=1
        elif(cur_col==dest_col):
            if(cur_row<dest_row):#向South走
                for i in range(cur_row,dest_row):#更新tensor
                    next_state_tensor[0][1][i*num_of_rows+cur_col]=1
            else:#向North走
                for i in range(cur_row,dest_row,-1):
                    next_state_tensor[0][0][i*num_of_rows+cur_col]=1
        #补全路由，计算contention
        full_Route=Get_full_route_by_XY(state[2],source_position,dest_position,num_of_rows)
        contention=computeContention(full_Route,link_set,num_of_rows,start_time,end_time-start_time)
        return [next_state_tensor,dest_position,full_Route],-1*contention,True
    else:
        return [],0,False

#这里reward返回的是-contention，需要再考虑一下
def Environment_improved(state,action,source_position,dest_position,link_set,num_of_rows,start_time,end_time):
    next_state_tensor=torch.Tensor(np.zeros((1,4,num_of_rows*num_of_rows),dtype=np.int))
    next_state_tensor.copy_(state[0])
    next_position=-1
    next_partRoute=copy.deepcopy(state[2])

    #执行action前
    cur_row=int(state[1]/num_of_rows)
    cur_col=state[1]%num_of_rows
    dest_row=int(dest_position/num_of_rows)
    dest_col=dest_position%num_of_rows
    
    #RL学习之前check一次，就能确保起码能走一步

    #开始执行action
    #state_tensor的四个channel,从0-3以此为N,S,W,E
    if(action==0):#沿x轴走
        if(cur_col<dest_col):
            next_state_tensor[0][3][cur_row*num_of_rows+cur_col]=1#更新tensor
            next_partRoute.append([cur_row*num_of_rows+cur_col,'E'])#更新路由表
            cur_col+=1#向East走了一步
        elif(cur_col>dest_col):
            next_state_tensor[0][2][cur_row*num_of_rows+cur_col]=1#更新tensor
            next_partRoute.append([cur_row*num_of_rows+cur_col,'W'])#更新路由表
            cur_col-=1#向West走了一步
    elif(action==1):#沿y轴走
        if(cur_row<dest_row):
            next_state_tensor[0][1][cur_row*num_of_rows+cur_col]=1#更新tensor
            next_partRoute.append([cur_row*num_of_rows+cur_col,'S'])#更新路由表
            cur_row+=1#向South走了一步
        elif(cur_row>dest_row):
            next_state_tensor[0][0][cur_row*num_of_rows+cur_col]=1#更新tensor
            next_partRoute.append([cur_row*num_of_rows+cur_col,'N'])#更新路由表
            cur_row-=1#向North走了一步
    
    next_position=cur_row*num_of_rows+cur_col

    ret_state,ret_reward,done=Check_if_Done_improved([next_state_tensor,next_position,next_partRoute],source_position,dest_position,link_set,num_of_rows,start_time,end_time)

    if(done==True):
        return ret_state,ret_reward,done
    else:
        contention=computeContention(next_partRoute,link_set,num_of_rows,start_time,end_time-start_time)
        return [next_state_tensor,next_position,next_partRoute],-1*contention,False



def CVB_method(execution,V_machine,num_of_rows):#execution里的task编号从0开始，没有多余的数据
    num_of_tasks=len(execution)
    mean_task=np.mean(execution)
    std_task=np.std(execution,ddof=1)
    V_task=std_task/mean_task
    #print("V_task=",V_task)
    #print("mean_task=",mean_task)
    alpha_task=1/(V_task*V_task)
    alpha_machine=1/(V_machine*V_machine)
    beta_task=mean_task/alpha_task
    #print("alpha_task=",alpha_task)
    #print("beta_task=",beta_task)
    beta_machine=[]
    q=[]
    e=np.zeros(shape=(num_of_tasks,num_of_rows*num_of_rows),dtype=np.int)
    for i in range(0,num_of_tasks):
        #q.append(np.random.gamma(shape=alpha_task,scale=beta_task))
        q.append(execution[i])
        #print("i=",i,"q[i]=",q[i])
        beta_machine.append(q[i]/alpha_machine)
        #print("i=",i,"beta_machine[i]=",beta_machine[i])
        for j in range(0,num_of_rows*num_of_rows):
            e[i][j]=int(np.random.gamma(shape=alpha_machine,scale=beta_machine[i]))
    return e


def read_NoC(NoC_file_name):
    ret=[]
    f=open(NoC_file_name)
    for line in f:
        tmp=[]
        for i in line[1:-2].split(','):
            tmp.append(int(i))
        ret.append(tmp)
    return ret


def init_from_json(input_json_file):
    task_graph={}
    with open(input_json_file,"r") as f:
        task_graph=json.load(f)
    num_of_tasks=len(task_graph)

    adj_matrix=np.zeros((num_of_tasks+1,num_of_tasks+1),dtype=np.int)#task从1开始算
    total_needSend=np.zeros(num_of_tasks+1,dtype=np.int)#task从1开始算的
    total_needReceive=np.zeros(num_of_tasks+1,dtype=np.int)#task从1开始算的
    execution=np.zeros(num_of_tasks+1,dtype=np.int)#task从1开始算的

    for i in task_graph.keys():#json里的task的序号是从0开始的
        total_needSend[int(i)+1]=task_graph[i]['total_needSend']
        total_needReceive[int(i)+1]=task_graph[i]['total_needReceive']
        execution[int(i)+1]=task_graph[i]['exe_time']
        for j in task_graph[i]['out_links']:
            adj_matrix[int(i)+1][int(j[0][0])+1]=int(j[0][1])
    
    return adj_matrix,total_needSend,total_needReceive,execution,num_of_tasks

def Get_weight(total_needSend,execution,c=4):
    exe_sum=0
    send_sum=0
    z=1
    num_of_tasks=len(execution)-1
    for i in range(1,num_of_tasks+1):
        exe_sum+=execution[i]
        send_sum+=total_needSend[i]
    
    map_w=exe_sum/z
    route_w=send_sum/c
    a=map_w/(map_w+route_w)
    b=route_w/(map_w+route_w)
    return a,b

def getMaxOne(l):
    ret=l[0]
    for i in l:
        if(i>=ret):
            ret=i
    return ret

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


if __name__ == '__main__':
    hyperperiod,num_of_tasks,edges,comp_cost=init('./inputFile/example.tgff')
    adj_matrix,total_needSend,total_needReceive,execution=Get_detailed_data(num_of_tasks,edges,comp_cost)
    #adj_matrix,total_needSend,total_needReceive,execution=init_from_json('./AIRfile/Autocor_Mesh8x8_AIR1_basic.json')
    
    print(adj_matrix)
    print(total_needSend)
    print(total_needReceive)
    print(execution)
    print(Get_weight(total_needSend,execution))
    
    #print(execution[1:])
    #print(CVB_method(execution=execution[1:],V_machine=0.5,num_of_rows=4))
    
