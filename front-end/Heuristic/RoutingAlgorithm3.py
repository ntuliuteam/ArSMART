import sys
import getopt
import json
import math
#import Queue
import networkx as nx
import pylab
import numpy as np
import logging, sys




class linklist():
    def __init__(self):
        '''
        初始化节点
        
        |type reserveList:
        |
        '''
        self.nList = 0
        self.sList = 0
        self.wList = 0
        self.eList = 0
        self.pList = 0

class onlineTimeline: 
    def __init__(self,inputfile,rowNum):
        self.inputfile =inputfile
        self.totalNum = int(rowNum)*int(rowNum)
        self.rowNum = int(rowNum)
        self.NoClink = []
     

        for i in range(0,self.totalNum):
            link = linklist()
            
            self.NoClink.append(link)
         

        self.sendMatrix = []
        self.receiveMatrix = []
        self.totalSize=0
        self.exeMatric = []
        self.taskGraph = {}
        self.pendTask = []
        self.routeNow = {}
     
       
        self.stateMatrix=[3]
        self.sendingNow = []
        self.MapResult=[]
        self.mapTo = []
        self.outLinks = {}

        self.nowPri = -1
        self.network = self.iniNetwork(self.totalNum,self.rowNum,self.rowNum)

        

        self.nowTime = 0

   
        
    def loadGraph(self):


        with open("./allfile_result/Autocor_Mesh8x8_AIR4_basic.json","r") as f:
            taskGraph1 = json.load(f)
            sumR= 0
            sumS=0
            print(len(taskGraph1))

            for i in range(0,len(taskGraph1)):
                self.sendMatrix.append(taskGraph1[str(i)]['total_needSend'])
                self.receiveMatrix.append(taskGraph1[str(i)]['total_needReceive'])
                self.totalSize = self.totalSize + taskGraph1[str(i)]['total_needSend']
                self.exeMatric.append(taskGraph1[str(i)]['exe_time'])

                # need change
                self.mapTo.append(str(i))
                
                self.stateMatrix.append(1000)
                for task in taskGraph1[str(i)]['out_links']:
                    linkName = str(i)+"to"+str(task[0][0])
                    self.outLinks[linkName] = {"state":-1,"size":task[0][1]}
                    task.append(0)
                    

            self.taskGraph = taskGraph1
        with open(self.inputfile+"MapResult.json","r") as f:
            self.MapResult = json.load(f)

        print("task graph loaded++++++++++++++++++++++")
        print("sendMatric",self.sendMatrix)
        print("receiveMatric",self.receiveMatrix)
        print("exeMatric",self.exeMatric)
        print("outLinks",self.outLinks)
        
        self.stateMatrix[1]=1
        
        
       
        print("taskGraph",self.taskGraph)
        print("MapResult",self.MapResult)
   
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++")



    def findStartExe(self):
        exeThisTime=[]
        for i in range(0,len(self.taskGraph)):
            #print(self.receiveMatrix[i],self.exeMatric[i])
            if(self.receiveMatrix[i]<=0 and self.exeMatric[i]>0):
                exeThisTime.append(i)
        return exeThisTime


    def startExe(self,exeThisTime):
        for i in exeThisTime:
            self.exeMatric[i]=self.exeMatric[i]-1



    def findStartSend(self):
        sendThisTime=[]
        #chagned here
        for i in range(0,len(self.taskGraph)):
            if(self.sendMatrix[i]>0 and self.exeMatric[i]<=0 ):#and self.stateMatrix[i]!=3):
                sendThisTime.append(i)
        return sendThisTime


  
    def sending(self,sendinfos):
        releaseList = []
        for item in sendinfos:
            linkName = str(item[0])+"to" + str(item[1])
            print(item,self.outLinks[linkName]["size"])
            if(self.outLinks[linkName]["size"]==0):
                releaseList.append(item)
                continue

            self.outLinks[linkName]["size"] = self.outLinks[linkName]["size"]-1
            self.sendMatrix[item[0]]=self.sendMatrix[item[0]]-1
            self.receiveMatrix[item[1]]=self.receiveMatrix[item[1]]-1
                
        return releaseList

    def reserveRoute(self,route,w):
        route3 = self.edgeToNet(route)
        self.removeLinks(self.network,route3,w)
        for rt in route:
            if(rt[1] == 'E'):
                self.NoClink[rt[0]].eList = 1
            elif(rt[1] == 'W'):
                self.NoClink[rt[0]].wList = 1
            elif(rt[1] == 'N'):
                self.NoClink[rt[0]].nList = 1
            elif(rt[1] == 'S'):
                self.NoClink[rt[0]].sList = 1


    def printNoC(self):
        for i in range(0,len(self.NoClink)):
            print(i,self.NoClink[i].nList,self.NoClink[i].sList,self.NoClink[i].wList,self.NoClink[i].eList,end=" | ")
     
    def changeIndex(self,index):
        return (int(int(index)/self.rowNum),int(int(index)%self.rowNum))

    def computeRouteOnline(self,network,src,dest):
    #Compute route. if there is no route, retuen []. Will add the route computing algorithm in the future
    #Input: network, src, dest
    #Output: path 
        try:
            path=nx.dijkstra_path(network, src, dest)
            return path
        except nx.NetworkXNoPath:
            print("no path here")
            return []

    def netToEdge(self,route):
        edges=[]
        for i in range(0,len(route)-1):
            if(route[i][0]-route[i+1][0]==-1):
                edges.append([route[i][0]*self.rowNum+route[i][1],'S'])
            elif(route[i][0]-route[i+1][0]==1):
                edges.append([route[i][0]*self.rowNum+route[i][1],'N'])
            elif(route[i][1]-route[i+1][1]==-1):
                edges.append([route[i][0]*self.rowNum+route[i][1],'E'])
            elif(route[i][1]-route[i+1][1]==1):
                edges.append([route[i][0]*self.rowNum+route[i][1],'W'])

        return edges
    def edgeToNet(self,route):
        edges = []
        edges.append(self.changeIndex(route[0][0]))
        for i in range(0,len(route)):
            nextNode = list(self.changeIndex(route[i][0]))
            if(route[i][1]=='S'):
                nextNode[0]=nextNode[0]+1
            elif(route[i][1]=='N'):
                nextNode[0]=nextNode[0]-1
            elif(route[i][1]=='E'):
                nextNode[1]=nextNode[1]+1
            elif(route[i][1]=='W'):
                nextNode[1]=nextNode[1]-1
            edges.append(tuple(nextNode))
        return edges

    def removeLinks(self,network,route,w):
        # for i in range(0,len(route)-1):
        #     #iPrint("remove links here",tuple([route[i],route[i+1]]))
        #     try:
        #         network.remove_edge(route[i],route[i+1])
        #     except:
        #         print("no need to remove")
        print(network.edges)
        for i in range(0,len(route)-1):
            print(network.edges[route[i],route[i+1]])
            network.edges[route[i],route[i+1]]["weight"]=network.edges[route[i],route[i+1]]["weight"]+w


    def addLinks(self,network,route,w):
        # for i in range(0,len(route)-1):
        #     network.add_edge(route[i],route[i+1])
        for i in range(0,len(route)-1):
            if(network.edges[route[i],route[i+1]]["weight"]-w>0):
                network.edges[route[i],route[i+1]]["weight"]=network.edges[route[i],route[i+1]]["weight"]-w
            else:
                network.edges[route[i],route[i+1]]["weight"] = 0

    def sendPackage(self,sendThisTime):
    #Task State: Not start yet -1
    #sending 1
    #Pending 2
    #finished 3  
        releaseList = []
        
        for i in sendThisTime:
            #print(self.taskGraph[str(i)]['out_links'])
            for dest in self.taskGraph[str(i)]['out_links'][:]:
                # Not start yet
                linkName = str(i)+"to"+str(dest[0][0])
                if(self.outLinks[linkName]["state"]==-1):
                    sendtoi = int(dest[0][0])
                    #if send or receive already finish
                    print(sendtoi)
                   
                    # route compute insert here
                    srcMapto = self.taskGraph[str(i)]['mapto']
                    dstMapto = dest[0][-1]

                    print("compute route",self.changeIndex(srcMapto),self.changeIndex(dstMapto))
                    route  = self.computeRouteOnline(self.network,self.changeIndex(srcMapto),self.changeIndex(dstMapto))
                    route = self.netToEdge(route)
                    dest[0][3] = route
                    
                    #route3 = self.edgeToNet(route2)
                   # print("dijkstra_path: ",route, "  defaultRoute: ",defaultRoute)

                    #print(route)

                    if(len(route)==0):
                        self.sendMatrix[i]=self.sendMatrix[i]-dest[1]
                        self.receiveMatrix[sendtoi]=self.receiveMatrix[sendtoi]-dest[1]
                        #dest[-1]=3
                        self.outLinks[linkName]["state"]=3
                        canSend=-1
                        print("Since Mapped to same core, this transmiss = 0", i,sendtoi)
                        self.taskGraph[str(i)]['out_links'].remove(dest)
                    else:  
                        canSend = self.checkCanSend(route)

                    if(canSend==0):
                        #print("should Pending",sendtoi)

                        self.outLinks[linkName]["state"]=2
                        self.pendTask.append([i,sendtoi,route,dest[0][1]])
                        self.nowPri=1000
                     
                    elif(canSend!=0):
                        #Start to transmiss
                        print("checking can send=",sendtoi)

                        #self.stateMatrix[sendtoi]=-1
                        #self.stateMatrix[i]=1
                        
                        self.sendingNow.append([i,sendtoi,route])
                        #self.taskGraph[str(i)]['out_links'].remove(dest)
                        self.outLinks[linkName]["state"]=1
                        #dest[-1]=1
                        self.reserveRoute(route,dest[0][1])


        for i in self.pendTask[:]:
            canSend = self.checkCanSend(i[2])

            if(canSend==1):
                print("from pending",i)
                #self.stateMatrix[i[1]]=-1
                linkName = str(i[0])+"to"+str(i[1])
                self.outLinks[linkName]["state"]=1
                #self.stateMatrix[i[0]]=1

                self.reserveRoute(i[2],i[-1])
                self.sendingNow.append(i)
                #releaseList.append(self.sending(i[0],i[1],i[2]))
                self.pendTask.remove(i)


        reList=[]
        reList=self.sending(self.sendingNow)
        if(reList!=None and len(reList)!=0):
            releaseList=reList

        return releaseList

    #if stateMatrix==3 deleted
    def releaseRec(self,releaseList):
        print("releaseNow,",releaseList)
        for item in releaseList:
            if(item!=None):
                linkName = str(item[0])+"to" + str(item[1])
                self.outLinks[linkName]["state"]==3
                #addlinks
                route1 = self.edgeToNet(item[2])
                self.addLinks(self.network,route1,item[1])
                

                for rt in item[2]:
                    if(rt[1] == 'E'):
                        self.NoClink[rt[0]].eList = 0
                    elif(rt[1] == 'W'):
                        self.NoClink[rt[0]].wList = 0
                    elif(rt[1] == 'N'):
                        self.NoClink[rt[0]].nList = 0
                    elif(rt[1] == 'S'):
                        self.NoClink[rt[0]].sList = 0


                self.sendingNow.remove(item)
                #print(self.sendingNow)
        #print("after releasing")
        #self.printNoC()


    def computeTime(self):
        z= 0
        remain = 1000
        releaseList = []
        while(remain>0):#z<20):#
            z=z+1
            # if(self.nowPri==1000):
            #     break
            

            print("In the cycle +++++++++++++++++----------------------------",self.nowTime)

            signal = 0
            remain = 0

            

            sendThisTime =self.findStartSend()
            print("sendThisTime",sendThisTime)
            exeThisTime = self.findStartExe()
            print("exeThistime",exeThisTime)

            releaseList = self.sendPackage(sendThisTime)
            

            

            #Find can exe this time 
            

            self.startExe(exeThisTime)


            #self.checkSend()


            self.releaseRec(releaseList)
            ##print Network State
            self.Dprint("sendMatric",self.sendMatrix)
            self.Dprint("receiveMatric",self.receiveMatrix)
            self.Dprint("exeMatric",self.exeMatric)
            self.Dprint1("sendingNow",self.sendingNow)
            self.Dprint1("releaseList",releaseList)
            self.Dprint1("pendingList",self.pendTask)
            self.Dprint("stateMatrix",self.stateMatrix)
            self.printNoC()

            for i in range(0,len(self.exeMatric)):
                remain=remain+self.exeMatric[i]


            self.nowTime=self.nowTime+1
            # if(self.nowTime>0):
            #     str1 = input()
    
        print("total time:",self.nowTime)
        self.Dprint("sendMatric",self.sendMatrix)
        self.Dprint("receiveMatric",self.receiveMatrix)
        self.Dprint("exeMatric",self.exeMatric)
        self.Dprint1("sendingNow",self.sendingNow)

        with open("result2.json","w") as f:
            json.dump(self.taskGraph,f)

    def Dprint(self,name1,list1):
        print(name1)
        i=0
        for i in range(0,len(list1)):
            print(i,list1[i],end=' | ')
        print('\n')

    def Dprint1(self,name1,list1):
        print(name1)
        i=0
        for i in range(0,len(list1)):
            print(list1[i])
        

    
    def computeRouteXY(self,i,dst):
        route = []
        (srcX,srcY) = self.changeIndex(i)
        (dstX,dstY) = self.changeIndex(dst)
        
        while(dstY > srcY):
            route.append([i,'E'])
            srcY = srcY + 1
            i = i + 1

        while(dstY < srcY):
            route.append([i,'W'])
            srcY = srcY - 1
            i = i - 1

        while(dstX > srcX):
            route.append([i,'S'])
            srcX = srcX +1
            i = i + self.rowNum

        while(dstX < srcX):
            route.append([i,'N'])
            srcX = srcX - 1
            i = i - self.rowNum

        return route        
            
    def iniNetwork(self,size,X,Y):
        print('Generate a graph')
        G = nx.DiGraph()
        #iPrint('Layout')
        H=nx.grid_2d_graph(X,Y)
        G.add_nodes_from(H.nodes())
        G.add_edges_from(H.edges(),weight=0)
        for edge in H.edges():
            G.add_edge(edge[1],edge[0],weight=0)
            #G.add_edge(edge[0],edge[1],weight=0)
        #pos=nx.shell_layout(G)
        #iPrint('Graph')
        #nx.draw(G,pos,with_labels=True, node_color='white', edge_color='red', node_size=400, alpha=0.5 )
        #pylab.title('Self_Define Net',fontsize=15)
        #pylab.show()
        return G

    def checkCanSend(self,route):
        print("this is for checking",route)
        #print("checking resout+++++++++++++")
        for rt in route:
            #print(rt,self.NoClink[rt[0]].eList,self.NoClink[rt[0]].wList,self.NoClink[rt[0]].nList,self.NoClink[rt[0]].sList)
            if(rt[1] == 'E' and self.NoClink[rt[0]].eList == 1):
                return False
            elif(rt[1] == 'W' and self.NoClink[rt[0]].wList == 1):
                return False
            elif(rt[1] == 'N' and self.NoClink[rt[0]].nList == 1):
                return False
            elif(rt[1] == 'S' and self.NoClink[rt[0]].sList == 1):
                return False

        return True

    def changeName(self,i,dst):
        return str(i)+'+'+str(dst)

    def startSending(self,i,dst,route):
        self.routeNow[self.changeName(i,dst)] = route
        print("this is starting------------------------",i,dst)
        
        for rt in route:
            if(rt[1] == 'E'):
                self.NoClink[rt[0]].eList = 1
            elif(rt[1] == 'W'):
                self.NoClink[rt[0]].wList = 1
            elif(rt[1] == 'N'):
                self.NoClink[rt[0]].nList = 1
            elif(rt[1] == 'S'):
                self.NoClink[rt[0]].sList = 1

        #for i in range(0,50):
           #print(rt,self.NoClink[i].eList,self.NoClink[i].wList,self.NoClink[i].nList,self.NoClink[i].sList)

    def changeIndex(self,index):
        return (int(int(index)/self.rowNum),int(int(index)%self.rowNum))

def main(argv):
    inputfile = ''
    rowNum = ''
    try:
        opts, args = getopt.getopt(argv,"hi:r:",["ifile=","row="])
    except getopt.GetoptError:
        print('Error RoutingAlgorithm3.py -i <inputfile> -r <row> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('RoutingAlgorithm3.py -i <inputfile> -r <row> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            print("read input file")
            inputfile = arg
        elif opt in ("-r", "--row"):
            rowNum = arg

    print('inputfile：', inputfile)
    print('row: ', rowNum)
    task = onlineTimeline(inputfile,rowNum)
    task.loadGraph()
    task.computeTime()
    print("RoutingAlgorithm3 completed!")


if __name__ == "__main__":
    main(sys.argv[1:])
    


