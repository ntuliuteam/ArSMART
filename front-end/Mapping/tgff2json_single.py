import sys
import getopt

import json

if sys.version > '3':
    import queue as Queue
else:
    import Queue


def takeSecond(elem):
        return int(elem[1])

def takeFirst(elem):
        return int(elem)

def sortedDict(adict):
        a=[]
        keys = adict.keys()
        keys.sort(key = takeFirst)
        for key in keys:
                a.append(tuple([key,adict[key]]))
        return a
def getNum(nameFull):
        a = nameFull.split('_')
        return a[1]

class TG2Timeline:
#read nodes and links
#return nodes[name,task size] links[from,to,tramsmission size]
        def __init__(self,filename,outputfile,rows):
                self.inputfile = filename
                self.outputfile = outputfile
                self.numPEs = rows*rows
                self.rows = rows

        def spDoc(self):
                # RPath = self.filename+".tgff"
                # NPath = self.filename+".n"
                # LPath = self.filename+".l"
                Nodes = []
                Links = []
                Dict = []


                RFileHandle = open (self.inputfile)


                relationList = RFileHandle.readlines()
                for oneLine in relationList:
                        words = oneLine.split()
                        if len(words) <= 1:
                                continue
                        #print(words[0])
                        if words[0][0] != "T" and words[0][0] != "A":
                                Dict.append(tuple([words[0],words[1]]))

                Hash = dict(Dict)

                for oneLine in relationList:
                        words = oneLine.split()
                        if len(words) <= 1:
                                continue
                        if words[0][0] == "T":
                                Nodes.append([words[1],Hash[words[3]]])
                        if words[0][0] == "A":
                                Links.append([words[3], \
                                        words[5],Hash[words[7]]])

                # if saveflag == 1:
                #         NFileHandle = open (NPath,"w")
                #         LFileHandle = open (LPath,"w")
                #         for Node in Nodes:
                #                 s = ' '.join(Node)
                #                 NFileHandle.write(s)
                #                 NFileHandle.write('\n')
                #         NFileHandle.close()
                #         print("save nodes into " + NPath)

                #         for Link in Links:
                #                 s = ' '.join(Link)
                #                 LFileHandle.write(s)
                #                 LFileHandle.write('\n')
                #         LFileHandle.close()
                #         print("save links into " + LPath)


                return Nodes,Links


        def getNode(name):
                # RPath = name+".tgff"
                # GPath = name+".g"

                RFileHandle = open (name)
                # GFileHandle = open (GPath,"w")


                rlist = RFileHandle.readline()
                rlist = rlist.strip()

                #mlen = int(rlist.split()[0])
                print(rlist)



        #set the node class
        #{'name':{'start_time':0,'end_time':0,'exe_time':0,
        #'input_links':[{'from':'size'}],'out_links':[{'to':'size'}]}}
        def GraphStruct(self,Nodes,Links):
                Graph = []
                a = []

                for node in Nodes:
                        #t = dict(node[0]:dict('start_time':0,'end_time':0,
                        #'exe_time':node[1],'input_links':a,'out_links':b))
                        t = int(getNum(node[0]))
                        mapto = t%self.numPEs
                        halfrow = self.rows/2
                        centreNode = self.numPEs/2 - halfrow
                        if(t == 0):
                            mapto = int(centreNode)
                        elif(t == centreNode):
                            mapto = 0
                        if(t>=self.numPEs):
                            mapto = int(centreNode + t%self.numPEs)


                        dict1 = {'start_time':1,'end_time':1,'exe_time': \
                                int(node[1]), \
                                'total_needSend':0, \
                                'total_needReceive':0,'input_links':[], \
                                'out_links':[],'visited':0,'mapto':mapto}
                        t = tuple([getNum(node[0]),dict1])
                        a.append(t)
                Graph = dict(a)

                Links.sort()

                for link in Links:

                        Graph[getNum(link[0])]['out_links'] \
                        .append([int(getNum(link[1])), \
                                int(int(link[2])),[],[],0,Graph[getNum(link[0])]["mapto"],Graph[getNum(link[1])]["mapto"]])
                        Graph[getNum(link[1])]['input_links'] \
                        .append(tuple([getNum(link[0]), \
                                int(int(link[2]))]))
                #Graph['t0_3']['out_links'].sort(key=takeSecond)
                #print(Graph['t0_3'])
                
                GPath = self.outputfile
                with open(GPath,"w") as f:
                        json.dump(Graph,f)

                return Graph


        #Use Queue to find the transmission start time
        def StartTime(self,Graph,root):
                q = Queue.Queue()
                q.put(root)
                Graph[root]['end_time'] = Graph[root]['start_time'] \
                + Graph[root]['exe_time']
                while not q.empty():
                        #check if it should be processed
                        t = q.get()
                        flag = 0
                        for fnode in Graph[t]['input_links']:
                                if Graph[fnode[0]]['end_time'] == 0:
                                        flag = 0
                                        break
                        #if its any parents nodes haven't be processed,
                        #in queue again
                        if flag == 1:
                                q.put(t)
                        else:
                                Graph[t]['out_links'].sort(key=takeSecond)
                                sumSend = 0
                                for node in Graph[t]['out_links']:
                                        #if have not been visited
                                        sumData = 0
                                        sumSend = sumSend + int(node[1])
                                        if Graph[node[0]]['end_time'] \
                                        == 0:
                                                #compute the start_time
                                                #and end_time
                                                maxEnd = 0

                                                for fnode in Graph[node[0]] \
                                                ['input_links']:
                                                        stime = \
                                                        Graph[fnode[0]] \
                                                        ['end_time'] \
                                                         + int(fnode[1])
                                                        sumData = \
                                                        sumData + int(fnode[1])
                                                        if maxEnd < stime:
                                                                maxEnd = stime
                                                Graph[node[0]] \
                                                ['start_time'] = 1
                                                Graph[node[0]]['end_time'] = \
                                                maxEnd + \
                                                Graph[node[0]]['exe_time']
                                                Graph[node[0]] \
                                                ['total_needReceive'] = sumData

                                                q.put(node[0])
                                Graph[t]['total_needSend'] = sumSend
                #print(Graph['t0_37'])
        
                GPath = self.outputfile
                with open(GPath,"w") as f:
                        json.dump(Graph,f)

                return Graph

        #Generate the timeline and save it to Json file
        #Timeline {'time':[src,dest]}
        def G2Timeline(self,Graph,root):
                q = Queue.Queue()
                q.put(root)
                timePoint = {}
                while not q.empty():
                        t = q.get()
                        #print("out put"+t)
                        Graph[t]['out_links'].sort(key=takeSecond)
                        traffic = []

                        for node in Graph[t]['out_links']:
                                traffic.append([t,node[0],node[1]])
                                if Graph[node[0]]['visited'] == 0:
                                        q.put(node[0])
                                        Graph[node[0]]['visited'] = 1
                                if timePoint.has_key \
                                (str(Graph[t]['end_time'])):
                                        timePoint[str(Graph[t] \
                                        ['end_time'])]. \
                                        append([t,node[0]])
                                else:
                                        timePoint[str(Graph[t] \
                                        ['end_time'])] = []
                                        timePoint[str(Graph[t] \
                                        ['end_time'])]. \
                                        append([t,node[0]])
                        timepre = 0
                        while(len(traffic) > 0):
                                time = \
                                Graph[t] \
                                ['end_time'] + \
                                int(traffic[0][2])
                                if timepre != time:
                                        for traf in traffic:
                                                if timePoint. \
                                                has_key(str(time)):
                                                        timePoint \
                                                        [str(time)]. \
                                                        append \
                                                        ([traf[0],traf[1]])
                                                else:
                                                        timePoint \
                                                        [str(time)] = []
                                                        timePoint \
                                                        [str(time)]. \
                                                        append \
                                                        ([traf[0],traf[1]])
                                timepre = time
                                traffic.remove(traffic[0])
                timp = {}
                #print(timePoint.items())
                timp = sortedDict(timePoint)


                # if saveflag == 1:
                #         TPath = self.filename+".t"
                #         with open(TPath,"w") as f:
                #                 json.dump(timp,f)
                #                 print("save timeline into " + TPath)

        def computeSend(self):
            with open(self.outputfile,"r") as f:
                taskGraph = json.load(f)

                for i in range(0,len(taskGraph)):
                    sumSend1 = 0
                    for outputLink in taskGraph[str(i)]['out_links']:
                        sumSend1 = sumSend1 + outputLink[1]
                    taskGraph[str(i)]['total_needSend'] = sumSend1
                    sumSend = 0
                    for outputLink in taskGraph[str(i)]['input_links']:
                        sumSend = sumSend + outputLink[1]
                    taskGraph[str(i)]['total_needReceive'] = sumSend
                    for j in range(0,len(taskGraph[str(i)]['out_links'])):
                        taskGraph[str(i)]['out_links'][j]=[taskGraph[str(i)]['out_links'][j]]
                        
                        taskGraph[str(i)]['out_links'][j][0][0] = int(taskGraph[str(i)]['out_links'][j][0][0])
                    

            GPath = self.outputfile
            with open(GPath,"w") as f:
                json.dump(taskGraph,f)
                print("save graph into " + GPath)


def main(argv):
    inputfile = ''
    outputfile = ''
    rows = 4
    try:
        opts, args = getopt.getopt(argv,"hi:o:r:",["ifile=","ofile=","rows"])
    except getopt.GetoptError:
        print('Error TG2Timeline.py -i <inputfile> -s <saveflag>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('TG2Timeline.py -i <inputfile> -s <saveflag>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            print("read input file")
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-r", "--rows"):
            rows = int(arg)
    

    # print('inputfile: ', inputfile)
    # print('outputfile: ', outputfile)
    G2T = TG2Timeline(inputfile,outputfile,rows)
    Nodes,Links = G2T.spDoc()
    Graph = G2T.GraphStruct(Nodes,Links)
    root = '0'
#    GraphUpdated = G2T.StartTime(Graph,root)
    G2T.computeSend()
#     G2T.G2Timeline(Graph,root)
    print("Transferred TGFF file to json.")

if __name__ == "__main__":
    main(sys.argv[1:])
    
