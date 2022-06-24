##
# @script   contentionMap.py
# @brief    Map tasks
# @author   dawnzju<chenhuizju1@gmail.com>
# @version  1.0.0
# @modify   2020-12-03

import sys 
import getopt
import json
import networkx as nx
import numpy as np

import matplotlib.pyplot as plt

def main(argv):
    inputfile = ''
    airDegree = ''
    outputfile = ''
    split = 1
    try:
        opts, args = getopt.getopt(argv,"hi:r:o:s:",["ifile=","row=","ofile=","split="])
    except getopt.GetoptError:
        print('Error contentionMap.py -i <inputfile> -r <row> -s <splitting>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('contentionMap.py -i <inputfile> -r <row> -s <splitting>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            # print("read input file")
            inputfile = arg
        elif opt in ("-r", "--row"):
            airDegree = int(arg)
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-s", "--split"):
            split = int(arg)

    mappingTask(inputfile,airDegree,outputfile)

def bfs_search(taskNet,root):
    edges_list = list(nx.traversal.bfs_edges(taskNet,root))
    nodes_list = list(edges_list[0])
    for k,v in edges_list[1:]:
        nodes_list.append(v)
    return(nodes_list)


def mappingTask(inputfile,airDegree,outputfile):
    with open(inputfile) as load_f:
        taskGraph = json.load(load_f)
        taskNet = nx.Graph()
        for taskName,item in taskGraph.items():
            taskNet.add_node(taskName)
        for taskName,item in taskGraph.items():
            for oneLink in item["out_links"]:
                taskNet.add_edge(taskName,str(oneLink[0][0]))
        nodelist = bfs_search(taskNet,"0")
        # print(nodelist)
        G = nx.DiGraph()
        #iPrint('Layout')
        H=nx.grid_2d_graph(airDegree,airDegree)
        G.add_nodes_from(H.nodes())
        G.add_edges_from(H.edges())
        for edge in H.edges():
            G.add_edge(edge[1],edge[0])
        mid = int(airDegree/2)-1
        routerList = bfs_search(G,(mid,mid))
        # print(routerList)
        unitLength = len(routerList)
        count = 0
        # mapping function
        for taskNode in nodelist:
            
            choose = routerList[count%unitLength]
            taskGraph[str(taskNode)]["mapto"] = choose[0]*airDegree + choose[1]
            count += 1
        
        for taskName,item in taskGraph.items():
            for onelink in item["out_links"]:
                onelink[0][-2] = taskGraph[taskName]["mapto"]
                onelink[0][-1] = taskGraph[str(onelink[0][0])]["mapto"]
        
        for idx, item in taskGraph.items():

            item["input_links"]=[]
        for idx, item in taskGraph.items():

            for outlink1 in item["out_links"]:
                outlink = outlink1[0]
                taskGraph[str(outlink[0])]["input_links"].append([idx,outlink[1]])
                #print(outlink,idx)

        with open(outputfile,'w') as f:
            json.dump(taskGraph,f)
        # nx.draw(taskNet)
        # plt.savefig("path.png")
        
        print("Mapping algorithm completed!!")

if __name__ == "__main__":
    main(sys.argv[1:])
    print("Mapping completed!")



