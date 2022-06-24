#scons build/Garnet_standalone/gem5.debug -j25 NUMBER_BITS_PER_SET=256
#python3 run_folder.py -i allresult
from nis import maps
import os
import sys
import getopt

def processMapping(path,rowNum):
    command = "python ./front-end/Mapping/tgff2json_single.py -i "+path+" -o ./mapRouteResult/result_Mesh"+rowNum+"x"+rowNum+"_AIR1_free.json"
    os.system(command)
    command = "python ./front-end/Mapping/contentionMap.py -i ./mapRouteResult/result_Mesh"+rowNum+"x"+rowNum+"_AIR1_free.json -r "+rowNum+" -o ./mapRouteResult/result_Mesh"+rowNum+"x"+rowNum+"_AIR1_free.json -s 1"
    os.system(command)

def processXY(rowNum):
    command = "python ./front-end/XY/RoutingAlgorithmXY.py -i ./mapRouteResult/result_Mesh"+rowNum+"x"+rowNum+"_AIR1_free.json -r "+rowNum+" -o ./mapRouteResult/result_Mesh"+rowNum+"x"+rowNum+"_AIR1_result.json"
    os.system(command)

def processA1(rowNum):
    command = "python ./front-end/Heuristic/RoutingAlgorithm1.py -i ./mapRouteResult/result_Mesh"+rowNum+"x"+rowNum+"_AIR1_free.json -r "+rowNum+" -o ./mapRouteResult/result_Mesh"+rowNum+"x"+rowNum+"_AIR1_result.json"
    os.system(command)

def processA2(rowNum):
    command = "python ./front-end/Heuristic/RoutingAlgorithm2.py -i ./mapRouteResult/result_Mesh"+rowNum+"x"+rowNum+"_AIR1_free.json -r "+rowNum+" -o ./mapRouteResult/result_Mesh"+rowNum+"x"+rowNum+"_AIR1_result.json"
    os.system(command)

def processLAMP(rowNum):
    command = "python ./front-end/LAMP/LAMP.py -i ./mapRouteResult/result_Mesh"+rowNum+"x"+rowNum+"_AIR1_free.json -r "+rowNum+" -o ./mapRouteResult/result_Mesh"+rowNum+"x"+rowNum+"_AIR1_result.json"
    os.system(command)

def processMARCO(inputfile,rowNum,archfile,vmachine):
    if archfile == "":
        command = "python ./front-end/MARCO/MARCO.py -i "+inputfile+" -r "+rowNum+" -o ./mapRouteResult/result_Mesh"+rowNum+"x"+rowNum+"_AIR1_result.json -v "+vmachine
    else:
        command = "python ./front-end/MARCO/MARCO.py -i "+inputfile+" -r "+rowNum+" -o ./mapRouteResult/result_Mesh"+rowNum+"x"+rowNum+"_AIR1_result.json -c "+archfile
    
    os.system(command)


def instructionMode():
    inputfile = input("Please input file name (.tgff). : ")
    
    if not os.path.isfile("./inputFile/"+inputfile):
        print("### File does not exist. Please check. Using the default file example.tgff")
        inputfile = "./inputFile/example.tgff"
    
    rowNum = input("### Please input the number of rows of the system (1-16 due to the limitation of simulator): ")
    try:
        ra = int(rowNum)
    except ValueError:
        print("### Invalid row number! Please follow the instruction. Use the default value (8).")
        rowNum = "8"
    else:
        if int(rowNum) <=0 or int(rowNum)>16:
            print("### Invalid row number! Please follow the instruction. Use the default value (8).")
            rowNum = "8"

    print("### Please choose the mapping algorithm you want to apply: ")
    print("### 0. Heuristic; 1. Co-optimization(with routing); 2. User specific (already put the mapping result to ./mapRouteResult/result_Mesh"+rowNum+"x"+rowNum+"_AIR1_free.json)")
    mapSelect = input("### Your choice: ")
    skipRouting = 0
    rowNum = int(rowNum)



    if mapSelect == "0":
        processMapping(inputfile,str(rowNum))

    elif mapSelect == "1":
        print("### Please choose the architecture description file")
        print("### 0. Generated; 1. User specified; 2. Default")
        vmachine = "0.25"
        archfile = ""
        archSelect = input("### Your choice: ")
        if archSelect == "1":
            print("### Please input the architecture description file in ./front-end/MARCO/NoCDescription/ ")
            archfile = input("File Name (Do not include the path): ")
            if not os.path.isfile("./front-end/MARCO/NoCDescription/"+archfile):
                print("### File does not exist. Please check. Using the default file ./front-end/MARCO/NoCDescription/example_NoCdescription.txt")
                archfile = "./front-end/MARCO/NoCDescription/example_NoCdescription.txt"

        elif archSelect == "0":
            vmachine = input("### Please input the vmachine of the architecture (0-1): ")
            if float(vmachine)>1 or float(vmachine)<0:
                print("### Invalid value! Please follow the instruction. Use the default value (0.25)")

        elif archSelect == "2":
            print("Use the generated method with vmachine = 0.25")
        else:
            print("### Invalid value! Please follow the instruction. Use the generated method with vmachine = 0.25")       

        processMARCO(inputfile,str(rowNum),archfile,vmachine)
        print("### No need to choose routing algorithm")
        skipRouting = 1
    elif mapSelect == "2":
        print("Use the user specific mapping result")
    else:
        print("### Invalid input. Please follow the instruction. Use the default value (0).")
        processMapping(inputfile,str(rowNum))

    
    if skipRouting == 0:
        print("### Please choose the routing algorithm you want to apply: ")
        print("### 0. XY; 1. Arbitrary-1; 2. Arbitrary-2; 3. LAMP; 4. User-specificed (already put the routing result to ./mapRouteResult/result_Mesh"+str(rowNum)+"x"+str(rowNum)+"_AIR1_result.json)")
        rtSelect = input("### Your choice: ")
        if rtSelect == "0":
            processXY(str(rowNum))
        elif rtSelect == "1":
            processA1(str(rowNum))
        elif rtSelect == "2":
            processA2(str(rowNum))
        elif rtSelect == "3":
            processLAMP(str(rowNum))
        elif rtSelect == "4":
            print("Use the user specific routing result")
        else:
            print("### Invalid input. Please follow the instruction. Use the default value (0)")
            processXY(str(rowNum))

    print("### The mapping and routing file is in ./mapRouteResult folder")
    print("### Do you want to simulate this solution?")
    print("### 0. No 1. Yes")
    exeSelect = input("### Your choice:")
    if exeSelect == "0":
        print("### Skip simulation!")
    elif exeSelect == "1":
        allsize = rowNum * rowNum
        
        simSelect = input("### Please input the number of cycle (>0) you want to simulate: ")

        try:
            ra = int(simSelect)
        except ValueError:
            print("### Invalid input. Please follow the instruction. Use the default value (100000).")
            simSelect = "100000"
        else:
            if int(simSelect) <=0:
                print("### Invalid input. Please follow the instruction. Use the default value (100000)")
                simSelect = "100000"
       
        
        commandNow = "sudo ./simulator/build/Garnet_standalone/gem5.debug --debug-flags=GarnetSyntheticTraffic ./simulator/configs/example/garnet_synth_traffic.py --network=garnet2.0 --single-flit --synthetic=taskgraph --num-cpus="+str(allsize)+" --num-dirs="+str(allsize)+" --topology=Mesh_XY --mesh-rows="+str(rowNum)+" --sim-cycles="+str(simSelect)+" --filename=mapRouteResult/result_Mesh8x8_AIR1_result.json"


        print("### Please choose the architecture you want to use: ")
        print("### 0. Traditional NoC 1. SMART NoC 2. ArSMART NoC (Please note that traditional NoC and SMART NoC can only support XY routing)")
        arSelect = input("### Your choice: ")
        commandFinal = ""
        if arSelect == "0":
            commandFinal = commandNow
        elif arSelect == "1":
            hpcSelect = input("### Please input the hpcmax of the architecture (>=1): ")
            try:
                ra = int(hpcSelect)
            except ValueError:
                print("### Invalid input. Please follow the instruction. Use the default value (8).")
                hpcSelect = "8"

            commandFinal = commandNow + " --smart --hpcmax="+hpcSelect    
        elif arSelect == "2":
            hpcSelect = input("### Please input the hpcmax of the architecture (>=1): ")
            try:
                ra = int(hpcSelect)
            except ValueError:
                print("### Invalid input. Please follow the instruction. Use the default value (8).")
                hpcSelect = "8"
            ConfigTime = input("### Please input the configuration time for the single router (>=0): ")
            try:
                ra = int(ConfigTime)
            except ValueError:
                print("### Invalid input. Please follow the instruction. Use the default value (3).")
                ConfigTime = "3"
            commandFinal = commandNow + " --arsmart --hpcmax="+hpcSelect + " --configure-time="+ConfigTime
        else:
            print("### Invalid input. Please follow the instruction. Use the default value (0)")
            commandFinal = commandNow
        
        os.system(commandFinal)
        #command = "sudo ./build/Garnet_standalone/gem5.debug --debug-flags=GarnetSyntheticTraffic configs/example/garnet_synth_traffic.py --network=garnet2.0 --num-cpus="+str(allsize)+" --num-dirs="+str(allsize)+" --smart_hpcmax=8 --topology=Mesh_XY "+"--mesh-rows="+size+" --sim-cycles=1000000 --single-flit --synthetic=taskgraph --central --filename="+fullname

    else:
        print("### Invalid input. Please follow the instruction. Use the default value (0)")
        print("### Skip simulation!")

def flagMode(inputfile,rowNum,mapSelect,rtSelect,vmachine,archfile,exeSelect,arSelect,simSelect,hpcSelect,configTime):
    
    if not os.path.isfile(inputfile):
        print("### File does not exist. Please check")
        return 0

    skipRouting = 0
    rowNum = int(rowNum)

    if mapSelect == "0":
        processMapping(inputfile,str(rowNum))
    elif mapSelect == "1":  
        processMARCO(inputfile,str(rowNum),archfile,vmachine)
        skipRouting = 1
    elif mapSelect == "2":
        print("")
    else:
        print("### Invalid mapping choose. Please follow the instruction.")
        return 0

    
    if skipRouting == 0:
        if rtSelect == "0":
            processXY(str(rowNum))
        elif rtSelect == "1":
            processA1(str(rowNum))
        elif rtSelect == "2":
            processA2(str(rowNum))
        elif rtSelect == "3":
            processLAMP(str(rowNum))
        elif rtSelect == "4":
            print("")
        else:
            print("### Invalid routing choice. Please follow the instruction.")
            return 0

    if exeSelect == "0":
        print("### Skip simulation!")
    elif exeSelect == "1":
        allsize = rowNum * rowNum
        
        
        if int(simSelect) <=0:
            print("### Invalid simulation cycles. Please follow the instruction")
            return 0        
        commandNow = "sudo ./simulator/build/Garnet_standalone/gem5.debug --debug-flags=GarnetSyntheticTraffic ./simulator/configs/example/garnet_synth_traffic.py --network=garnet2.0 --single-flit --synthetic=taskgraph --num-cpus="+str(allsize)+" --num-dirs="+str(allsize)+" --topology=Mesh_XY --mesh-rows="+str(rowNum)+" --sim-cycles="+str(simSelect)+" --filename=mapRouteResult/result_Mesh8x8_AIR1_result.json"

        commandFinal = ""
        if arSelect == "0":
            commandFinal = commandNow
        elif arSelect == "1":
            commandFinal = commandNow + " --smart --hpcmax="+hpcSelect    
        elif arSelect == "2":
            commandFinal = commandNow + " --arsmart --hpcmax="+hpcSelect + " --configure-time="+configTime
        else:
            print("### Invalid architecture selection. Please follow the instruction")
            return 0
        
        os.system(commandFinal)
        #command = "sudo ./build/Garnet_standalone/gem5.debug --debug-flags=GarnetSyntheticTraffic configs/example/garnet_synth_traffic.py --network=garnet2.0 --num-cpus="+str(allsize)+" --num-dirs="+str(allsize)+" --smart_hpcmax=8 --topology=Mesh_XY "+"--mesh-rows="+size+" --sim-cycles=1000000 --single-flit --synthetic=taskgraph --central --filename="+fullname

    else:
        print("### Invalid simulation selection. Please follow the instruction")
        return 0

def main(argv):
    print("************************************************************")
    print("*                Welcome to NoC Optimization               *")
    print("************************************************************")

    print("### Please choose the mode:")
    print("### 0. Using flags directly; 1. Step by step following instructions; 2. Exit.")
    modeSelect = input("### Your choice: ")
    if modeSelect == "0":
        inputfile = "./inputFile/example.tgff"
        rowNum = "8"
        mapSelect = "0"
        rtSelect = "0"
        vmachine = "0.25"
        archfile = ""            
        exeSelect = "1"
        arSelect = "0"
        simSelect = "100000"
        hpcSelect = "8"
        configTime = "3"

        try:
            opts, args = getopt.getopt(argv,"hi:r:m:t:v:f:e:a:s:p:c:",["ifile=","row=","map=","route=","vmachine=","archfile=","simlution=","arch=","simCyc=","hpcmax=","cfgTime="])
        except getopt.GetoptError:
            print('Error main.py -i <inputfile> -r <row> -m <mapping selection> -t <routing selection> -v <v_machine> -f <architecture description file> -e <simulation selection> -a <architecture selection> -s <simulation cycles> -p <hpc max> -c <configuration time>')
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                print('main.py -i <inputfile> -r <row> -m <mapping selection> -t <routing selection> -v <v_machine> -f <architecture description file> -e <simulation selection> -a <architecture selection> -s <simulation cycles> -p <hpc max> -c <configuration time>')
                sys.exit()
            elif opt in ("-i", "--ifile"):
                inputfile = arg
            elif opt in ("-r", "--row"):
                rowNum = arg
            elif opt in ("-m", "--map"):
                mapSelect = arg
            elif opt in ("-t", "--route"):
                rtSelect = arg
            elif opt in ("-v", "--vmachine"):
                vmachine = arg
            elif opt in ("-f", "--archfile"):
                archfile = arg
            elif opt in ("-e", "--simulation"):
                exeSelect = arg
            elif opt in ("-a", "--arch"):
                arSelect = arg
            elif opt in ("-s", "--simCyc"):
                simSelect = arg
            elif opt in ("-p", "--hpcmax"):
                hpcSelect = arg
            elif opt in ("-c", "--cfgTime"):
                configTime = arg
        
        print("inputfile:",inputfile)
        print("rowNum:",rowNum)
        print("mapSelect:",mapSelect)
        print("rtSelect:",rtSelect)
        print("vmachine:",vmachine)
        print("archfile:",archfile)
        print("exeSelect:",exeSelect)
        print("arSelect:",arSelect)
        print("simSelect:",simSelect)
        print("hpcSelect:",hpcSelect)
        print("configTime:",configTime)

        flagMode(inputfile,rowNum,mapSelect,rtSelect,vmachine,archfile,exeSelect,arSelect,simSelect,hpcSelect,configTime)    
    elif modeSelect == "1":
        instructionMode()
    elif modeSelect == "2":
        exit("### Thanks!")
    else:
        print("### Invalid input. Please follow the instruction. Using the default value (1)")
        instructionMode()
    
    
    while 1:
        print("### Please choose the mode:")
        print("### 1. Step by step following instructions; 2. Exit.")
        modeSelect = input("### Your choice: ")  
        if modeSelect == "1":
            instructionMode()
        elif modeSelect == "2":
            exit("### Thanks!")
        else:
            print("### Invalid input. Please follow the instruction. Using the default value (1)")
            instructionMode()
        
        



if __name__ == "__main__":
    main(sys.argv[1:])