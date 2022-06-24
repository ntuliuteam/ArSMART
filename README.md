# NoC-optimization
This is a integrated tool including NoC optimization mapping/routing algorithms and simulator. 
--------------------------------------------------------
## Python dependency:  
> See in different parts specifically.
---------------------------
# To use this program:  
> 1. Compile simulatir according to   
> 2. Put the tgff file in ./inputFile  
> 3. Run main.py using flags or according instructions.  
    > a. flags:  
        > "-i", "--ifile", str, File name (.tgff)
        > -r", "--row", int, The number of rows of the system (1-16 due to the limitation of simulator)
        > "-m", "--map", int, The mapping algorithm you want to apply: 0. Heuristic; 1. Co-optimization(with routing); 2. User specific (already put the mapping result to ./mapRouteResult/result_Mesh8x8_AIR_free.json)
        > "-t", "--route", int, The routing algorithm you want to apply: 0. XY; 1. Arbitrary-1; 2. Arbitrary-2; 3. LAMP; 4. User-specificed (already put the routing result to ./mapRouteResult/result_Mesh8x8+"_AIR1_result.json)
        > "-v", "--vmachine", float, For co-optimization only. The vmachine of the architecture (0-1)
        > "-f", "--archfile", str, For co-optimization only. The architecture description file in ./front-end/MARCO/NoCDescription/  
        > "-e", "--simulation", bool, Run simulation or not
        > "-a", "--arch", int, The architecture you want to use: 0. Traditional NoC 1. SMART NoC 2. ArSMART NoC (Please note that traditional NoC and SMART NoC can only support XY routing)
        > "-s", "--simCyc", int, The number of cycle (>0) you want to simulate
        > "-p", "--hpcmax", int, The hpcmax (the number of hops can be traversed in one cycle) of the architecture (>=1)
        > "-c", "--cfgTime", int, The configuration time for the single router (>=0)
    > b. Step by step:
        > Follow the instructions
        > Input 
> 4. Examples:
    > Compute the mapping and routing using co-optimization algorithm for ./inputFile/example.tgff with the mesh size of 8x8. Simulation the result using ArSMART with the hpcmax is 3 and configuration time for each router is 3.  

    python main.py -i ./inputFile/example.tgff -r 8 -m 1 -f ./front-end/MARCO/NoCDescription/example_NoCdescription.txt -e 1 -a 2 -s 10000 -p 8 -c 3