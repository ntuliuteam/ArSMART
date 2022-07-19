# ArSMART - An integrated tool for software configurable NoC. 
--------------------------------------------------------
- This tool includes:
- An improved SMART NoC simulator suppriting arbitrary-turn transmssion: ArSMART | in ./simulator
- The task mapping and routing co-optimization framework: MARCO | in ./front-end/MARCO
- The multipath parallel transmission routing algorithm: LAMP | in ./front-end/LAMP
--------------------------------------------------------
## Python dependency:  
See in different parts specifically.
--------------------------------------------------------
## To use this program:  
*  1. Compile simulatir according to   
*  2. Put the tgff file in ./inputFile  
*  3. Run main.py using flags or according instructions.  
    * a. flags:  
        * "-i", "--ifile", str, File name (.tgff)
        * -r", "--row", int, The number of rows of the system (1-16 due to the limitation of simulator)
        * "-m", "--map", int, The mapping algorithm you want to apply: 0. Heuristic; 1. Co-optimization(with routing); 2. User specific (already put the mapping result to ./mapRouteResult/result_Mesh8x8_AIR_free.json)
        * "-t", "--route", int, The routing algorithm you want to apply: 0. XY; 1. Arbitrary-1; 2. Arbitrary-2; 3. LAMP; 4. User-specificed (already put the routing result to ./mapRouteResult/result_Mesh8x8+"_AIR1_result.json)
        * "-v", "--vmachine", float, For co-optimization only. The vmachine of the architecture (0-1)
        * "-f", "--archfile", str, For co-optimization only. The architecture description file in ./front-end/MARCO/NoCDescription/  
        * "-e", "--simulation", bool, Run simulation or not
        * "-a", "--arch", int, The architecture you want to use: 0. Traditional NoC 1. SMART NoC 2. ArSMART NoC (Please note that traditional NoC and SMART NoC can only support XY routing)
        * "-s", "--simCyc", int, The number of cycle (>0) you want to simulate
        * "-p", "--hpcmax", int, The hpcmax (the number of hops can be traversed in one cycle) of the architecture (>=1)
        * "-c", "--cfgTime", int, The configuration time for the single router (>=0)
    * b. Step by step:
        * Follow the instructions
        * Enter to use default value
* 4. Examples:
    * Compute the mapping and routing using co-optimization algorithm for ./inputFile/example.tgff with the mesh size of 8x8. Simulation the result using ArSMART with the hpcmax is 3 and configuration time for each router is 3.  

    * python main.py -i ./inputFile/example.tgff -r 8 -m 1 -f ./front-end/MARCO/NoCDescription/example_NoCdescription.txt -e 1 -a 2 -s 10000 -p 8 -c 3
--------------------------------------------------------
## Project Information

Copyright (c) Nanyang Technological University, Singapore.

If you use the tool or adapt the tool in your works or publications, you are required to cite the following reference:
```bib
@article{chen2021arsmart,
  title={ArSMART: An improved SMART NoC design supporting arbitrary-turn transmission},
  author={Chen, Hui and Chen, Peng and Zhou, Jun and Duong, Luan HK and Liu, Weichen},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  volume={41},
  number={5},
  pages={1316--1329},
  year={2021},
  publisher={IEEE}
}
@article{chen2022lamp,
  title={LAMP: Load-bAlanced Multipath Parallel Transmission in Point-to-point NoCs},
  author={Chen, Hui and Chen, Peng and Luo, Xiangzhong and Huai, Shuo and Liu, Weichen},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  year={2022},
  publisher={IEEE}
}
@article{chen2021marco,
  title={MARCO: A High-performance Task M apping a nd R outing Co-optimization Framework for Point-to-Point NoC-based Heterogeneous Computing Systems},
  author={Chen, Hui and Zhang, Zihao and Chen, Peng and Luo, Xiangzhong and Li, Shiqing and Liu, Weichen},
  journal={ACM Transactions on Embedded Computing Systems (TECS)},
  volume={20},
  number={5s},
  pages={1--21},
  year={2021},
  publisher={ACM New York, NY}
}
```
--------------------------------------------------------
**Contributors:**
Hui Chen, Peng Chen, Zihao Zhang, Xiangzhong Luo, Shuo Huai, Jun Zhou, Luan H. K. Duong, Weichen Liu

If you have any comments, questions, or suggestions please create an issue on github or contact us via email.

Hui Chen <hui [DOT] chen [AT] ntu [DOT] edu [DOT] sg>
