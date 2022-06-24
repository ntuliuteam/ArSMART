# LAMP  
This is the algorithm to generate the routing after data splitting. Please read paper: https://ieeexplore.ieee.org/document/9709895

------------------------------------  
## Python dependency:  
* See in requirements.txt  
---------------------------
## To use this program:  
* 1. Put the json file for task graph with mapping information  
* 2. python LAMP.py -i <inputfile> -r <row> -o <outputfile>
* 3. The result is saved in the ./mapRouteResult/ folder

## Example:
* python ./front-end/LAMP/LAMP.py -i ./mapRouteResult/result_Mesh8x8_AIR1_free.json -r 8 -o ./mapRouteResult/result_Mesh8x8_AIR1_result.json 

