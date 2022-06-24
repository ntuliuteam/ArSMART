# MARCO
This is the co-optimization algorithm used to generate mapping and routing solution for a given task graph. More details about this algorithm can be found in:  

--------------------------------------------------------

## Python dependency:  
See in requirements.txt

---------------------------

## To use this program:  
*  1. Generate NoC description (example: N12_autocor_Mesh8x8_NoCdescription.txt)   
*  2. Put the task graph description file in the folder (example: N12_autocor.tgff)  
*  3. Adjust the hyper-parameter
*  4. python MARCO.py -i <inputfile> -r <row> -o <outputfile> -c <configurefile> -v <v_machine>'

## Example:  
*  python MARCO.py -i ./inputFile/example.tgff -r 8 -m 1 -f ./front-end/MARCO/NoCDescription/example_NoCdescription.txt -e 1 -a 2 -s 10000 -p 8 -c 3
