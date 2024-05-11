# DARTSWell
DARTSWell simulation framework for modeling coupled wellbore and reservoir in the subsurface. 
The code is written for general formulation that can capture multi-physics phenomena in the subsurface. Some of the applications could be
* geothermal reservoirs,
* carbon sequestration,
* Reactive transport.


The framework is based on the Decoupled velocity engine, where velocity is another degree of freedom written at each interface between two nodes  and initializes the matrices in CSR format based on mesh info (connection list). 

The decoupled velocity engine class has several methods the major important methods are: 
* init_jacobian_structure - it initializes the Jacobian matrix based on CSR format from mesh info (connection list)
* Jacobian assembly - Assemble the Jacobian matrix based on node and connection. 
 

Since the initialization is based on the abstract connection list, it allows the engine to work with unstructured mesh. Jacobian matrix has a multi-level structure. based on connection and node. where each velocity is written in connection and other unknowns are cell centers. Next, each connection could be part of the connection between two well blocks or two reservoir blocks. based on each the momentum equation could differ.


The following paper explains in details governing equations, the matrix structure and Jacobian assembly and the application for this code: 
https://www.sciencedirect.com/science/article/pii/S2949891023005134?via%3Dihub


