# DARTSWell
DARTSWell simulation framework for modeling coupled wellbore and reservoir in the subsurface. 

The framework is based on the Decoupled velocity engine, where velocity is another degree of freedom written at each interface between two nodes  and initializes the matrices in CSR format based on mesh info (connection list). 
The decoupled velocity engine class has several methods the major important methods are: 
* Unstructured mesh - it initializes the Jacobian matrix based on an abstract connection list in CSR format
* Jacobian assembly - Assemble the Jacobian matrix based on node and connection. Each connection could belong to the wellbore domain or reservoir domain based on which the equation could differ.
 

Jacobian matrix has a multi-level structure. based on connection and node. where each velocity is written in connection and other unknowns are cell centers. next, each connection could be part of the connection between two well blocks or two reservoir blocks. based on each the momentum equation could differ.

The code provides advanced solvers for several target applications, including those used for coupled wellbore and reservoir.

The following paper explains in detail the matrix structure and Jacobian assembly and the application for this code: 
https://www.sciencedirect.com/science/article/pii/S2949891023005134?via%3Dihub


