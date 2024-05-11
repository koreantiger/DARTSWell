# DARTSWell
Source code for decoupled-velocity engine used for coupled wellbore and reservoir. 
Decoupled velocity engine, initialize the matrices in CSR format based on mesh info (connection list).
Jacobian matrix has a multi-level structure. based on connection and node. where each velocity is written in connection and other unknowns are cell centers. next, each connection could be part of the connection between two well blocks or two reservoir blocks. based on each the momentum equation is different.
The following paper explains in detail the matrix structure and Jacobian assembly for this code: 
https://www.sciencedirect.com/science/article/pii/S2949891023005134?via%3Dihub


