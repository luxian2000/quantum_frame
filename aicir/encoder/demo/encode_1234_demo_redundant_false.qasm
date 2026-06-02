OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ry(pi/2) q[0];
x q[0];
cry(pi/2) q[0],q[1];
x q[0];
cry(pi/2) q[0],q[1];
