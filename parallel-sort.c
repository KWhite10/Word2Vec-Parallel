#include <mpi.h>
#include <stdio.h>
#include <string.h>




int main(int argc, char* argv[]){
    int np, rank;
    int i,j;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if(rank==0){
        
    }
}




master:

initialize matrices
load job queue
for batch in num_synchronizations:
    for 1 = 1 to 10
        for p in num processes:
            send job to p
    update matrices from processes (add and normalize)
    send updated matrices to all processes
kill all workers
save results


worker:

while true:
    get matrices from master
    for i = i to 10:
        get job (1000 words) from master
        update matrices
    send matrices to master