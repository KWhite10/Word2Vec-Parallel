#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>

#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_STRING 50
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40


typedef struct{
    long cn;
    int* point;
    char* word;
    char* code;
    char* codelen;
} vocabWord;

char outputFile[MAX_STRING];
char vocabFile[MAX_STRING];
vocabWord* vocab;

const int hash_size = 30000000;

float *inputToHidden;
float *hiddenToOutput;

char outputFile[MAX_STRING];
char vocabFile[MAX_STRING];
vocabWord* vocab;
int binary = 0, cbow = 0, debug_mode = 2, window = 5, min_count = 3, num_threads = 1, min_reduce = 1;
int* vocab_hash;
int vocab_max_size = 10000, vocab_size = 0, layer1_size = 100;
int train_words = 0, word_count_actual = 0, file_size = 0, classes = 0;
float alpha = 0.025, starting_alpha, sample = 0;
float *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 1, negative = 0;
const int table_size = 1e8;
int *table;



void initNet(){ //same technique/code as word2vec source

	long long a,b;
	unsigned long long nextRandom = 1;
	
	//initialize inputToHidden matrix 
	a = posix_memalign((void **)&inputToHidden, 128, (long long)vocab_size * layer1_size * sizeof(float));
	
	//intialize hiddenToOutput matrix
	a = posix_memalign((void **)&hiddenToOutput, 128, (long long)vocab_size * layer1_size * sizeof(float));
	
    if (hiddenToOutput == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++){
		hiddenToOutput[a * layer1_size + b] = 0;
	}
	
	
	for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
		nextRandom = nextRandom * (unsigned long long)25214903917 + 11;
		inputToHidden[a * layer1_size + b] = (((nextRandom & 0xFFFF) / (float)65536) - 0.5) / layer1_size;
	}
}




int main(int argc, char* argv[]){
    int np, rank;
    int i,j, p;
    int numJobs = 100;
	int job;
	int numSynchronizations = 5;
	int tasksPerBatch=10; 
	int termination;
	int terminationCount=1;
	int lastProcessRunning;
	int taskNum=0;
	
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if(rank==0){
        
		printf("master rank == %d\n", rank);
		
		initNet();
		
		int* jobQueueTest = (int*)malloc(numJobs*sizeof(int)); //dummy job queue
		for (i=0; i< numJobs; i++){
			*(jobQueueTest + i) = i;
		}
		job=0;
		
		
		for (i=0; i< numSynchronizations; i++){
			
			for (j=0; j< tasksPerBatch; j++ ){
			
				for (p=1; p< np; p++){
				
					if (job < numJobs){
						MPI_Send(jobQueueTest + job, 1, MPI_INT, p, p, MPI_COMM_WORLD);
						job++;
					}
					else{  //jobs are done
						printf("jobs are done\n");
						
						if (terminationCount < np){	
							termination = -1;
							MPI_Send(&termination, 1, MPI_INT, p, p, MPI_COMM_WORLD);	
							terminationCount++;
						}
						else{ //finished sending termination signals
							
							j = tasksPerBatch;
							i = numSynchronizations;
							break;
						
						}
						
					}
				}
				
			}
			if (i == numSynchronizations -1 && terminationCount < np){ //still not done, need more synchronizations
					
				numSynchronizations = numSynchronizations+1;
			}
			
		}
		
		
	
		free(inputToHidden);
		free(hiddenToOutput);	
		
    }
	else{
		
		MPI_Status status;
		printf("worker rank == %d\n", rank);
		
		expTable = (float *)malloc((EXP_TABLE_SIZE + 1) * sizeof(float));   //same idea as original
		for (i = 0; i < EXP_TABLE_SIZE; i++) {	
			expTable[i] = exp((i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table  e^-6 to e^6
			expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)    (e^x)/(1+e^x)  x: -6 to 6
		}
	
	
		
		
		job = 0;
		
		while (job != -1){
		
			//for (i = 0; i< tasksPerBatch; i++){
			
			MPI_Recv(&job, 1, MPI_INT, 0, rank, MPI_COMM_WORLD, &status);
			
			printf("process %d received job %d\n", rank, job);
			if (job == -1){
					break;
			}
			
			
			taskNum++;
			if (taskNum == tasksPerBatch){
			
				printf("synchronize\n");
				taskNum = 0;
			}
	
		}
		
	
	
	
	
	
		free(expTable);
	}
	
	MPI_Finalize(); 
	return 0;
	
}
