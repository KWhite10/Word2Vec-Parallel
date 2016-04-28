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
    for (a = 0; a < vocab_size; a++){ 
		for (b = 0; b < layer1_size; b++){
			hiddenToOutput[a * layer1_size + b] = 0;
		}
	}
	
	
	for (a = 0; a < vocab_size; a++){ 
		for (b = 0; b < layer1_size; b++){
			//The google code I am using from dav github with Tetsuo memory patch v2 seems to use a different line here.
			nextRandom = nextRandom * (unsigned long long)25214903917 + 11;
			inputToHidden[a * layer1_size + b] = (((nextRandom & 0xFFFF) / (float)65536) - 0.5) / layer1_size;
		}
	}
}




int main(int argc, char* argv[]){
    int np, rank;
	MPI_Status status;
    int i,j, p, r;
    int bufferSize = 10; //10 jobs with 1000 words each
	int jobSize = 1000;
	int numSynchronizations = 5;
	int jobBatch=10; 
	int receiveData = -4;
	
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	    
	bufferSize = np - 1;
	
    if(rank==0){
        
		printf("master rank == %d\n", rank);
		int jobBuffer[bufferSize][jobSize]; //Initialize all elements to 0;
		memset(jobBuffer, 0, bufferSize*jobSize*sizeof(int));
		for (i=0; i< bufferSize; i++){
			for(r=0;r<jobSize; r++){	
				jobBuffer[i][r] = i+r;
			}
		}
		
		//produce matrices here and send out first matrix
		initNet();

			for(p=1;p<np;p++){
				MPI_Send(&receiveData, 1, MPI_INT, p, p, MPI_COMM_WORLD);
			}
		for (i=0; i< numSynchronizations; i++){			
				
			for(j=0;j<jobBatch; j++){
				//scan jobs or start from beginning of file and scan jobs 
				//scan jobs equal to proc - 1
				for (p=1; p < np; p++){
					//can be asynchronous
					MPI_Send(&(jobBuffer[p-1][0]), 1000, MPI_INT, p, p, MPI_COMM_WORLD);
				}
			}

			//recv matrices here
			for (r=1; r< np; r++){ 
				printf("rank = %d, synchronizing\n", rank);
				MPI_Recv(&receiveData, 1, MPI_INT, r, r, MPI_COMM_WORLD, &status);
				printf("rank = %d, received %d\n",rank, receiveData);

				//send out new matrices
				MPI_Send(&receiveData, 1, MPI_INT, r, r, MPI_COMM_WORLD);
			}
		}			
		//send termination
		for(p = 1; p<np;p++){
			jobBuffer[p-1][0] = -1;
			printf("rank = %d, synchronizing once more\n", rank);
			MPI_Send(&(jobBuffer[p-1][0]), 1000, MPI_INT, p, p, MPI_COMM_WORLD);
			//recieve final matrix
			MPI_Recv(&receiveData, 1, MPI_INT, p, p, MPI_COMM_WORLD, &status);
			printf("rank = %d, received %d\n",rank, receiveData);
		}	


		//output data
		free(inputToHidden);
		free(hiddenToOutput);	
		
    }
	else{
		int data;
		
		//printf("worker rank == %d\n", rank);
		int workBuffer[jobSize]; //Initialize all elements to 0;
		memset(workBuffer, 0,jobSize*sizeof(int));
		
		expTable = (float *)malloc((EXP_TABLE_SIZE + 1) * sizeof(float));   //same idea as original
		for (i = 0; i < EXP_TABLE_SIZE; i++) {	
			expTable[i] = exp((i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table  e^-6 to e^6
			expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)    (e^x)/(1+e^x)  x: -6 to 6
		}
	
		//Recv first matrices here
		MPI_Recv(&data, 1, MPI_INT, 0, rank, MPI_COMM_WORLD, &status);
		int taskNum = 0;
		printf("Recieved first matrix %d\n", data);
		
		while (1){
		
			
			MPI_Recv(workBuffer, 1000, MPI_INT, 0, rank, MPI_COMM_WORLD, &status);
		
			//printf("process %d received job %d tasknum %d\n", rank, workBuffer[0], taskNum);
		
			//check first element for -1
			if (workBuffer[0] == -1){
					break;
			}
			taskNum++;
			if (taskNum == jobBatch){
			
				printf("rank = %d, synchronize\n", rank);
				data = rank;

				MPI_Send(&data, 1, MPI_INT, 0, rank, MPI_COMM_WORLD);
				taskNum = 0;

				//Recieve new matrice
				MPI_Recv(&data, 1,MPI_INT, 0, rank, MPI_COMM_WORLD, &status);
				printf("rank = %d, done synchronizing\n", rank);
			}	
		}
		data = rank;
		
		
		printf("rank = %d, sending once more\n", rank);
		MPI_Send(&data, 1, MPI_INT, 0, rank, MPI_COMM_WORLD);
		printf("rank = %d, sent\n", rank);
	
		free(expTable);
	}
	
	MPI_Finalize(); 
	return 0;
	
}
