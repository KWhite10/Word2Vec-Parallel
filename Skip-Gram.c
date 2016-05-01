#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <pthread.h>

#define MAX_STRING 50
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int hash_size = 30000000;

typedef struct{
    long cn;
    int* point;
    char* word;
    char* code; //not used
    char* codelen; // not used
} vocabWord;

char vocabFile[MAX_STRING];

//for ease of access I'm making these global for now
int np, rank;
MPI_Status status;



vocabWord* vocab;

int* vocab_hash;
int vocab_max_size = 10000, vocab_size = 0, layer1_size = 100;
int train_words = 0, word_count_actual = 0, file_size = 0, classes = 0, window = 5, min_count = 3, num_threads = 1, min_reduce = 1;
float alpha = 0.025, starting_alpha, sample = 0;
float *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5; //In our case we use negative
const int table_size = 1e8;
int *table;

int VocabCompare(const void* a, const void* b){
     return ((vocabWord *)b)->cn - ((vocabWord *)a)->cn;
}

int getWordHash(char word[]){ //used google hash method
    unsigned int a, hash = 0;
    for (a = 0; a < strlen(word); a++){
        hash = hash * 257 + word[a];
    }
    hash = hash % hash_size;
    return hash;
}

int searchCurrentVocab(char word[]){ //uses linear probing approach for hash table
    unsigned int i;
    unsigned int hash = getWordHash(word);
    for(i = hash; 1; i = (i+1) % hash_size){
        if(vocab_hash[i] == -1){
            return -1;
        }
        if(!strcmp(word, vocab[vocab_hash[i]].word)){
            return vocab_hash[i];
        }
    }
    //shouldn't reach here ever
    return -1;
}



int addtoCurrentVocab(char word[]){
    unsigned int hash;
    unsigned int length = strlen(word) + 1;
    vocab[vocab_size].word = (char*) calloc(length, sizeof(char));
    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn = 0;
    vocab_size++;
    hash = getWordHash(word);
    vocab_hash[hash] = vocab_size - 1;
    return vocab_size - 1;
}

void sortVocab(){
    int a, size;
    unsigned int hash;
    //doesn't use </s>
    qsort(vocab, vocab_size, sizeof(vocabWord), VocabCompare);
    for(a=0;a<hash_size;a++){
        vocab_hash[a] = -1;
    }
    size = vocab_size;
    train_words=0;
    for(a=0;a<size;a++){
        if(vocab[a].cn < min_count){
            vocab_size--;
            free(vocab[a].word);
            vocab[a].word = (char*)0;
        }
        else{
            hash = getWordHash(vocab[a].word);
            while(vocab_hash[hash] != -1){
                vocab_hash[hash] = a;
                train_words += vocab[a].cn;
            }
        }
    }
    for(a=0;a<vocab_size;a++){
        vocab[a].code = (char*) calloc(MAX_CODE_LENGTH, sizeof(char));
        vocab[a].point = (int*) calloc(MAX_CODE_LENGTH, sizeof(int));
    }
}

void readVocab(){
    long a, i;
    char c;
    char word[MAX_STRING];
    FILE* input = fopen("holmes-vocab.txt", "rb");
    
    if(input == (FILE*)0){
        printf("Run input-output.c with input-file.txt first.");
        return;
    }
    for(a=0; a<hash_size; a++){
        vocab_hash[a] = -1;
    }
    vocab_size = 0;
    while (1) {
		// now it seperates ' from letters
        while(fscanf(input, "%49[a-zA-Z|']%*[^a-zA-Z|']", word) == 1){
            train_words++;
        }
        a = addtoCurrentVocab(word);
        fscanf(input, "%ld%c", &vocab[a].cn, &c);
        i++;
    }
    sortVocab();
    input = fopen("input-file.txt", "rb");
    if (input == (FILE*)0) {
        printf("Use github to get input-file.txt!\n");
        exit(1);
    }
    fseek(input, 0, SEEK_END);//Used for pthreads
    file_size = ftell(input);
    fclose(input);
}

void InitNet() {// google code
    long long a, b;
    a = posix_memalign((void **) &syn0, 128, (long long) vocab_size * layer1_size * sizeof(float));
    if (syn0 == NULL) {
		printf("Memory allocation failed\n"); 
		exit(1);
	}
    if (hs) { //Not used
        a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(float));
        if (syn1 == NULL) {
			printf("Memory allocation failed\n"); 
			exit(1);
		}
        for (b = 0; b < layer1_size; b++) 
			for (a = 0; a < vocab_size; a++)
           		syn1[a * layer1_size + b] = 0;
    }
    if (negative>0) { //Used
        a = posix_memalign((void **) &syn1neg, 128, (long long) vocab_size * layer1_size * sizeof(float));
        if (syn1neg == NULL) {
			printf("Memory allocation failed\n"); 
			exit(1);
		}
        for (b = 0; b < layer1_size; b++) 
			for (a = 0; a < vocab_size; a++)
            	syn1neg[a * layer1_size + b] = 0;
    }
    for (b = 0; b < layer1_size; b++) 
		for (a = 0; a < vocab_size; a++) //Code found from Dav gihub who got it from google feel free to change this two the other rand method
        	syn0[a * layer1_size + b] = (rand() / (float)RAND_MAX - 0.5) / layer1_size;
    //CreateBinaryTree(); Ignore for now
}

void InitUnigramTable() { //google code
  int a, i;
  long long train_words_pow = 0;
  float d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  if (table == NULL) {
    fprintf(stderr, "cannot allocate memory for the table\n");
    exit(1);
  }
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / (float)train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (float)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / (float)train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}


void SkipGram(long long* sentence){
	float *h = (float *)calloc(layer1_size, sizeof(float));
	float *hE = (float *)calloc(layer1_size, sizeof(float));		
	int label, d, c, cw, target, sampleOffset;
	float f, g;
	unsigned long long next_random = 1;
	float alpha = 0.05;
	int sentLength = MAX_SENTENCE_LENGTH;
	float error;
	float totalError = 0;
	long long wordId;

	int i, j;
	for(i=0;i<sentLength;i++){
		wordId = sentence[i];
		if(wordId == -1){
			continue;
		}
		for (c = 0; c < layer1_size; c++) h[c] = 0;
    	for (c = 0; c < layer1_size; c++) hE[c] = 0;
		int range = window / 2;
	
		for (j=i-range; j<= i+range; j++){

            if (j < 0 || j>= sentLength || j == i){ //skip elements in window outside sentence
                continue;
			}
			for (c = 0; c < layer1_size; c++){
                h[c] += 0;
            }
			for (d = 0; d < negative + 1; d++) {
          		if (d == 0) {
					target = wordId;
					label = 1;
				}
				else{
					next_random = next_random * (unsigned long long)25214903917 + 11;
           			target = table[(next_random >> 16) % table_size];//unigram table
           			if (target == 0) target = next_random % (vocab_size - 1) + 1;
           			if (target == wordId) continue;
           			label = 0;
				}
				sampleOffset = target * layer1_size;
				f = 0;
				for (c = 0; c < layer1_size; c++){ 
					f += syn0[c + wordId*layer1_size] * syn1neg[c + sampleOffset];
				}
				error = 0;
	            if (f > MAX_EXP){
    	            error = label -1;
        	        g = (error) * alpha;
            	    totalError = totalError + fabs(error);
                }
           	    else if (f < -MAX_EXP){ 
            	    error = label -0;
            	    g = (error) * alpha;
           	        totalError = totalError + fabs(error);
	           	}
    	        else{
       				error = label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
            	    g = (error) * alpha;  //<0 if negative, >0 if positive (-1,1)
              	    totalError = totalError + error;
               	}
				for (c = 0; c < layer1_size; c++) hE[c] += g * syn1neg[c + sampleOffset];
          		for (c = 0; c < layer1_size; c++) syn1neg[c + sampleOffset] += g * syn0[c + wordId*layer1_size];
			}
			for (c = 0; c < layer1_size; c++) syn0[c + wordId * layer1_size] += hE[c];
			// place error here for each word update error
		}
		//place error here for each window update error
	}
	//here for each sentence.
	printf("error = %f\n", totalError); //total loss per update. Could keep accumulating until end of sentence 
}


void CBOW(long long *sentence){  //reworded/slightly rewritten from google code. Just pass sentence array of numbers
	float *h = (float *)calloc(layer1_size, sizeof(float));
	float *hE = (float *)calloc(layer1_size, sizeof(float));		
	int label, d, c, cw, target, sampleOffset;
	float f, g;
	unsigned long long next_random = 1;
	float alpha = 0.05;
	int sentLength = MAX_SENTENCE_LENGTH;	
	float error;
	float totalError = 0;
	
	int i;
	for (i=0; i< sentLength; i++){
		
		long long wordId = sentence[i];
		if (wordId == -1){
			continue;
		}
		
		int range = window/2;
		cw = 0;
		int j;
		//for each context
		for (j=i-range; j<= i+range; j++){
		
			if (j < 0 || j>= sentLength){ //skip elements in window outside sentence
				continue;
			}
			for (c = 0; c < layer1_size; c++){ 
				h[c] += inputToHidden[c + wordId* layer1_size]; //x*w_input = h
				
			}	
				
			cw++;
		}
		if (cw){
			for (c = 0; c < layer1_size; c++){
				h[c] /= cw;  // h/C
			
			}
			for (d = 0; d < negative + 1; d++) {
				
				if (d == 0) {
					target = wordId;
					label = 1;
				}else {
					next_random = next_random * (unsigned long long)25214903917 + 11;
					target = table[(next_random >> 16) % table_size];
					if (target == 0) target = next_random % (vocab_size - 1) + 1;
					if (target == wordId) continue;
					label = 0;
				}
				sampleOffset = target * layer1_size;
			
				f = 0;
			
				for (c = 0; c < layer1_size; c++){
				
					f += h[c]*hiddenToOutput[c + sampleOffset];
				
				}
				error = 0;
				if (f > MAX_EXP){ 
					error = label -1;
					g = (error) * alpha;
					totalError = totalError + fabs(error);
				}
				else if (f < -MAX_EXP){ 
					error = label -0;
					g = (error) * alpha;
					totalError = totalError + fabs(error);
				}
				else{
					error = label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
					g = (error) * alpha;  //<0 if negative, >0 if positive (-1,1)
					totalError = totalError + error;
				}		
				for (c = 0; c < layer1_size; c++){
				
					hE[c] += g * hiddenToOutput[c + sampleOffset];
				
				}
				for (c = 0; c < layer1_size; c++){
				
					hiddenToOutput[c + sampleOffset] += g * h[c];
				
				}
			}
			printf("error = %f\n", totalError); //total loss per update. Could keep accumulating until end of sentence 
			totalError = 0;
			
			for (j=i-range; j<= i+range; j++){
		
				if (j < 0 || j>= sentLength){ //skip elements in window outside sentence
					continue;
				}
				long long contextWordId = *(sentence + j);  //each context word
				if (contextWordId == -1) continue;
				
				for (c = 0; c < layer1_size; c++) {
				
					inputToHidden[c + contextWordId * layer1_size] += hE[c];
				
				}
			}
		}
	}
	free(h);
	free(hE);
}




//The hulk of MPI code goes here. From this function after we pass the data to
//other processors we can have them start threads before updating the matrices.
//For now I'll just implement MPI, but adding pthreads is a consideration.
void trainModelParallelSkipGram(){
	long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
  	long long word_count = 0, last_word_count = 0;
	long long l1, l2, c, target, label;
	unsigned long long next_random = (long long)2;
  	float f, g;
	FILE* fi;
	int i,j,p,r;
	int bufferSize = np -1; //numb processors-1 jobs with 1000 words each
    int jobSize = 1000;
    int numSynchronizations = 5;
    int jobBatch=10;
	//the parallel-outline will help fill in this part, I tested the communication and it works great
	//Only difference is sending two matrices
	if(rank == 0){
		int jobBuffer[bufferSize][jobSize]; //Initialize all elements to 0;
        memset(jobBuffer, 0, bufferSize*jobSize*sizeof(int));

		//Send out the two matrices from processor 0
			//for(p=1;p<np;p++){
				//MPI_Send
                //MPI_Send(&receiveData, 1, MPI_FLOAT, p, p, MPI_COMM_WORLD);
            //}

		float *neu1 = (float *)calloc(layer1_size, sizeof(float));
  		float *neu1e = (float *)calloc(layer1_size, sizeof(float));
  		fi = fopen("input-file.txt", "rb");
  		if (fi == (FILE*)0) {
    		printf("no such file \n");
    		exit(1);
  		}		
		for (i=0; i< numSynchronizations; i++){
   
            for(j=0;j<jobBatch; j++){
				//scan word by word until you reach 1000 words or end of file
				//if end of file then restart at beginning of file
				//do this for each other processor
				for (p=1; p < np; p++){
                    //can be asynchronous
                    MPI_Send(&(jobBuffer[p-1][0]), 1000, MPI_INT, p, p, MPI_COMM_WORLD);
                }
			}	
		}
	}
	else{
		
	}


	if(rank == 0){
		fclose(fi);
	}

}




//The hulk of MPI code goes here. From this function after we pass the data to
//other processors we can have them start threads before updating the matrices.
//For now I'll just implement MPI, but adding pthreads is a consideration.
void trainModelParallelCBOW(){
	long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
  	long long word_count = 0, last_word_count = 0;
	long long l1, l2, c, target, label;
	unsigned long long next_random = (long long)2;
  	float f, g;
	FILE* fi;
	int i,j,p,r;
	int bufferSize = np -1; //numb processors-1 jobs with 1000 words each
    int jobSize = 1000;
    int numSynchronizations = 5;
    int jobBatch=10;
	//the parallel-outline will help fill in this part, I tested the communication and it works great
	//Only difference is sending two matrices
	if(rank == 0){
		int jobBuffer[bufferSize][jobSize]; //Initialize all elements to 0;
        memset(jobBuffer, 0, bufferSize*jobSize*sizeof(int));

		//Send out the two matrices from processor 0
			//for(p=1;p<np;p++){
				//MPI_Send
                //MPI_Send(&receiveData, 1, MPI_FLOAT, p, p, MPI_COMM_WORLD);
            //}

		
  		fi = fopen("input-file.txt", "rb");
  		if (fi == (FILE*)0) {
    		printf("no such file \n");
    		exit(1);
  		}		
		for (i=0; i< numSynchronizations; i++){
   
            for(j=0;j<jobBatch; j++){
				//scan word by word until you reach 1000 words or end of file
				//if end of file then restart at beginning of file
				//do this for each other processor
				for (p=1; p < np; p++){
                    //can be asynchronous
                    MPI_Send(&(jobBuffer[p-1][0]), 1000, MPI_INT, p, p, MPI_COMM_WORLD);
                }
			}	
		}
	}
	else{
		//add parallel-outline related code
		//call CBOW with received array of words in sentence (array of longs)
		
	}


	if(rank == 0){
		fclose(fi);
	}

}





void trainSkipGram(){
	if(rank == 0){
    	readVocab();
    	InitNet();
		if (negative > 0){ 
			InitUnigramTable();
		}
	}
	trainModelParallelSkipGram();
	
}

void trainCBOW(){
	if(rank == 0){
    	readVocab();
    	InitNet();
		if (negative > 0){ 
			InitUnigramTable();
		}
	}
	trainModelParallelCBOW();
	
}



int main(int argc, char** argv){
	int np, rank;
	MPI_Status status;
    int i;

	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    vocab = (vocabWord*)calloc(vocab_max_size, sizeof(vocabWord*));
    vocab_hash = (int *)calloc(hash_size, sizeof(int));
    expTable = (float *)malloc((EXP_TABLE_SIZE + 1) * sizeof(float));
    if (expTable == NULL) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    for (i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
    trainSkipGram();
	MPI_Finalize(); 
	return 0;
}
