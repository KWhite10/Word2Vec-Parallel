#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>



#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6

float *inputToHidden;
float *hiddenToOutput;

float *expTable;


char **vocab;


int vocabSize= 14;
int layerSize = 5;
int window = 5;
int negative = 5;

int tableSize = 20;
int *table;


//default selection table. Need to decide on how to do this
void InitUnigramTable() {


	table = (int *)malloc(tableSize*sizeof(int));
	int i;
	for (i=0; i<vocabSize; i++){
	
		table[i] = i;
	
	}
	for (i=vocabSize; i<tableSize; i++){ //wrap back around
	
		table[i] = i-vocabSize;
	
	}

}

//just based on position in vocab. Need to decide on this
int getHash(char *word){
	int i;
	for (i=0; i<vocabSize; i++){
		
		if (strcmp(word, *(vocab+i)) ==0){
			return i;
		}
	}
	return -1;
}

void initNet(){ //same technique/code as word2vec source

	long long a,b;
	unsigned long long nextRandom = 1;
	
	//initialize inputToHidden matrix 
	a = posix_memalign((void **)&inputToHidden, 128, (long long)vocabSize * layerSize * sizeof(float));
	
	//intialize hiddenToOutput matrix
	a = posix_memalign((void **)&hiddenToOutput, 128, (long long)vocabSize * layerSize * sizeof(float));
	
    if (hiddenToOutput == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocabSize; a++) for (b = 0; b < layerSize; b++){
		hiddenToOutput[a * layerSize + b] = 0;
	}
	
	
	for (a = 0; a < vocabSize; a++) for (b = 0; b < layerSize; b++) {
		nextRandom = nextRandom * (unsigned long long)25214903917 + 11;
		inputToHidden[a * layerSize + b] = (((nextRandom & 0xFFFF) / (float)65536) - 0.5) / layerSize;
	}

	
	
}



void CBOL(char **sentence, int sentLength){  //unparallelized rough version, mainly from original word2vec code

	float *h = (float *)calloc(layerSize, sizeof(float));
	float *hE = (float *)calloc(layerSize, sizeof(float));		
	int label, d, c, cw, target, sampleOffset;
	float f, g;
	unsigned long long next_random = 1;
	float alpha = 0.05;
	
	int i;
	for (i=0; i< sentLength; i++){
		char *word = *(sentence + i);
		int wordId = getHash(word);
		printf("wordHash = %d\n", wordId);
		printf("\n");
		printf("\n");
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
			
		
			for (c = 0; c < layerSize; c++){ 
				h[c] += inputToHidden[c + wordId* layerSize]; //x*w_input = h
				printf("h[%d] = %f\n", c, h[c]);
			}	
				
			cw++;
		}
		
		printf("\n");
		
		if (cw){
			for (c = 0; c < layerSize; c++){
				h[c] /= cw;  // h/C
				printf("h[%d] = %f\n", c, h[c]);
			}
			for (d = 0; d < negative + 1; d++) {
				
				if (d == 0) {
					target = wordId;
					label = 1;
				}else {
					next_random = next_random * (unsigned long long)25214903917 + 11;
					target = table[(next_random >> 16) % tableSize];
					if (target == 0) target = next_random % (vocabSize - 1) + 1;
					if (target == wordId) continue;
					label = 0;
				}
				
				printf("target = %d\n", target);
				
				sampleOffset = target * layerSize;
				printf("sampleOffset = %d\n", sampleOffset);
				f = 0;
				printf("\n");
				for (c = 0; c < layerSize; c++){
					printf("h[%d] = %f, hiddenToOutput[%d + %d] = %f\n", c, h[c], c,sampleOffset, hiddenToOutput[c+sampleOffset]);
					printf("h[c]*hiddenToOutput[] = %f\n", h[c]*hiddenToOutput[c + sampleOffset]);
					f += h[c]*hiddenToOutput[c + sampleOffset];
					printf("f = %f\n", f);
				}
				
				if (f > MAX_EXP) g = (label - 1) * alpha;
				else if (f < -MAX_EXP) g = (label - 0) * alpha;
				else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;  //<0 if negative, >0 if positive (-1,1)
				
				printf("\n");
				printf("expTable[x] = %f\n", expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))] );
				printf("where x  = %d\n", (int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2)));
				printf("\n");
				printf("label = %d\n", label);
				
				printf("g = %f\n", g);
				printf("\n");
				for (c = 0; c < layerSize; c++){
					printf("hiddenToOutput[%d + %d] = %f\n", c, sampleOffset,  hiddenToOutput[c + sampleOffset]);
					hE[c] += g * hiddenToOutput[c + sampleOffset];
					printf("hE[%d] = %f\n", c, hE[c]);
				}
				printf("\n");
				
				for (c = 0; c < layerSize; c++){
					printf("h[%d] = %f\n", c, h[c]);
					hiddenToOutput[c + sampleOffset] += g * h[c];
					printf("g*h[c] = %f\n",g * h[c] );
					printf("hiddenToOutput[%d + %d] = %f\n",c, sampleOffset, hiddenToOutput[c + sampleOffset]);
				}
			}
			for (j=i-range; j<= i+range; j++){
		
				if (j < 0 || j>= sentLength){ //skip elements in window outside sentence
					continue;
				}
				char *contextWord = *(sentence + j);  //each context word
				printf("\n");
				int contextWordId = getHash(contextWord);
				printf("contextWordId = %d\n", contextWordId);
				
				if (contextWordId == -1) continue;
				
				for (c = 0; c < layerSize; c++) {
					printf("inputToHidden[%d + %d] = %f, hE[%d] = %f\n", c, contextWordId*layerSize,inputToHidden[c + contextWordId * layerSize], c, hE[c] );
					inputToHidden[c + contextWordId * layerSize] += hE[c];
					printf("inputToHidden[%d + %d] = %f\n", c, contextWordId*layerSize, inputToHidden[c + contextWordId * layerSize]);
				}
			}
		}
	}
	free(h);
	free(hE);
	
	
}



int main(){

	printf("START%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
	printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
	printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
	printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");

	int i;
	expTable = (float *)malloc((EXP_TABLE_SIZE + 1) * sizeof(float));   //same idea as original
	for (i = 0; i < EXP_TABLE_SIZE; i++) {
		expTable[i] = exp((i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table  e^-6 to e^6
		expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)    (e^x)/(1+e^x)  x: -6 to 6
	}


	char *sentence1[] = {"the", "cat",  "in",  "the", "hat"};
	int sentenceLength = 5;
	char *sentence2[] = {"an" ,"apple", "is", "a", "fruit"};
	
	char *sentence3[] = {"cat", "hat"};
	
	
	char* vocabulary[] = {"the", "dog", "cat", "bird", "in", "and", "to", "hat", "house", "apple", "fruit", "is", "a", "an"};
	vocab = vocabulary;
	
	InitUnigramTable();
	initNet();
	CBOL(sentence1, sentenceLength);  //sentenceLength can be fixed, regardless of whether words from a sentence
	CBOL(sentence2, sentenceLength);
	CBOL(sentence3, 2);
	//CBOL(sentence1, sentenceLength);
	//CBOL(sentence3, 2);
	
	
	
	
	printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
	printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
	printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
	printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
	
	int j;
	printf("inputToHidden\n");
	for (i=0; i< vocabSize; i++){
		for (j=0; j< layerSize; j++){
		
			printf(" %f ", inputToHidden[i + j]);
			
		}
		printf("\n");
		
	}
	
	printf("hiddenToOutput\n");
	for (i=0; i< vocabSize; i++){
		for (j=0; j< layerSize; j++){
		
			printf(" %f ", hiddenToOutput[i + j]);
			
		}
		printf("\n");
		
	}
	
	
	
	
	
	free(expTable);
	free(inputToHidden);
	free(hiddenToOutput);
	
	return 0;
}