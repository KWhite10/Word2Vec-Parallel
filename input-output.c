#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <ctype.h>

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
    char* code;
    char* codelen;
} vocabWord;

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

void learnModelInput(){
    char word[MAX_STRING];
    int i, a;
    FILE* input;
    for (a = 0; a < hash_size; a++){
    vocab_hash[a] = -1;
    }
    input = fopen("input-file.txt", "r");
    if(input == (FILE*)0){
        printf("Download input-file.txt from github\n");
        exit(0);
    }
    vocab_size = 0;
    while(fscanf(input, "%49[a-zA-Z|']%*[^a-zA-Z|']", word) == 1){
        train_words++;
        for(i=0;word[i];i++){ //Make all the letters lowercase
            word[i] = tolower((unsigned char) word[i]);
        }
        
        i = searchCurrentVocab(word);
        if(i==-1){
            a = addtoCurrentVocab(word);
            vocab[a].cn = 1;
        }
        else{
            vocab[i].cn++;
        }
    }
    sortVocab();
    file_size = ftell(input);
    fclose(input);
}
void saveVocab(){
long long i;
    FILE *fo = fopen("holmes-vocab.txt", "wb");
    for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %ld\n", vocab[i].word, vocab[i].cn);
    fclose(fo);
}

int main(int argc, char** argv){
    int i;
    outputFile[0] = 0;
    vocabFile[0] = 0;
    
    vocab = (vocabWord*)calloc(vocab_max_size, sizeof(vocabWord));
    vocab_hash = (int *)calloc(hash_size, sizeof(int));
    expTable = (float *)malloc((EXP_TABLE_SIZE + 1) * sizeof(float));

    for (i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
    //first run generate vocab file
    learnModelInput();
    saveVocab();
}
