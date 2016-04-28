#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

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

char vocabFile[MAX_STRING];

vocabWord* vocab;

int* vocab_hash;
int vocab_max_size = 10000, vocab_size = 0, layer1_size = 100;
int train_words = 0, word_count_actual = 0, file_size = 0, classes = 0, window = 5, min_count = 3, num_threads = 1, min_reduce = 1;
float alpha = 0.025, starting_alpha, sample = 0;
float *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 1, negative = 0;
const int table_size = 1e8;
int *table;


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
        return 0;
    }
    for(a=0; a<vocab_hash_size; a++){
        vocab_hash[a] = -1;
    }
    vocab_size = 0;
    while (1) {
        while(fscanf(input, "%49[a-zA-Z']%*[^a-zA-Z']", word) == 1){
            train_words++;
        }
        a = addtoCurrentVocab(word);
        fscanf(input, "%lld%c", &vocab[a].cn, &c);
        i++;
    }
    sortVocab();
    input = fopen("input-file.txt", "rb");
    if (input == (FILE*)0) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    fseek(input, 0, SEEK_END);
    file_size = ftell(input);
    fclose(input);
}

void CreateBinaryTree() {
    long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
    char code[MAX_CODE_LENGTH];
    long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
    long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
    long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
    for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
    for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
    pos1 = vocab_size - 1;
    pos2 = vocab_size;
    // Following algorithm constructs the Huffman tree by adding one node at a time
    for (a = 0; a < vocab_size - 1; a++) {
        // First, find two smallest nodes 'min1, min2'
        if (pos1 >= 0) {
            if (count[pos1] < count[pos2]) {
                min1i = pos1;
                pos1--;
            } else {
                min1i = pos2;
                pos2++;
            }
        } else {
            min1i = pos2;
            pos2++;
        }
        if (pos1 >= 0) {
            if (count[pos1] < count[pos2]) {
                min2i = pos1;
                pos1--;
            } else {
                min2i = pos2;
                pos2++;
            }
        } else {
            min2i = pos2;
            pos2++;
        }
        count[vocab_size + a] = count[min1i] + count[min2i];
        parent_node[min1i] = vocab_size + a;
        parent_node[min2i] = vocab_size + a;
        binary[min2i] = 1;
    }
    // Now assign binary code to each vocabulary word
    for (a = 0; a < vocab_size; a++) {
        b = a;
        i = 0;
        while (1) {
            code[i] = binary[b];
            point[i] = b;
            i++;
            b = parent_node[b];
            if (b == vocab_size * 2 - 2) break;
        }
        vocab[a].codelen = i;
        vocab[a].point[0] = vocab_size - 2;
        for (b = 0; b < i; b++) {
            vocab[a].code[i - b - 1] = code[b];
            vocab[a].point[i - b] = point[b] - vocab_size;
        }
    }
    free(count);
    free(binary);
    free(parent_node);
}

void InitNet() {
    long long a, b;
    a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(float));
    if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    if (hs) {
        a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(float));
        if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
        for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
            syn1[a * layer1_size + b] = 0;
    }
    if (negative>0) {
        a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(float));
        if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
        for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
            syn1neg[a * layer1_size + b] = 0;
    }
    for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
        syn0[a * layer1_size + b] = (rand() / (float)RAND_MAX - 0.5) / layer1_size;
    CreateBinaryTree();
}

void trainSkipGram(){
    readVocab();
    initNet();
}

int main(int argc, char** argv){
    int i;
    vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
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
}