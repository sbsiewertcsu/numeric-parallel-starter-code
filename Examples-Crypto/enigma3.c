#include "stdio.h"
#include "unistd.h"

// Enigma3
//
// Simple C implementation to automate the 3 rotor enigma based upon the Paper Enigma
// Machine by Michael Koss found at http://mckoss.com/Crypto/Enigma.htm
//
// Will make on Linux, Windows cygwin, or MacOS
//

#define ALPHA_SIZE 26
#define ALPHA_BASE (int)('A')

char inputFile[80]="Unspecified";
char outFile[80]="Unspecified";
char demoString[]="QMJIDO MZWZJFJR";
char demoEncode[]="ENIGMA REVEALED";
char plainTextLine[256];
char cipherTextLine[256];

typedef struct
{
   char leftChar;
   char rightChar;
} cipherPair_t;

typedef struct
{
   int inPos;
   int outPos;
} refMap_t;

cipherPair_t rotorOne[]={
   {'A', 'E'}, {'B', 'K'}, {'C', 'M'}, {'D', 'F'},
   {'E', 'L'}, {'F', 'G'}, {'G', 'D'}, {'H', 'Q'},
   {'I', 'V'}, {'J', 'Z'}, {'K', 'N'}, {'L', 'T'},
   {'M', 'O'}, {'N', 'W'}, {'O', 'Y'}, {'P', 'H'},
   {'Q', 'X'}, {'R', 'U'}, {'S', 'S'}, {'T', 'P'},
   {'U', 'A'}, {'V', 'I'}, {'W', 'B'}, {'X', 'R'},
   {'Y', 'C'}, {'Z', 'J'}
};

cipherPair_t rotorTwo[]={
   {'A', 'A'}, {'B', 'J'}, {'C', 'D'}, {'D', 'K'},
   {'E', 'S'}, {'F', 'I'}, {'G', 'R'}, {'H', 'U'},
   {'I', 'X'}, {'J', 'B'}, {'K', 'L'}, {'L', 'H'},
   {'M', 'W'}, {'N', 'T'}, {'O', 'M'}, {'P', 'C'},
   {'Q', 'Q'}, {'R', 'G'}, {'S', 'Z'}, {'T', 'N'},
   {'U', 'P'}, {'V', 'Y'}, {'W', 'F'}, {'X', 'V'},
   {'Y', 'O'}, {'Z', 'E'}
};

cipherPair_t rotorThree[]={
   {'A', 'B'}, {'B', 'D'}, {'C', 'F'}, {'D', 'H'},
   {'E', 'J'}, {'F', 'L'}, {'G', 'C'}, {'H', 'P'},
   {'I', 'R'}, {'J', 'T'}, {'K', 'X'}, {'L', 'V'},
   {'M', 'Z'}, {'N', 'N'}, {'O', 'Y'}, {'P', 'E'},
   {'Q', 'I'}, {'R', 'W'}, {'S', 'G'}, {'T', 'A'},
   {'U', 'K'}, {'V', 'M'}, {'W', 'U'}, {'X', 'S'},
   {'Y', 'Q'}, {'Z', 'O'}
};

refMap_t reflectorMap[]={
   {0,  24}, {1,  17}, {2,  20}, {3,   7},
   {4,  16}, {5,  18}, {6,  11}, {7,   3},
   {8,  15}, {9,  23}, {10, 13}, {11,  6},
   {12, 14}, {13, 10}, {14, 12}, {15,  8},
   {16,  4}, {17,  1}, {18,  5}, {19, 25},
   {20,  2}, {21, 22}, {22, 21}, {23,  9},
   {24,  0}, {25, 19}
};

char refString[]="ABCDEFGDIJKGMKMIEBFTCVVJAT";

int findMatch(cipherPair_t *cmap, char inChar)
{
   int idx;

   for(idx=0;idx<ALPHA_SIZE;idx++)
   {
      if(cmap[idx].rightChar == inChar)
         break;
   }

   return idx;
}

int findDist(char pos1, char pos2)
{
  if(pos2 > pos1)
     return (pos2 - pos1);
  else
     return ((ALPHA_SIZE - pos1) + pos2);
}


int main( int argc, char *argv[] )
{
   int demoRun=0;
   int dbgOn=0, foundEnd=0;
   FILE *fpIn, *fpOut;
   int lineSz=128;
   int bytesRd=0;

   // Initial conditions for machine
   int r1Idx=12, r2Idx=2, r3Idx=10, inputIdx=0, idx, rIdx, rDiff;
   int notch1Idx=16, notch2Idx=4, notch3Idx=21;


   if(argc == 1 || argc == 2)
   {
      demoRun=1;
      printf("Will run simple demo\n");
      if(argc == 2 && (strncmp(argv[1],"-D", 2) == 0))
      {
         dbgOn=1;
      }
   }
   else if(argc >= 3)
   {
      strcpy(inputFile, argv[1]);
      strcpy(outFile, argv[2]);
      printf("Will run Enigma3 encode on input file=%s and produce file=%s\n",
             inputFile, outFile);

      if(argc == 4 && (strncmp(argv[3],"-D", 2) == 0))
      {
         dbgOn=1;
      }
   }
   else 
   {
      printf("Usage: enigma3 <file-to-encode> <outfile> [-D]\n");
      exit(-1);
   }


   if(demoRun)
   {
      strcpy(plainTextLine, demoEncode);
      foundEnd = 1;
   }
   else
   {
      if( (fpIn=fopen(inputFile, "r")) == (FILE *)0)
      {
         printf("ERROR: can't open input file\n");
         exit(-1);
      }
      if( (fpOut=fopen(outFile, "w")) == (FILE *)0)
      {
         printf("ERROR: can't open output file\n");
         exit(-1);
      }

      if((fgets(plainTextLine, lineSz, fpIn)) == (char *)0)
         exit(-1);
   }


   do
   {
      for(idx=0;idx<strlen(plainTextLine);idx++)
      {

          // Ensure that string is uppercase
          plainTextLine[idx] = (char)toupper((char)plainTextLine[idx]);

          // just copy over all non-alpha characters in string
          if(plainTextLine[idx] < 'A' || plainTextLine[idx] > 'Z')
          {
             cipherTextLine[idx] = plainTextLine[idx]; 
             continue;
          }

          inputIdx = ((int)plainTextLine[idx]) - ALPHA_BASE;

          // Rotor adjust before encode/decode
          //
          r3Idx++; // rotate Right rotor
          r3Idx %= ALPHA_SIZE;

          if(dbgOn)
          {
             printf("%c, %c, %c, input=%c\n",
                    (char)(ALPHA_BASE + r1Idx),
                    (char)(ALPHA_BASE + r2Idx),
                    (char)(ALPHA_BASE + r3Idx),
                    (char)(ALPHA_BASE + inputIdx));
          }

          // select r3 pair input
          rIdx = (inputIdx + r3Idx) % ALPHA_SIZE;
          if(dbgOn) printf("R3: %c, ", (char)(ALPHA_BASE + rIdx));

          // select r3 pair output
          rIdx = (rotorThree[rIdx].rightChar) - ALPHA_BASE;
          if(dbgOn) printf("%c\n", (char)(ALPHA_BASE + rIdx));


          // select r2 pair input
          rIdx = (r2Idx + findDist(r3Idx, rIdx)) % ALPHA_SIZE;
          if(dbgOn) printf("R2: %c, ", (char)(ALPHA_BASE + rIdx));

          // select r2 pair output
          rIdx = (rotorTwo[rIdx].rightChar) - ALPHA_BASE;
          if(dbgOn) printf("%c\n", (char)(ALPHA_BASE + rIdx));


          // select r1 pair input
          rIdx = (r1Idx + findDist(r2Idx, rIdx)) % ALPHA_SIZE;
          if(dbgOn) printf("R1: %c, ", (char)(ALPHA_BASE + rIdx));

          // select r1 pair output
          rIdx = (rotorOne[rIdx].rightChar) - ALPHA_BASE;
          if(dbgOn) printf("%c\n", (char)(ALPHA_BASE + rIdx));


          // select reflector input
          rIdx = (findDist(r1Idx, rIdx)) % ALPHA_SIZE;
          if(dbgOn) printf("REF: %c, ", refString[rIdx]);

          // select reflector output
          rIdx = reflectorMap[rIdx].outPos;
          if(dbgOn) printf("%c\n", refString[rIdx]);


          // select r1 pair input
          rIdx = (r1Idx + rIdx) % ALPHA_SIZE;
          if(dbgOn) printf("R1: %c, ", (char)(ALPHA_BASE + rIdx));

          // select r1 pair output
          rIdx = findMatch(&rotorOne[0], (char)(ALPHA_BASE + rIdx));
          if(dbgOn) printf("%c\n", (char)(ALPHA_BASE + rIdx));



          // select r2 pair input
          rIdx = (r2Idx + findDist(r1Idx, rIdx)) % ALPHA_SIZE;
          if(dbgOn) printf("R2: %c, ", (char)(ALPHA_BASE + rIdx));

          // select r2 pair output
          rIdx = findMatch(&rotorTwo[0], (char)(ALPHA_BASE + rIdx));
          if(dbgOn) printf("%c\n", (char)(ALPHA_BASE + rIdx));



          // select r3 pair output - final output
          rIdx = (r3Idx + findDist(r2Idx, rIdx)) % ALPHA_SIZE;
          if(dbgOn) printf("R3: %c, ", (char)(ALPHA_BASE + rIdx));

          rIdx = findMatch(&rotorThree[0], (char)(ALPHA_BASE + rIdx));
          if(dbgOn) printf("%c\n", (char)(ALPHA_BASE + rIdx));


          cipherTextLine[idx] = ALPHA_BASE + (findDist(r3Idx, rIdx) % ALPHA_SIZE);
          if(dbgOn) printf("output=%c\n", cipherTextLine[idx]);


          // rotate other two rotors if needed
          if(r3Idx == notch3Idx)
          {
             r2Idx++; // rotate Middle rotor

             if(r2Idx == notch2Idx)
             {
                r1Idx++;
             }
          }
          r2Idx %= ALPHA_SIZE; r1Idx %= ALPHA_SIZE;

      }

      cipherTextLine[idx] = '\0';

      printf("Input = %s\n", plainTextLine);
      printf("Output = %s\n", cipherTextLine);

      if(!foundEnd)
      {
         fputs(cipherTextLine, fpOut);

         if((fgets(plainTextLine, lineSz, fpIn)) == (char *)0)
            foundEnd=1;
      }

   } while (!foundEnd);


}
