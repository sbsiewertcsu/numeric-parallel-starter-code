#include <iostream>
using namespace std;

// For fast wrappers, please see:
// https://algorithmic-solutions.com/leda/index.htm
// For fast rappers, please see:
// https://theurbandaily.cassiuslife.com/2961046/guinness-world-records-rappers/
//
// Either way, please comple with - g++ twista.cpp -o twista
//
// What is the difference between a transposition and a rotation?
//
// Transposition is rotation about the diagonal
// Rotation is around the center element as a pivot (more complex)
//     Transpose, then reflect vertical or horizontal

#ifdef OUTPUT
#define BIG (3)
#else
#define BIG (10000)
#endif

#define LUNT unsigned long long int

#ifdef WRAPPER
struct wrapper 
{
    LUNT W[BIG][BIG];
} X, Y;
#else
LUNT X[BIG][BIG];
LUNT Y[BIG][BIG];
#endif


int main(void)
{
   int idx, jdx;
   LUNT cnt=0;

    { for(idx=0; idx<BIG; idx++)       
        for(jdx=0; jdx<BIG; jdx++)
        {
#ifdef WRAPPER
            X.W[idx][jdx]=cnt;
            Y.W[idx][jdx]=0;
#else
            X[idx][jdx]=cnt;
            Y[idx][jdx]=0;
#endif
            cnt++;
        }
    }

#ifdef WRAPPER
    Y=X;
#else
    for(idx=0; idx<BIG; idx++)       
        for(jdx=0; jdx<BIG; jdx++)  
            // exact row-col copy                           
            //Y[idx][jdx]=X[idx][jdx];   

            // exact col-row copy                           
            //Y[jdx][idx]=X[jdx][idx]; 

            // transpose row as column                      
            Y[jdx][idx]=X[idx][jdx]; 
            
            // transpose column as row                
            //Y[idx][jdx]=X[jdx][idx]; 


#endif

#ifdef OUTPUT
    cout <<  endl;
    for(idx=0; idx<BIG; idx++)
        { cout << endl;       
         for(jdx=0; jdx<BIG; jdx++)
#ifdef WRAPPER
             cout << X.W[idx][jdx];
#else
             cout << X[idx][jdx];
#endif
        }
    cout <<  endl;

    for(idx=0; idx<BIG; idx++)
        { cout << endl;       
         for(jdx=0; jdx<BIG; jdx++)
#ifdef WRAPPER
             cout << Y.W[idx][jdx];
#else
             cout << Y[idx][jdx];
#endif
        }
    cout <<  endl;
#endif

    return 0;
}
