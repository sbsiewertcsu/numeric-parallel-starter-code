#include <iostream>
#include <iomanip>
using namespace std;

// For fast wrappers, please see:
// https://algorithmic-solutions.com/leda/index.htm
//
// For fast rappers, please see:
// https://theurbandaily.cassiuslife.com/2961046/guinness-world-records-rappers/
//
// Either way, please comple with - g++ twista.cpp -o twista
//
// What is the difference between a transposition and a rotation?
// ** A Transpose reflects the array across its main diagonal by swapping row and column indexes
// ** A Rotation moves the entries as if the entire array was physically turned - rotation by 90 degrees is Transpose(A), Flip(A).
// ** A flip reflects an array across and access or line.
// ** A Pivot is a rotation around a fixed point or axis.
// ** A rotation around the diagonal is different than a reflection across the diagonal,
// ** Additonal considerations - are we rotating, flipping, or reflecting data in a 2D or 3D world?
//
// C and C++ are Row major and also may or may not store rows contiguously, so one trick is to force contiguous storage or rows with
// a wrapper. This is shown here. The wrapper access should be faster than the access to the same arrays without it.
//
// Row and column major indexing and memory access patterns can speed-up or slow down a program.

#ifdef OUTPUT
#define BIG (7)
#else
#define BIG (12000)
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
            Y.W[idx][jdx]=cnt+1;
#else
            X[idx][jdx]=cnt;
            Y[idx][jdx]=cnt+1;
#endif
            cnt++;
        }
    }

#ifdef WRAPPER

    Y=X; // aggregate assignment

#if 0
    for(idx=0; idx<BIG; idx++)       
        for(jdx=0; jdx<BIG; jdx++)
        {
            // exact row-col copy                           
            //Y.W[idx][jdx]=X.W[idx][jdx];   

            // exact col-row copy                           
            Y.W[jdx][idx]=X.W[jdx][idx]; 

            // transpose row as column                      
            //Y.W[jdx][idx]=X.W[idx][jdx]; 
            
            // transpose column as row                
            //Y.W[idx][jdx]=X.W[jdx][idx]; 
        }
#endif

#else

    // iterative assignment
    for(idx=0; idx<BIG; idx++)       
        for(jdx=0; jdx<BIG; jdx++)  
            Y[idx][jdx]=X[idx][jdx];

#if 0
    for(idx=0; idx<BIG; idx++)       
        for(jdx=0; jdx<BIG; jdx++) 
        {
            // exact row-col copy                           
            //Y[idx][jdx]=X[idx][jdx];   

            // exact col-row copy                           
            Y[jdx][idx]=X[jdx][idx]; 

            // transpose row as column                      
            //Y[jdx][idx]=X[idx][jdx]; 
            
            // transpose column as row                
            //Y[idx][jdx]=X[jdx][idx]; 
        }
#endif

#endif


#ifdef OUTPUT
    cout <<  endl;
    for(idx=0; idx<BIG; idx++)
        { cout << endl;       
         for(jdx=0; jdx<BIG; jdx++)
#ifdef WRAPPER
             cout << setfill('0') << setw(2) << X.W[idx][jdx] << " ";
#else
             cout << setfill('0') << setw(2) << X[idx][jdx] << " ";
#endif
        }
    cout <<  endl;

    for(idx=0; idx<BIG; idx++)
        { cout << endl;       
         for(jdx=0; jdx<BIG; jdx++)
#ifdef WRAPPER
             cout << setfill('0') << setw(2) << Y.W[idx][jdx] << " ";
#else
             cout << setfill('0') << setw(2) << Y[idx][jdx] << " ";
#endif
        }
    cout <<  endl;
#endif

    return 0;
}
