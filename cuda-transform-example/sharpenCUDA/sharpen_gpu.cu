/* ****************************************
Requirements: CUDA enabled GPU installed and NVIDIA driver installed
Input: a 320x240 ppm image file given in command line as argument
Output: Image processed ppm file. This looks sharper.
How: This program uses the same algorithm as found in Prof Siewert's "sharpen.c" code for the image 
processing part.  The computation is performed in the GP-GPU.

Aurhor: Adnan Reza
Credit: Sam Siewert for the original image processing code segment.
****************************************** */

//#include<cutil_inline.h>
#include<cuda_runtime.h>
#include<stdlib.h>
#include<stdio.h>
#include<stdint.h>
#include<unistd.h>
#include<sys/types.h>
#include<unistd.h>
#include<fcntl.h>
#include<sys/time.h>
#include<errno.h>
#include<time.h>

#define nROW (240)
#define nCOL (320)
#define SIZE (nROW*nCOL)
#define USE_CUDA

#define DEBUG (0)

// Host memory for image pixels
uint8_t r[SIZE];
uint8_t g[SIZE];
uint8_t b[SIZE];
uint8_t sharpR[SIZE];
uint8_t sharpG[SIZE];
uint8_t sharpB[SIZE];

// Device memory for image pixels
uint8_t *rdev; size_t rdevp;
uint8_t *gdev; size_t gdevp;
uint8_t *bdev; size_t bdevp;
uint8_t *sharpRdev; size_t sharpRdevp;
uint8_t *sharpGdev; size_t sharpGdevp;
uint8_t *sharpBdev; size_t sharpBdevp;

struct timeval tv_start, tv_end;
#ifdef USE_CUDA
// The CUDA Kernel to perform the computations
__global__ void image_process(uint8_t* r, uint8_t* g, uint8_t* b, size_t rP, size_t gP, size_t bP
		,uint8_t* newR, uint8_t* newG, uint8_t* newB, size_t newRp, size_t newGp, size_t newBp){
	int temp;
	double K=4.0;
	double PSF[9] = {-K/8.0, -K/8.0, -K/8.0, -K/8.0, K+1.0, -K/8.0, -K/8.0, -K/8.0, -K/8.0};
	int th_row=blockIdx.y*blockDim.y+threadIdx.y;
	int th_col=blockIdx.x*blockDim.x+threadIdx.x;
	int i=th_row; int j=th_col;
	
	newR[(i*newRp)+j]=(uint8_t)0;
	newG[(i*newGp)+j]=(uint8_t)0;
	newB[(i*newBp)+j]=(uint8_t)0;
	if(0<th_row && nROW-1>th_row && 0<th_col && nCOL-1>th_col){
		temp=0;
		temp += (PSF[0] * (float)r[((i-1)*rP)+j-1]);
		temp += (PSF[1] * (float)r[((i-1)*rP)+j]);
		temp += (PSF[2] * (float)r[((i-1)*rP)+j+1]);
		temp += (PSF[3] * (float)r[((i)*rP)+j-1]);
		temp += (PSF[4] * (float)r[((i)*rP)+j]);
		temp += (PSF[5] * (float)r[((i)*rP)+j+1]);
		temp += (PSF[6] * (float)r[((i+1)*rP)+j-1]);
		temp += (PSF[7] * (float)r[((i+1)*rP)+j]);
		temp += (PSF[8] * (float)r[((i+1)*rP)+j+1]);
		if(temp<0.0) temp=0.0;
		if(temp>255.0) temp=255.0;
		newR[(i*newRp)+j]=(uint8_t)temp;
	
		temp=0;
		temp += (PSF[0] * (float)g[((i-1)*gP)+j-1]);
		temp += (PSF[1] * (float)g[((i-1)*gP)+j]);
		temp += (PSF[2] * (float)g[((i-1)*gP)+j+1]);
		temp += (PSF[3] * (float)g[((i)*gP)+j-1]);
		temp += (PSF[4] * (float)g[((i)*gP)+j]);
		temp += (PSF[5] * (float)g[((i)*gP)+j+1]);
		temp += (PSF[6] * (float)g[((i+1)*gP)+j-1]);
		temp += (PSF[7] * (float)g[((i+1)*gP)+j]);
		temp += (PSF[8] * (float)g[((i+1)*gP)+j+1]);
		if(temp<0.0) temp=0.0;
		if(temp>255.0) temp=255.0;
		newG[(i*newGp)+j]=(uint8_t)temp;
		
		temp=0;
		temp += (PSF[0] * (float)b[((i-1)*bP)+j-1]);
		temp += (PSF[1] * (float)b[((i-1)*bP)+j]);
		temp += (PSF[2] * (float)b[((i-1)*bP)+j+1]);
		temp += (PSF[3] * (float)b[((i)*bP)+j-1]);
		temp += (PSF[4] * (float)b[((i)*bP)+j]);
		temp += (PSF[5] * (float)b[((i)*bP)+j+1]);
		temp += (PSF[6] * (float)b[((i+1)*bP)+j-1]);
		temp += (PSF[7] * (float)b[((i+1)*bP)+j]);
		temp += (PSF[8] * (float)b[((i+1)*bP)+j+1]);
		if(temp<0.0) temp=0.0;
		if(temp>255.0) temp=255.0;
		newB[(i*newBp)+j]=(uint8_t)temp;	  
	
	}
	else if(0==th_row || nROW-1==th_row || 0==th_col || nCOL-1==th_col){
	    newR[(i*newRp)+j]=(uint8_t)r[(i*rP)+j];	
		newG[(i*newGp)+j]=(uint8_t)g[(i*gP)+j];
	    newB[(i*newBp)+j]=(uint8_t)b[(i*bP)+j];
	}
}
#endif

int main(int argc, char *argv[]){
	int i;
	char infilename[128];
	int infd,outfd;
	char ppm_header[128];	
#ifdef USE_CUDA	
	cudaError_t cuda_ret;
	dim3 mainGrid(80,60); dim3 rowBlock(4,4);
#endif	
	if(2!=argc){
		printf("Usage:: ./filename imagefile.ppm\nExit\n");
		return -1;
	}
        if(DEBUG) printf("size of uint8_t is %lu\n",sizeof(uint8_t));
	sprintf(infilename,"%s",argv[1]);
	infd=open(infilename, O_RDONLY,0644);
	if(0>infd){
		perror("ERROR opening file");
		exit(-1);
	}
	outfd=open("sharpened.ppm",(O_RDWR | O_CREAT),0666);
	read(infd, ppm_header,38);
	ppm_header[38]='\0';

        if(DEBUG) printf("HEADER is %s",ppm_header);

	// Read the image
	for(i=0;i<SIZE;i++){
		read(infd,(void*)&r[i],1);
		read(infd,(void*)&g[i],1);
		read(infd,(void*)&b[i],1);
	}
	close(infd);
#ifdef USE_CUDA		
	// Allocate Device memory
        if(DEBUG) printf("success=%d; error=%d\n",cudaSuccess,cudaErrorMemoryAllocation);
	cuda_ret=cudaMallocPitch(&rdev,&rdevp,nCOL*sizeof(uint8_t),nROW);
        if(cuda_ret==cudaSuccess && DEBUG) printf("CUDA MEM ALLOC SUCCESS\n");
        else if(cuda_ret==cudaErrorMemoryAllocation) printf("CUDA MEM ALLOC ERROR\n");
	cuda_ret=cudaMallocPitch(&gdev,&gdevp,nCOL*sizeof(uint8_t),nROW);
        if(cuda_ret==cudaSuccess && DEBUG) printf("CUDA MEM ALLOC SUCCESS\n");
        else if(cuda_ret==cudaErrorMemoryAllocation) printf("CUDA MEM ALLOC ERROR\n");	
	cuda_ret=cudaMallocPitch(&bdev,&bdevp,nCOL*sizeof(uint8_t),nROW);
        if(cuda_ret==cudaSuccess && DEBUG) printf("CUDA MEM ALLOC SUCCESS\n");
        else if(cuda_ret==cudaErrorMemoryAllocation) printf("CUDA MEM ALLOC ERROR\n");	
	cuda_ret=cudaMallocPitch(&sharpRdev,&sharpRdevp,nCOL*sizeof(uint8_t),nROW);
        if(cuda_ret==cudaSuccess && DEBUG) printf("CUDA MEM ALLOC SUCCESS\n");
        else if(cuda_ret==cudaErrorMemoryAllocation) printf("CUDA MEM ALLOC ERROR\n");
	cuda_ret=cudaMallocPitch(&sharpGdev,&sharpGdevp,nCOL*sizeof(uint8_t),nROW);
        if(cuda_ret==cudaSuccess && DEBUG) printf("CUDA MEM ALLOC SUCCESS\n");
        else if(cuda_ret==cudaErrorMemoryAllocation) printf("CUDA MEM ALLOC ERROR\n");
	cuda_ret=cudaMallocPitch(&sharpBdev,&sharpBdevp,nCOL*sizeof(uint8_t),nROW);
        if(cuda_ret==cudaSuccess && DEBUG) printf("CUDA MEM ALLOC SUCCESS\n");
        else if(cuda_ret==cudaErrorMemoryAllocation) printf("CUDA MEM ALLOC ERROR\n");

        printf("Pitch sizes: %lu %lu %lu %lu %lu %lu\n",rdevp,gdevp,bdevp,sharpRdevp,sharpGdevp,sharpBdevp);	

	gettimeofday(&tv_start,NULL);
	// Copy from host to device memory
	cuda_ret=cudaMemcpy2D((void*)rdev,rdevp,(const void*)r,nCOL*sizeof(uint8_t),nCOL*sizeof(uint8_t),nROW,cudaMemcpyHostToDevice);
        if(DEBUG) printf("cudaMemcpy2D returns=%d\n", cuda_ret);	
	cuda_ret=cudaMemcpy2D((void*)gdev,gdevp,(const void*)g,nCOL*sizeof(uint8_t),nCOL*sizeof(uint8_t),nROW,cudaMemcpyHostToDevice);
        if(DEBUG) printf("cudaMemcpy2D returns=%d\n", cuda_ret);	
	cuda_ret=cudaMemcpy2D((void*)bdev,bdevp,(const void*)b,nCOL*sizeof(uint8_t),nCOL*sizeof(uint8_t),nROW,cudaMemcpyHostToDevice);
        if(DEBUG) printf("cudaMemcpy2D returns=%d\n", cuda_ret);

printf("Host to device copy .. done\n");	
	image_process<<<mainGrid,rowBlock>>>(rdev, gdev, bdev, rdevp, gdevp, bdevp
		, sharpRdev, sharpGdev, sharpBdev, sharpRdevp, sharpGdevp, sharpBdevp);
	//cudaThreadSynchronize();	
		
	// Copy processed RBG data from device to host memory	
	cudaMemcpy2D((void*)sharpR,nCOL*sizeof(uint8_t),(const void*)sharpRdev,sharpRdevp,nCOL*sizeof(uint8_t),nROW,cudaMemcpyDeviceToHost);
	cudaMemcpy2D((void*)sharpG,nCOL*sizeof(uint8_t),(const void*)sharpGdev,sharpGdevp,nCOL*sizeof(uint8_t),nROW,cudaMemcpyDeviceToHost);
	cudaMemcpy2D((void*)sharpB,nCOL*sizeof(uint8_t),(const void*)sharpBdev,sharpBdevp,nCOL*sizeof(uint8_t),nROW,cudaMemcpyDeviceToHost);
	gettimeofday(&tv_end,NULL);
#endif
	
	write(outfd, (void *)ppm_header, 38);	
	for(i=0; i<SIZE; i++)
	{
		write(outfd, (void *)&sharpR[i], 1);
		write(outfd, (void *)&sharpG[i], 1);
		write(outfd, (void *)&sharpB[i], 1);
	}
	close(outfd);	
#ifdef USE_CUDA	
	cudaFree(rdev);
	cudaFree(gdev);
	cudaFree(bdev);
	cudaFree(sharpRdev);
	cudaFree(sharpGdev);
	cudaFree(sharpBdev);
	//cudaThreadExit();	
#endif
	printf("Time elapsed= %f ms\n",(1000000*tv_end.tv_sec+tv_end.tv_usec-1000000*tv_start.tv_sec-tv_start.tv_usec)/1000.0);
	
	return 0;
}
