#include <cassert>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <png.h>

#define TILE_WIDTH 32

struct event_pair
{
  cudaEvent_t start;
  cudaEvent_t end;
};

inline void start_timer(event_pair * p)
{
  cudaEventCreate(&p->start);
  cudaEventCreate(&p->end);
  cudaEventRecord(p->start, 0);
}


inline float stop_timer(event_pair * p, char *name)
{
  cudaEventRecord(p->end, 0);
  cudaEventSynchronize(p->end);

  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, p->start, p->end);
  printf("%s took %.1f ms\n",name, elapsed_time);
  cudaEventDestroy(p->start);
  cudaEventDestroy(p->end);
  return elapsed_time;
}

//Memory allocation
bool prepMem(float *&d_DataR, float *&d_DataG, float *&d_DataB, float *&d_ResultR, float *&d_ResultG, float *&d_ResultB,
		  int data_size, float*&d_Mask, int mask_size){

	cudaMalloc((void**)&d_DataR,data_size);
	cudaMalloc((void**)&d_DataG,data_size);
	cudaMalloc((void**)&d_DataB,data_size);
	cudaMalloc((void**)&d_ResultR,data_size);
	cudaMalloc((void**)&d_ResultG,data_size);
	cudaMalloc((void**)&d_ResultB,data_size);

	cudaMalloc((void**)&d_Mask,mask_size);

	if(d_DataR != 0 && d_DataG != 0 && d_DataB != 0 && d_ResultR != 0 && d_ResultG != 0 && d_ResultB != 0 && d_Mask != 0)
		return true;
	else
		return false;
}

void freeMem(float *h_DataR, float *h_DataG, float *h_DataB, float *h_ResultR, float *h_ResultG, float *h_ResultB,
		  float *&d_DataR, float *&d_DataG, float *&d_DataB, float *&d_ResultR, float *&d_ResultG, float *&d_ResultB,
		  float *&d_check_ResultR, float *&d_check_ResultG, float *&d_check_ResultB, float *&d_Mask){
	free(h_ResultR);
    free(h_ResultG);
    free(h_ResultB);
    free(h_DataR);
    free(h_DataG);
    free(h_DataB);

	free(d_check_ResultR);
	free(d_check_ResultG);
	free(d_check_ResultB);

	cudaFree(d_ResultR);
	cudaFree(d_ResultG);
	cudaFree(d_ResultB);
	cudaFree(d_DataR);
	cudaFree(d_DataG);
	cudaFree(d_DataB);

	cudaFree(d_Mask);

}

void CPU_convolution(const float *input, float *output, int width, int height,
		const float* Mask, int mask_size) {
	int fila, col, i, j, mask_sz = 0;
	double suma = 0;
	mask_sz = mask_size / 2;

	for (fila = 0; fila < height; fila++) {
		for (col = 0; col < width; col++) {
			suma = 0; // set to zero before sum
			for (i = -mask_sz; i <= mask_sz; i++)
				for (j = -mask_sz; j <= mask_sz; j++) {
					if (fila + i >= 0 && fila + i < height && col + j >= 0
							&& col + j < width)
						suma +=
								input[width * (fila + i) + (col + j)]
										* Mask[mask_size * (mask_sz + i)
												+ (mask_sz + j)];
				}
			if (suma > 255)
				output[width * fila + col] = 255;

			else if (suma < 0)
				output[width * fila + col] = 0;

			else
				output[width * fila + col] = suma;

		} //fin for
	} //fin for
} //fin funcion

// kernel
__global__ void GPU_convolution(float *channel, float *mask, float *result,
		int dimMask, int dimW, int dimH) {
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int x, y;

	// Identify the row and column of the Pd element to work on
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	int nidRow = Row - dimMask / 2;
	int nidCol = Col - dimMask / 2;

	int tid = Row * dimW + Col;

	if (tid < dimW * dimH) {
		result[tid] = 0;
		for (int i = 0; i < dimMask; ++i) {
			x = nidRow * dimW + i * dimW;
			for (int j = 0; j < dimMask; ++j) {
				y = nidCol + j;
				// When the value is not beyond the borders
				if (x >= 0 && y >= 0 && x < dimW * dimH && y < dimW) {
					result[tid] += mask[dimMask * i + j] * channel[x + y];
				}
			}
		}
		if (result[tid] > 255)
			result[tid] = 255;
		if (result[tid] < 0)
			result[tid] = 0;
	}
}

png_byte color_type;
png_byte bit_depth;
png_bytep *row_pointers;
int width, height;

void obtenerPixeles(float *h_DataR, float *h_DataG, float *h_DataB){
    for(unsigned int i = 0; i < height; i++) {
        png_bytep row = row_pointers[i];
        for(unsigned int j = 0; j < width; j++) {
            png_bytep px = &(row[j*4]);
            h_DataR[height*i+j] =  px[0];
            h_DataG[height*i+j] =  px[1];
            h_DataB[height*i+j] =  px[2];
        }
    }
}

void reemplazarPixeles(float *h_DataR, float *h_DataG, float *h_DataB){
    for(unsigned int i = 0; i < height; i++) {
        png_bytep row = row_pointers[i];
        for(unsigned int j = 0; j < width; j++) {
            png_bytep px = &(row[j*4]);
            px[0] = h_DataR[height*i+j];
            px[1] = h_DataG[height*i+j];
            px[2] = h_DataB[height*i+j];
        }
    }
}

void escribirImagen(char *filename) {
  FILE *fp = fopen(filename, "wb");
  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  png_infop info = png_create_info_struct(png);
  png_init_io(png, fp);

  png_set_IHDR(
    png,
    info,
    width, height,
    8,
    PNG_COLOR_TYPE_RGBA,
    PNG_INTERLACE_NONE,
    PNG_COMPRESSION_TYPE_DEFAULT,
    PNG_FILTER_TYPE_DEFAULT
  );

  png_set_expand(png);
  png_write_info(png, info);
  png_write_image(png, row_pointers);
  png_write_end(png, NULL);

  for(int y = 0; y < height; y++) {
    free(row_pointers[y]);
  }
  free(row_pointers);

  fclose(fp);
}

void leerImagen(char *filename) {
    FILE *fp = fopen(filename, "rb");
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) abort();
    png_infop info = png_create_info_struct(png);
    if (!info) abort();
    if(setjmp(png_jmpbuf(png))) abort();

    png_init_io(png, fp);
    png_read_info(png, info);
    width      = png_get_image_width(png, info);
    height     = png_get_image_height(png, info);
    color_type = png_get_color_type(png, info);
    bit_depth  = png_get_bit_depth(png, info);

    if(bit_depth == 16)
        png_set_strip_16(png);

    if(color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);

    if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);

    if(png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);

    if(color_type == PNG_COLOR_TYPE_RGB ||
        color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

    if(color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);
    row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for(int y = 0; y < height; y++) {
        row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png,info));
    }

    png_read_image(png, row_pointers);
    fclose(fp);
}


int main(int argc, char **argv){
	//declaracion de variables
	char *iFilename  = "resources/paisajecambio.png";
	char *odFilename = "resources/image_od.png";

	//gestion del tiempo de ejecucion
	event_pair timer;

	//arreglos para el procesamiento de imagenes (h) = host, (d) = device
	float *h_DataR, *h_DataG, *h_DataB, *h_ResultR, *h_ResultG, *h_ResultB,
			*d_DataR, *d_DataG, *d_DataB, *d_ResultR, *d_ResultG, *d_ResultB,
			*d_Mask, *check_d_ResultR, *check_d_ResultG, *check_d_ResultB,
			error;

	int data_size, mask_size, mask_side;
	bool error_flag = false;
	error = 1;

	/*3x3 dev = 0.57*/
	mask_side = 17;
	mask_size = mask_side * mask_side * sizeof(float);
	float h_Mask[289] = {1,3.8856e+26,4.30185e+49,1.35703e+69,1.21974e+85,3.12377e+97,2.27946e+106,4.73941e+111,2.80772e+113,4.73941e+111,2.27946e+106,3.12377e+97,1.21974e+85,1.35703e+69,4.30185e+49,3.8856e+26,1,
			3.8856e+26,1.50979e+53,1.67153e+76,5.27289e+95,4.73941e+111,1.21377e+124,8.85708e+132,1.84154e+138,1.09097e+140,1.84154e+138,8.85708e+132,1.21377e+124,4.73941e+111,5.27289e+95,1.67153e+76,1.50979e+53,3.8856e+26,
			4.30185e+49,1.67153e+76,1.85059e+99,5.83776e+118,5.24712e+134,1.3438e+147,9.8059e+155,2.03882e+161,1.20784e+163,2.03882e+161,9.8059e+155,1.3438e+147,5.24712e+134,5.83776e+118,1.85059e+99,1.67153e+76,4.30185e+49,
			1.35703e+69,5.27289e+95,5.83776e+118,1.84154e+138,1.65522e+154,4.23907e+166,3.09331e+175,6.43154e+180,3.81018e+182,6.43154e+180,3.09331e+175,4.23907e+166,1.65522e+154,1.84154e+138,5.83776e+118,5.27289e+95,1.35703e+69,
			1.21974e+85,4.73941e+111,5.24712e+134,1.65522e+154,1.48776e+170,3.81018e+182,2.78034e+191,5.78082e+196,3.42468e+198,5.78082e+196,2.78034e+191,3.81018e+182,1.48776e+170,1.65522e+154,5.24712e+134,4.73941e+111,1.21974e+85,
			3.12377e+97,1.21377e+124,1.3438e+147,4.23907e+166,3.81018e+182,9.75796e+194,7.12052e+203,1.48048e+209,8.7707e+210,1.48048e+209,7.12052e+203,9.75796e+194,3.81018e+182,4.23907e+166,1.3438e+147,1.21377e+124,3.12377e+97,
			2.27946e+106,8.85708e+132,9.8059e+155,3.09331e+175,2.78034e+191,7.12052e+203,5.19595e+212,1.08033e+218,6.4001e+219,1.08033e+218,5.19595e+212,7.12052e+203,2.78034e+191,3.09331e+175,9.8059e+155,8.85708e+132,2.27946e+106,
			4.73941e+111,1.84154e+138,2.03882e+161,6.43154e+180,5.78082e+196,1.48048e+209,1.08033e+218,2.2462e+223,1.33069e+225,2.2462e+223,1.08033e+218,1.48048e+209,5.78082e+196,6.43154e+180,2.03882e+161,1.84154e+138,4.73941e+111,
			2.80772e+113,1.09097e+140,1.20784e+163,3.81018e+182,3.42468e+198,8.7707e+210,6.4001e+219,1.33069e+225,7.88332e+226,1.33069e+225,6.4001e+219,8.7707e+210,3.42468e+198,3.81018e+182,1.20784e+163,1.09097e+140,2.80772e+113,
			4.73941e+111,1.84154e+138,2.03882e+161,6.43154e+180,5.78082e+196,1.48048e+209,1.08033e+218,2.2462e+223,1.33069e+225,2.2462e+223,1.08033e+218,1.48048e+209,5.78082e+196,6.43154e+180,2.03882e+161,1.84154e+138,4.73941e+111,
			2.27946e+106,8.85708e+132,9.8059e+155,3.09331e+175,2.78034e+191,7.12052e+203,5.19595e+212,1.08033e+218,6.4001e+219,1.08033e+218,5.19595e+212,7.12052e+203,2.78034e+191,3.09331e+175,9.8059e+155,8.85708e+132,2.27946e+106,
			3.12377e+97,1.21377e+124,1.3438e+147,4.23907e+166,3.81018e+182,9.75796e+194,7.12052e+203,1.48048e+209,8.7707e+210,1.48048e+209,7.12052e+203,9.75796e+194,3.81018e+182,4.23907e+166,1.3438e+147,1.21377e+124,3.12377e+97,
			1.21974e+85,4.73941e+111,5.24712e+134,1.65522e+154,1.48776e+170,3.81018e+182,2.78034e+191,5.78082e+196,3.42468e+198,5.78082e+196,2.78034e+191,3.81018e+182,1.48776e+170,1.65522e+154,5.24712e+134,4.73941e+111,1.21974e+85,
			1.35703e+69,5.27289e+95,5.83776e+118,1.84154e+138,1.65522e+154,4.23907e+166,3.09331e+175,6.43154e+180,3.81018e+182,6.43154e+180,3.09331e+175,4.23907e+166,1.65522e+154,1.84154e+138,5.83776e+118,5.27289e+95,1.35703e+69,
			4.30185e+49,1.67153e+76,1.85059e+99,5.83776e+118,5.24712e+134,1.3438e+147,9.8059e+155,2.03882e+161,1.20784e+163,2.03882e+161,9.8059e+155,1.3438e+147,5.24712e+134,5.83776e+118,1.85059e+99,1.67153e+76,4.30185e+49,
			3.8856e+26,1.50979e+53,1.67153e+76,5.27289e+95,4.73941e+111,1.21377e+124,8.85708e+132,1.84154e+138,1.09097e+140,1.84154e+138,8.85708e+132,1.21377e+124,4.73941e+111,5.27289e+95,1.67153e+76,1.50979e+53,3.8856e+26,
			1,3.8856e+26,4.30185e+49,1.35703e+69,1.21974e+85,3.12377e+97,2.27946e+106,4.73941e+111,2.80772e+113,4.73941e+111,2.27946e+106,3.12377e+97,1.21974e+85,1.35703e+69,4.30185e+49,3.8856e+26,1};

	//lectura y obtencion de pixeles de la imagen
	leerImagen(iFilename);
	printf("width: %d, height: %d", width, height);
	//calculo del total de pixeles para manejo de memoria
	data_size = width * height * sizeof(float);

	//reserva de memoria para arreglos del host
	h_DataR = (float *) malloc(data_size);
	h_DataG = (float *) malloc(data_size);
	h_DataB = (float *) malloc(data_size);
	h_ResultR = (float *) malloc(data_size);
	h_ResultG = (float *) malloc(data_size);
	h_ResultB = (float *) malloc(data_size);
	//reserva de memoria para arreglos de comprobacion
	check_d_ResultR = (float *) malloc(data_size);
	check_d_ResultG = (float *) malloc(data_size);
	check_d_ResultB = (float *) malloc(data_size);
	//reserva de memoria el arreglos del device
	if (!prepMem(d_DataR, d_DataG, d_DataB, d_ResultR, d_ResultG, d_ResultB,
				data_size, d_Mask, mask_size)) {
		std::cerr << "Error allocating device memory!" << std::endl;
		abort();
	}

	//convertir imagen en arreglos rgb
	obtenerPixeles(h_DataR, h_DataG, h_DataB);

	/*|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||*/
	// copy array inputs to GPU
	start_timer(&timer);
	cudaMemcpy(d_DataR, h_DataR, data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_DataG, h_DataG, data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_DataB, h_DataB, data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Mask, h_Mask, mask_size, cudaMemcpyHostToDevice);
	stop_timer(&timer, "host to device copy of inputs");

	/*-----------Kernel call - GPU Convolution--------------*/
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 dimGrid(width / TILE_WIDTH, height / TILE_WIDTH, 1);

	start_timer(&timer);
	GPU_convolution<<<dimGrid, dimBlock>>>(d_DataR, d_Mask, d_ResultR, mask_side,
			width, height);
	GPU_convolution<<<dimGrid, dimBlock>>>(d_DataG, d_Mask, d_ResultG, mask_side,
			width, height);
	GPU_convolution<<<dimGrid, dimBlock>>>(d_DataB, d_Mask, d_ResultB, mask_side,
			width, height);
	stop_timer(&timer, "GPU convolution time");

	/*-----------CPU Convolution--------------*/
	start_timer(&timer);
	CPU_convolution(h_DataR, h_ResultR, width, height, h_Mask, mask_side);
	CPU_convolution(h_DataG, h_ResultG, width, height, h_Mask, mask_side);
	CPU_convolution(h_DataB, h_ResultB, width, height, h_Mask, mask_side);
	stop_timer(&timer, "CPU convolution time");

	/*-----------Revision de resultados--------------*/
	// copy array outputs to CPU
	start_timer(&timer);
	cudaMemcpy(check_d_ResultR, d_ResultR, data_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(check_d_ResultG, d_ResultG, data_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(check_d_ResultB, d_ResultB, data_size, cudaMemcpyDeviceToHost);
	stop_timer(&timer, "device to host copy of results");

	int num_errores = 0;
	for (int i = 0; i < (width*height); i++) {
		if (abs(check_d_ResultR[i] - h_ResultR[i]) > error
			|| abs(check_d_ResultG[i] - h_ResultG[i]) > error
			|| abs(check_d_ResultB[i] - h_ResultB[i]) > error) {
				printf("Error in the result is bigger than threshold %f for data %d\n", error, i);
				error_flag = true;
				num_errores++;
		}
	}

	if (error_flag == false)
		printf("Results on device and host match\n");

	else
		printf("Results don't match, number of errors: %d\n", num_errores);

	/*-----------Escritura de resultados--------------*/
	//GPU
	reemplazarPixeles(check_d_ResultR, check_d_ResultG, check_d_ResultB);
	escribirImagen(odFilename);
	//CPU
	//reemplazarPixeles(h_ResultR, h_ResultG, h_ResultB);
	//escribirImagen(ohFilename);

	/*-----------Libera la memoria--------------*/
	freeMem(h_DataR, h_DataG, h_DataB, h_ResultR, h_ResultG, h_ResultB, d_DataR,
			d_DataG, d_DataB, d_ResultR, d_ResultG, d_ResultB, check_d_ResultR,
			check_d_ResultG, check_d_ResultB, d_Mask);

	return 0;

}
