################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../cudaconvolucion.cu 

OBJS += \
./cudaconvolucion.o 

CU_DEPS += \
./cudaconvolucion.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.1/bin/nvcc -G -g -O0 -gencode arch=compute_70,code=sm_70  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.1/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_70,code=compute_70 -gencode arch=compute_70,code=sm_70  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


