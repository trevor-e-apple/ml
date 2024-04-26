#ifndef MATRIX_H

#include "project_flags.h"

#include <stdio.h>
#include <stdint.h>

struct Matrix
{
	uint32_t num_rows;
	uint32_t num_columns;
	float* data;
};

HOST_PREFIX DEVICE_PREFIX uint32_t get_matrix_array_count(Matrix* matrix);

#define MATRIX_H

#endif