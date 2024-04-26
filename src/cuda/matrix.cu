#include "matrix.h"

#include <stdio.h>

HOST_PREFIX DEVICE_PREFIX
uint32_t get_matrix_array_count(Matrix* matrix)
{
	return matrix->num_rows * matrix->num_columns;
}