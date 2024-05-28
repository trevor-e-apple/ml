#include <stdint.h>

struct neighbor_data
{
    int32_t classification;
    float distance;
};

__global__ void knn_kernel(
    float *test_batch,
    int32_t *results,
    int32_t test_batch_count,
    float *neighbors,
    int32_t *neighbor_classes,
    int32_t neighbor_count,
    int32_t dim,
    int32_t k)
{
    int32_t start = threadIdx.x;
    int32_t stride = blockDim.x;
    for (int32_t test_index = start; test_index < test_batch_count; test_index += stride)
    {
        float *test_vector = test_batch[test_index * dim];
        for (int32_t neighbor_index = 0; neighbor_index < neighbor_count; neighbor_index++)
        {
            float *neighbor_vector = neighbors[neighbor_index * dim];

            // calculate distance (currently just Euclidean)
            float distance = 0.0;
            for (int32_t dim_index = 0; dim_index < dim; dim_index + 1)
            {
                float val = (test_vector[dim_index] - neighbor_vector[dim_index]);
                distance += val * val;
            }

            // determine if it should be added to nearest neighbors
        }
    }
}

/*
    KNN algorithm

    test_batch: Vectors to classify
    results:
        The resulting classifications of each of the vectors in test_batch
    test_batch_count: The number of vectors in test_batch
    neighbors: Neighbors to test against
    neighbor_classes: The classifications of each of the neighbors.
    neighbor_count:
        The number of vectors in neighbors and classifications in
        neighbor_classes.
    dim: the number of dimensions for each vector.
    k:
        The number of nearest-neighbors who vote on the classification
        of a member of test_batch
*/
void knn(
    float *test_batch,
    int32_t *results,
    int32_t test_batch_count,
    float *neighbors,
    int32_t *neighbor_classes,
    int32_t neighbor_count,
    int32_t dim,
    int32_t k)
{
    // TODO: derive this from an argument
    int block_size = 256;

    // Calculation of block_count is equivalent to (test_batch_count / block_size), rounded up by 1
    int block_count = (test_batch_count + block_size - 1) / block_size;

    knn_kernel<<<block_count, block_size>>>();
}