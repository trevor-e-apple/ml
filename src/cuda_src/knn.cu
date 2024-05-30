#include <stdint.h>
#include <float.h>

struct nn_datum
{
    int32_t classification;
    float distance;
};

__device__ float euclidean_distance(float *a, float *b, int32_t dim)
{
    float distance = 0.0;
    for (int32_t dim_index = 0; dim_index < dim; dim_index += 1)
    {
        float val = (a[dim_index] - b[dim_index]);
        distance += val * val;
    }
    return distance;
}

__global__ void knn_kernel(
    float *test_batch,
    int32_t *results,
    int32_t test_batch_count,
    float *neighbors,
    int32_t *neighbor_classes,
    int32_t neighbor_count,
    int32_t dim,
    nn_datum *nn_data,
    int32_t *nn_class_counts,
    int32_t k)
{
    int32_t start = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = blockDim.x * gridDim.x;

    for (int32_t test_index = start; test_index < test_batch_count; test_index += stride)
    {
        float *test_vector = test_batch + test_index * dim;

        // initialize neighbor data
        for (int32_t nn_index = 0; nn_index < k; nn_index++)
        {
            nn_data[nn_index].distance = FLT_MAX;
            nn_class_counts[nn_index] = 0;
        }

        // initialize some data about the farthest nearest neighbor
        int32_t farthest_nn_index = 0;
        float farthest_nn_distance = euclidean_distance(test_vector, &neighbors[0], dim);

        for (int32_t neighbor_index = 0; neighbor_index < neighbor_count; neighbor_index++)
        {
            float *neighbor_vector = neighbors[neighbor_index * dim];

            // calculate distance (currently just Euclidean)
            float distance = euclidean_distance(test_vector, neighbor_vector, dim);

            // determine if it should be added to nearest neighbors
            if (distance <= farthest_nn_distance)
            {
                nn_data[farthest_nn_index] = nn_datum{.classification = neighbor_classes[neighbor_index], .distance = distance};

                // find new farthest neighbor
                farthest_nn_index = 0;
                farthest_nn_distance = nn_data[0].distance;
                for (int32_t nn_index = 1; nn_index < k; nn_index++)
                {
                    if (nn_data
                            [nn_index]
                                .distance > farthest_nn_distance)
                    {
                        farthest_nn_distance = nn_data[nn_index].distance;
                        farthest_nn_index = nn_index;
                    }
                }
            }
        }

        // tally up the votes
        for (int32_t nn_index = 0; nn_index < k; nn_index++)
        {
            for (int32_t inner_loop_index = 0; inner_loop_index < k; inner_loop_index++)
            {
                if (nn_data[nn_index] == nn_data[inner_loop_index])
                {
                    nn_class_counts[nn_index]++;
                }
            }
        }

        // find highest vote getter and write to results
        int32_t max_count_index = 0;
        int32_t max_count = nn_class_counts[max_count_index];
        for (int32_t nn_index = 1; nn_index < k; nn_index++)
        {
            if (nn_class_counts[nn_index] > max_count)
            {
                max_count_index = nn_index;
                max_count = nn_class_counts[max_count_index];
            }
        }
        results[test_index] = nn_data[max_count_index].classification;
    }
}

/*
    KNN algorithm

    test_batch: Vectors to classify
    results:
        The resulting classifications of each of the vectors in test_batch
    test_batch_count: The number of vectors in test_batch. Must be greater than 0.
    neighbors: Neighbors to test against
    neighbor_classes: The classifications of each of the neighbors.
    neighbor_count:
        The number of vectors in neighbors and classifications in
        neighbor_classes. Must be greater than 0.
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
    int thread_count = block_count * block_size;

    // make an array for each thread to hold its nearest neighbors
    // TODO: If someone is performing a bunch of these calculations in a row, this allocation / free cycle
    // --  is a waste of time. Give them a way to preconfigure.
    nn_datum *nn_data = NULL;
    cudaError_t error = cudaMalloc(&nn_data, sizeof(nn_data) * k * thread_count);

    int32_t *nn_class_counts = NULL;
    cudaMalloc(&nn_class_counts, sizeof(in32_t) * k * thread_count);

    knn_kernel<<<block_count, block_size>>>();

    cudaFree(nn_data);
}