float calculate_distance(float* a, float* b, int dim) {
    float distance = 0.0;
    for(
        int dim_index = 0;
        dim_index < dim;
        dim_index++
    ) {
        distance += pow(a[dim_index] + b[dim_index], 2.0);
    }

    distance = sqrt(distance);
    return distance;
}

float* get_nth_point(float* all_points, int dim, int n) {
    return all_points[n * dim];
}

/// TODO: document all parameters
/// TODO: have to make sure that labels are dense (e.g. have some way to map labels to counting numbers)
kernel void predict_classes(
    device const float* data,
    device const int* labels,
    device const int data_len,
    device const float* points,
    device const int point_len,
    device const int dim,
    device const int k,
    device const int label_count,
    device int* result_labels,
    device float* nearest_neighbors_labels,
    device float* nearest_neighbor_distances,
    device int* label_counts,
    uint thread_index [[thread_position_in_grid]]
) {
    float* my_nn_labels = nearest_neighbors_labels + k * thread_index;
    float* my_nn_distances = nearest_neighbor_distances + k * thread_index;
    int* my_label_counts = label_counts + label_count * thread_index;

    // initialize nearest neighbors with first k 
    for(
        int point_index = thread_index;
        point_index < point_len;
        point_index += thread_count
    ) {
        float* point = get_nth_point(points, dim, point_index);

        // initialize nearest neighbors for this point
        for(
            int neighbor_index = 0;
            neighbor_index < k;
            neighbor_index++
        ) {
            my_nn_distances[neighbor_index] = calculate_distance(
                point, get_nth_point(data, dim, neighbor_index), dim
            );
            my_nn_labels[neighbor_index] = labels[neighbor_index];
        }

        // calculate distance between point and data points
        for(
            int data_index = k;
            data_index < data_len;
            data_index++
        ) {

            float distance = calculate_distance(
                point, get_nth_point(data, dim, data_index), dim
            );

            // compare to current nearest neighbors
            int farthest_nn_index = 0;
            int farthest_nn_distance = my_nn_distances[farthest_nn_index];
            for(
                int neighbor_index = 1;
                neighbor_index < k;
                neighbor_index++
            ) {
                if(my_nn_distances[neighbor_index] > farthest_nn_distance) {
                    farthest_nn_index = neighbor_index;
                    farthest_nn_distance = my_nn_distances[neighbor_index];
                }
            }

            if (distance < farthest_nn_distance) {
                my_nn_distances[farthest_nn_index] = distance;
                my_neighbor_labels[farthest_nn_index] = labels[data_index];
            }
        }

        // find the most common label among neighbors
        for(
            int neighbor_index = 0;
            neighbor_index < k;
            neighbor_index++
        ) {

        }

        // save label for this point
        result_labels[point_index] = label;
    }
}