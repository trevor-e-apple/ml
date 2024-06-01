use core::panic;
use std::{collections::HashMap, thread};

impl Knn {
    pub fn new(k: usize, data: Vec<Vec<f32>>, labels: Vec<Label>) -> Self {
        if k == 0 {
            panic!();
        }
        Self { k, data, labels }
    }

    // TODO: make this replaceable somehow (maybe with generics?)
    fn calc_distance(a: &[f32], b: &[f32]) -> f32 {
        let mut distance = 0.0;
        for (a_item, b_item) in a.iter().zip(b) {
            distance += (a_item - b_item).powf(2.0);
        }

        distance.sqrt()
    }

    pub fn predict_class(&self, point: &Vec<f32>) -> Label {
        #[derive(Clone, Copy)]
        struct NeighborData {
            distance: f32,
            label: Label,
        }

        // initialize with first k neighbors
        let mut nearest_neighbors: Vec<NeighborData> =
            Vec::with_capacity(self.k);

        let (first_k_data, remaining_data) = self.data.split_at(self.k + 1);
        let (first_k_labels, remaining_labels) =
            self.labels.split_at(self.k + 1);
        for (datum, label) in first_k_data.iter().zip(first_k_labels.iter()) {
            let distance = Self::calc_distance(&point, datum);
            nearest_neighbors.push(NeighborData { distance, label: *label });
        }

        // find the k nearest neighbors
        for (datum, label) in remaining_data.iter().zip(remaining_labels.iter())
        {
            let distance = Self::calc_distance(&point, datum);

            let (farthest_neighbor_index, farthest_neighbor) = {
                let mut farthest_neighbor_index = 0;
                let mut farthest_neighbor =
                    match nearest_neighbors.get(farthest_neighbor_index) {
                        Some(neighbor) => *neighbor,
                        None => panic!(),
                    };

                for (index, near_neighbor) in
                    nearest_neighbors.iter().enumerate()
                {
                    if near_neighbor.distance > farthest_neighbor.distance {
                        farthest_neighbor_index = index;
                        farthest_neighbor = *near_neighbor;
                    }
                }

                (farthest_neighbor_index, farthest_neighbor)
            };

            if distance < farthest_neighbor.distance {
                nearest_neighbors.swap_remove(farthest_neighbor_index);
                nearest_neighbors
                    .push(NeighborData { distance, label: *label });
            }
        }

        // map the classes to the neighbor count
        let mut class_map: HashMap<Label, i32> = HashMap::with_capacity(self.k);
        for neighbor_data in nearest_neighbors {
            let count = match class_map.get(&neighbor_data.label) {
                Some(count) => *count + 1,
                None => 1,
            };

            class_map.insert(neighbor_data.label, count);
        }

        // find the class with the most close neighbors
        let class = {
            let mut class_map_iter = class_map.iter();
            let first_pair = match class_map_iter.next() {
                Some(pair) => pair,
                None => panic!(),
            };

            let mut nearest_neighbor_class = first_pair.0;
            let mut nearest_neighbor_count = first_pair.1;
            for (neighbor_class, neighbor_count) in class_map_iter {
                if neighbor_count > nearest_neighbor_count {
                    nearest_neighbor_class = neighbor_class;
                    nearest_neighbor_count = neighbor_count;
                }
            }

            *nearest_neighbor_class
        };

        class
    }

    fn predict_classes_thread(
        &self,
        points: &[Vec<f32>],
        results: &mut [Label],
    ) {
        for (index, point) in points.iter().enumerate() {
            results[index] = self.predict_class(point);
        }
    }

    pub fn predict_classes(
        &self,
        points: &Vec<Vec<f32>>,
        thread_count: usize,
    ) -> Vec<Label> {
        let mut results = vec![0; points.len()];

        let points_per_thread = points.len() / thread_count;

        thread::scope(|s| {
            let points_iter = points.chunks(points_per_thread);
            let results_iter = results.chunks_mut(points_per_thread);

            for (points_slice, results_slice) in points_iter.zip(results_iter) {
                s.spawn(|| {
                    self.predict_classes_thread(points_slice, results_slice);
                });
            }
        });

        results
    }
}

// todo: some testing
