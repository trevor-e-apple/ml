use core::panic;
use std::collections::HashMap;

type Class = i64;

pub struct Knn {
    k: usize,
    pub data: Vec<(Class, Vec<f64>)>
}

impl Knn {
    pub fn new(k: usize, data: Vec<(Class, Vec<f64>)>) -> Self {
        if k == 0 {
            panic!();
        }
        Self {k, data}
    }

    // TODO: make this replaceable somehow (maybe with generics?)
    fn calc_distance(a: &[f64], b: &[f64]) -> f64 {
        let mut distance = 0.0;
        for (a_item, b_item) in a.iter().zip(b) {
            distance += (a_item - b_item).powf(2.0);
        }

        distance.sqrt()
    }

    pub fn predict_class(&self, point: Vec<f64>) -> Class {
        #[derive(Clone, Copy)]
        struct NeighborData {
            distance: f64,
            class: i64,
        }

        // initialize with first k neighbors
        let mut nearest_neighbors: Vec<NeighborData> = Vec::with_capacity(self.k);
        for datum in &self.data[0..self.k] {
            let class = datum.0;
            let distance = Self::calc_distance(&point, &datum.1);
            nearest_neighbors.push(NeighborData { distance, class });
        }

        // find the k nearest neighbors
        for datum in &self.data[self.k..] {
            let class = datum.0;
            let distance = Self::calc_distance(&point, &datum.1);

            let (farthest_neighbor_index, farthest_neighbor) = {
                let mut farthest_neighbor_index = 0;
                let mut farthest_neighbor = match nearest_neighbors.get(farthest_neighbor_index) {
                    Some(neighbor) => *neighbor,
                    None => panic!(),
                };

                for (index, near_neighbor) in nearest_neighbors.iter().enumerate() {
                    if near_neighbor.distance > farthest_neighbor.distance {
                        farthest_neighbor_index = index;
                        farthest_neighbor = *near_neighbor;
                    }
                }

                (farthest_neighbor_index, farthest_neighbor)
            };

            if distance < farthest_neighbor.distance {
                nearest_neighbors.swap_remove(farthest_neighbor_index);
                nearest_neighbors.push( NeighborData { distance, class });
            }
        }

        // map the classes to the neighbor count
        let mut class_map: HashMap<Class, i32> = HashMap::with_capacity(self.k);
        for neighbor_data in nearest_neighbors {
            let count = match class_map.get(&neighbor_data.class) {
                Some(count) => *count + 1,
                None => 1,
            };

            class_map.insert(neighbor_data.class, count);
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
}

// todo: some testing
