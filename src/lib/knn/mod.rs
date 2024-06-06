mod cpu_implementation;
mod data_structs;

#[cfg(target_os = "macos")]
mod metal_implementation;
#[cfg(target_os = "macos")]
pub use metal_implementation::Knn;

pub use data_structs::Label;

pub trait KnnLib {
    fn new(k: usize, data: Vec<Vec<f32>>, labels: Vec<Label>) -> Self;
    fn add_test(library_path: &String, num_elements: u32);
    fn predict_class(&self, point: &Vec<f32>) -> Label;
    fn predict_classes(
        &self,
        points: &Vec<Vec<f32>>,
        thread_count: usize,
    ) -> Vec<Label>;
}
