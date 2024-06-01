pub type Label = i64;

pub struct Knn {
    k: usize,
    pub data: Vec<Vec<f32>>,
    pub labels: Vec<Label>,
}
