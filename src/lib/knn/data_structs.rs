pub type Label = i64; // TODO: move

pub struct KnnCommon {
    pub k: usize,
    pub data: Vec<Vec<f32>>,
    pub labels: Vec<Label>,
}
