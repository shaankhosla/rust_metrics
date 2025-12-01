#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Reduction {
    Sum,
    #[default]
    Mean,
    Max,
    Min,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MetricAggregator {
    reduction: Reduction,
    total: usize,
    sum: f64,
    min: Option<f64>,
    max: Option<f64>,
}

impl MetricAggregator {
    pub fn new(reduction: Reduction) -> Self {
        Self {
            reduction,
            total: 0,
            sum: 0.0,
            min: None,
            max: None,
        }
    }

    pub fn update(&mut self, value: f64) {
        self.total += 1;
        self.sum += value;
        self.min = Some(self.min.map_or(value, |m| m.min(value)));
        self.max = Some(self.max.map_or(value, |m| m.max(value)));
    }
    pub fn reset(&mut self) {
        self.total = 0;
        self.sum = 0.0;
        self.min = None;
        self.max = None;
    }

    pub fn compute(&self) -> Option<f64> {
        if self.total == 0 {
            return None;
        }
        match self.reduction {
            Reduction::Sum => Some(self.sum),
            Reduction::Mean => Some(self.sum / self.total as f64),
            Reduction::Min => self.min,
            Reduction::Max => self.max,
        }
    }
}
