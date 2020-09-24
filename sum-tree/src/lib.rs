use pyo3::prelude::*;
use pyo3::{PyObjectProtocol, PySequenceProtocol};
use std::fmt::Write;

/// The SumTree data structure.
#[pyclass(module = "sum_tree")]
#[text_signature = "(len)"]
pub struct SumTree {
    tree: Box<[f64]>,
    len: usize,
    left_end: usize,
    right_sum: f64,
}

impl SumTree {
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn get(&self, index: usize) -> f64 {
        self.tree[index + self.len - 1]
    }

    pub fn set(&mut self, mut index: usize, value: f64) {
        index += self.len - 1;
        let change = value - self.tree[index];
        if index < self.left_end {
            self.right_sum += change;
        }
        self.tree[index] = value;
        while index != 0 {
            index = (index - 1) >> 1;
            self.tree[index] += change;
        }
    }

    fn tree_sum(&self, mut left: usize, mut right: usize) -> f64 {
        let mut res = 0.;
        while left < right {
            if left & 1 == 0 {
                res += self.tree[left];
                left += 1;
            }
            if right & 1 == 0 {
                right -= 1;
                res += self.tree[right];
            }
            left = (left - 1) >> 1;
            right = (right - 1) >> 1;
        }
        res
    }
}

#[pymethods]
impl SumTree {
    #[new]
    pub fn new(len: usize) -> Self {
        let mut left_end = 0;
        let mut i = len;
        while i != 0 {
            i >>= 1;
            left_end += 1;
        }
        left_end = (1 << left_end) - 1;
        Self {
            tree: vec![0.; (len << 1) - 1].into_boxed_slice(),
            left_end,
            len,
            right_sum: 0.,
        }
    }

    /// Returns the sum of all elements in the tree.
    #[text_signature = "($self)"]
    #[inline]
    pub fn total(&self) -> f64 {
        self.tree[0]
    }

    /// sums all elements in the range [left, right).
    #[text_signature = "($self, left, right)"]
    pub fn sum(&self, mut left: usize, mut right: usize) -> f64 {
        left += self.len - 1;
        right += self.len - 1;
        if left == 0 && right == self.len {
            self.total()
        } else if left < self.left_end && self.left_end < right {
            self.tree_sum(left, self.left_end) + self.tree_sum(self.left_end, right)
        } else {
            self.tree_sum(left, right)
        }
    }

    /// Searches the index of the prefix sum `s`.
    #[text_signature = "($self, s)"]
    pub fn sample(&self, s0: f64) -> (usize, f64) {
        let mut s = if self.right_sum < s0 {
            s0 - self.right_sum
        } else if s0 < 0. {
            panic!("s is out of bounds: {} < 0", s0)
        } else {
            s0 + self.total() - self.right_sum
        };
        let mut idx = 0;
        while idx < self.len - 1 {
            let left = (idx << 1) + 1;
            if s + f64::EPSILON < self.tree[left] {
                idx = left;
            } else {
                s -= self.tree[left];
                if self.tree[left + 1] < s {
                    panic!("s is out of bounds: {} < {}", self.total(), s0);
                }
                idx = left + 1;
            }
        }
        (idx + 1 - self.len, self.tree[idx])
    }

    #[text_signature = "($self, value)"]
    pub fn fill(&mut self, value: f64) {
        for i in 0..self.len {
            self.set(i, value);
        }
    }
}

#[pyproto]
impl PySequenceProtocol for SumTree {
    fn __len__(&self) -> usize {
        self.len
    }
    fn __getitem__(&self, mut index: isize) -> f64 {
        if index < 0 {
            index += self.len as isize;
        }
        self.get(index as usize)
    }
    fn __setitem__(&mut self, mut index: isize, value: f64) {
        if index < 0 {
            index += self.len as isize;
        }
        self.set(index as usize, value)
    }
}

#[pyproto]
impl PyObjectProtocol for SumTree {
    fn __repr__(&self) -> String {
        const REPR_ERROR: &str = "Error occurred while trying to write in String";
        let mut s = String::new();
        if self.len < 200 {
            write!(
                &mut s,
                "SumTree({}, {:?})",
                self.len,
                &self.tree[self.len - 1..]
            )
            .expect(REPR_ERROR);
        } else {
            write!(
                &mut s,
                "SumTree({}, [{}, {}, ..., {}])",
                self.len,
                self.get(0),
                self.get(1),
                self.get(self.len - 1),
            )
            .expect(REPR_ERROR);
        }
        s
    }
}

#[pymodule]
pub fn sum_tree(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SumTree>()?;
    Ok(())
}
