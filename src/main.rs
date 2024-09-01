extern crate nalgebra as na;
use na::{DMatrix, DVector};

struct Matrix {
    data : DMatrix<f32>,
    rows: i32,
    cols: i32
}

struct Vector {
    data : DVector<f32>,
    cols : i32
}

struct Activation {
    forward : fn (Vector) -> Vector,
    backward : fn(Vector) -> Matrix
} 

fn max(val : f32, comp : f32, out : f32, default : f32) -> f32 {
    if val >= comp {
	out
    } else { default }
}

fn relu_unit(x : f32) -> f32 {
    max(x, 0.0, x, 0.0)
}

fn drelu_unit(x: f32) -> f32 {
    max(x, 0.0, 1.0, 0.0)
}



fn relu(v : Vector) -> Vector {
    Vector {
	data : v.data.map(relu_unit),
	cols : v.cols
    }
}

fn drelu(v : Vector) -> Matrix {
    Matrix {
	data : DMatrix::from_diagonal(&v.data.map(drelu_unit)),
	cols : v.cols,
	rows : v.cols
    }
}

struct Layer {
    bias : Vector,
    weights : Matrix,
    activation : Activation
}

fn make_standard_Layer(bias : Vector, weights : Matrix) -> Option<Layer> {
    if bias.cols == weights.cols {
	Some(Layer {
	    bias,
	    weights,
	    activation : Activation {
		forward : relu,
		backward : drelu
	    }
	})
    } else { None }
}

pub trait Model {
    fn forward(&self,v : Vector) -> Vector;
    fn backward(&self, v : Vector) -> Vector;
}

impl Model for Layer {
    fn forward(&self, v : Vector) -> Vector {
	(self.activation.forward)((self.weights.data * v.data) + self.bias.data) 
    }

    fn backward(&self, v : Vector) -> Vector {
	(self.activation.backward)(self.weights.data * v.data + self.bias.data) * self.weights.data
    }
}


fn main() {
    // A matrix with three lines and four columns.
    // We chose values such that, for example, 23 is at the row 2 and column 3.
    

    println!("Hello, world!");
}
