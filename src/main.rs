extern crate nalgebra as na;
use na::{DMatrix, DVector};

type Vector = DVector<f32>;
type Matrix = DMatrix<f32>;

struct Activation {
    forward : fn (Vector) -> Vector,
    backward : fn (Vector) -> Matrix
} 

fn max(val : f32, comp : f32, out : f32, default : f32) -> f32 {
    if val >= comp {
	out
    } else { default }
}

fn relu_scalar(x : f32) -> f32 {
    max(x, 0.0, x, 0.0)
}

fn drelu_scalar(x: f32) -> f32 {
    max(x, 0.0, 1.0, 0.0)
}


fn relu(v : Vector) -> Vector {
	v.map(relu_scalar)
}

fn drelu(v : Vector) -> Matrix {
    DMatrix::from_diagonal(&v.map(drelu_scalar))
}

struct Layer {
    bias : Vector,
    weights : Matrix,
    activation : Activation
}

fn make_standard_layer(bias : Vector, weights : Matrix) -> Layer {
    Layer {
	    bias,
	    weights,
	    activation : Activation {
		forward : relu,
		backward : drelu
	    }
	}
}

pub trait Model {
    fn forward(&self,v : Vector) -> Vector;
    fn backward(&self, v : Vector) -> Matrix;
}

impl Model for Layer {
    fn forward(&self, v : Vector) -> Vector {
	let linear_part = (self.weights.clone() * v.clone()) + self.bias.clone();
	(self.activation.forward)(linear_part)
    }

    fn backward(&self, v : Vector) -> Matrix {
	let linear_part = self.weights.clone() * v.clone() + self.bias.clone();
	(self.activation.backward)(linear_part) * self.weights.clone()
    }
}

fn main() {
    // A matrix with three lines and four columns.
    // We chose values such that, for example, 23 is at the row 2 and column 3.
    

    println!("Hello, world!");
}
