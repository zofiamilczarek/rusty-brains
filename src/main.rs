extern crate nalgebra as na;
use na::{DMatrix, DVector};


type Vector = DVector<f32>;
type Matrix = DMatrix<f32>;

struct Activation {
    act_forward : fn (Vector) -> Vector,
    act_backward : fn (Vector) -> Matrix
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

struct Linear {
    bias : Vector,
    weights : Matrix
}

type Objective = ();

pub trait BProp {
    fn forward(&self,v : Vector) -> Vector;
    fn backward(&self, v : Vector) -> Matrix;
}

impl BProp for Linear {
    fn forward(&self, v : Vector) -> Vector {
	(self.weights.clone() * v.clone()) + self.bias.clone()
    }

    fn backward(&self, _v : Vector) -> Matrix {
	self.weights.clone()
    }
}

impl BProp for Activation {
    fn forward(&self, v : Vector) -> Vector {
	(self.act_forward)(v)
    }

    fn backward(&self, v : Vector) -> Matrix {
	(self.act_backward)(v)
    }
}

const RELU : Activation = Activation {
    act_forward : relu,
    act_backward : drelu
};

struct Model {
    sequence : Vec<Box<dyn BProp>>,
    objective : Objective
}

// Idea implement a mutable model with append, etc

fn make_Model() {
}
 

impl BProp for Model {
    fn forward(&self, v : Vector) -> Vector {
	 (&self.sequence).into_iter().fold(v, |acc, f| f.forward(acc))
    }

    fn backward(&self, v : Vector) -> Matrix {
	// IDEA:
	// (ders, loss) = fs.fold(([],v), | (acc, val), f | (acc.unshift(f.backward(val), f.forward(val))))
	// (totalders, _) = ders.fold(([], ID), |(acc, curr), d| let v = val * d in (acc.append(v), v)) 
    }
}



fn main() {
    // A matrix with three lines and four columns.
    // We chose values such that, for example, 23 is at the row 2 and column 3.
    

    println!("Hello, world!");
}
