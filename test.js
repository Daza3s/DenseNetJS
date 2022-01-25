import { Matrix } from "./netzwerk/Matrix.js";
import { Netzwerk } from "./netzwerk/Netzwerk.js";
import { activations, loss } from "./netzwerk/functions.js";

import * as fs from "fs";

//xOr

let reluPair = [activations.lrelu,activations.lreluA];
let softmaxPair = [activations.softmax, activations.softmaxA];
let sigmoidPair = [activations.sigmoid, activations.sigmoidA];


let x00 = new Matrix(1,2);
x00.werte = [1,0];
let x01 = new Matrix(1,2);
x01.werte = [0,1];
let x10 = new Matrix(1,2);
x10.werte = [0,1];
let x11 = new Matrix(1,2);
x11.werte = [1,0];

let random = new Matrix(2,2);
random.randomize();

let random23 = new Matrix(2,3);
random23.randomize();

let testNetz = new Netzwerk([2,10,1]);

let testNetz2 = new Netzwerk([2,10,10,2]);

testNetz.lernRate = 0.5;

let options = false;
let options2 = {
    momentum: 0.5,
    normClipping: 1,
    weightDecay: 0.01
}

testNetz2.setActivations([reluPair,reluPair,softmaxPair]);

testNetz2.setErrorFunction(loss.crossEntropy);
testNetz2.setLoss(loss.crossEntropyA);

let inps = [
    [-0.5,-0.5],
    [-0.5,0.5],
    [0.5,-0.5],
    [0.5,0.5]
];

let outs = [
    [0],
    [1],
    [1],
    [0]
];



let predictMatrix = new Matrix(1,2);

predictMatrix.werte = [0.5,0.1];

let p0 = new Matrix(1,2);
let p1 = new Matrix(1,2);
let p2 = new Matrix(1,2);
let p3 = new Matrix(1,2);

p0.werte = [-0.5,-0.5];
p1.werte = [-0.5,0.5];
p2.werte = [0.5,-0.5];
p3.werte = [0.5,0.5];


let t0 = new Matrix(1,2);
let t1 = new Matrix(1,2);
let t2 = new Matrix(1,2);
let t3 = new Matrix(1,2);

t0.werte = [1,0];
t1.werte = [0,1];
t2.werte = [0,1];
t3.werte = [1,0];

let outs2 = [
    [1,0],
    [0,1],
    [0,1],
    [1,0]
]

testNetz.trainSet(inps, outs, 1000, options);

console.log(testNetz.predict(p0).werte);
console.log(testNetz.predict(p1).werte);
console.log(testNetz.predict(p2).werte);
console.log(testNetz.predict(p3).werte);

console.log("");

testNetz2.lernRate = 0.01;

testNetz2.trainSet(inps, outs2, 10000, options2);


console.log(testNetz2.predict(p0).werte);
console.log(testNetz2.predict(p1).werte);
console.log(testNetz2.predict(p2).werte);
console.log(testNetz2.predict(p3).werte);

let data = JSON.stringify(testNetz2.toJSON(true));

fs.writeFileSync("xOR-slim-Modle-2-10-10-2.json", data);


//console.dir(testNetz2, {depth: 10});