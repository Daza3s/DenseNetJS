import { Matrix } from "./netzwerk/Matrix.js";
import { Netzwerk } from "./netzwerk/Netzwerk.js";
import { activations, loss } from "./netzwerk/functions.js";
import * as MNIST from "./netzwerk/src/mnist.js";

let reluPair = [activations.lrelu,activations.lreluA];
let softmaxPair = [activations.softmax, activations.softmaxA];
let sigmoidPair = [activations.sigmoid, activations.sigmoidA];

//Mnist formating
let mnist = window.mnist;
let set = mnist.set(1000, 9000);

let inps = [];
let outs = [];

for(let i = 0;i < set.training.length;i++) {
    inps.push(set.training[i].input.map(x => { x-0.5  }));
    outs.push(set.training[i].output);
}


let netz = new Netzwerk([28*28,500,300,200,10]);

netz.lernRate = 0.01;

let options = {
    momentum: 0.9,
    normClipping: 1,
    weightDecay: 0.01
}

netz.setActivations([reluPair,reluPair,reluPair,softmaxPair]);

netz.setErrorFunction(loss.crossEntropy);
netz.setLoss(loss.crossEntropyA);



function trainingStep() {
    let err = netz.trainSet(inps, outs, 1, options);
    document.getElementById("lastError").innerHTML = "Last " + document.getElementById("Error").innerHTML;
    document.getElementById("Error").innerHTML = "Error: "+err;
    console.log("Finished training with Error: ");
    console.log(err);
}

window.trainingStep = trainingStep;


let c = document.createElement("canvas");
let cc = c.getContext("2d");
       
window.scale = function() {
    let canvas = document.getElementById("paint");
    cc.fillStyle = "black";
    cc.fillRect(0,0,28,28);
    cc.scale(28/canvas.width,28/canvas.height);
    cc.drawImage(canvas, 0, 0);
    cc.scale(canvas.width/28,canvas.height/28);
    let imgData = cc.getImageData(0,0,28,28);
    let data = [];
    for(let i = 0;i < 28*28;i++) {
        data.push(imgData.data[i*4]/255);
    }
    let prid = document.getElementById("predictPaint");
    let matrix = new Matrix(1, netz.inputCount);
    matrix.werte = data;
    let erg = netz.predict(matrix);
    let num = 0;
    for(let i = 0;i < erg.werte.length;i++) {
        if(erg.werte[i] > erg.werte[num]) num = i;
    }
    prid.innerHTML = `Prediction of drawing: ${num}`;
}

function predict() {
    let prid = document.getElementById("predictMNIST");
    let val09 = parseInt(Math.random()*9)
    var digit = mnist[val09].get();
    let canvas = document.getElementById('mnistPic');
    canvas.setAttribute('width', window.getComputedStyle(canvas, null).getPropertyValue("width"));
    canvas.setAttribute('height', window.getComputedStyle(canvas, null).getPropertyValue("height"));
    var context = canvas.getContext('2d');
    mnist.draw(digit, context); // draws a '1' mnist digit in the canvas
    context.scale(canvas.width/28,canvas.height/28);
    context.drawImage(canvas, 0, 0);
    context.scale(1,1);
    let matrix = new Matrix(1, netz.inputCount);
    matrix.werte = digit;
    let erg = netz.predict(matrix);
    let num = 0;
    for(let i = 0;i < erg.werte.length;i++) {
        if(erg.werte[i] > erg.werte[num]) num = i;
    }

    prid.innerHTML = `Prediction: ${num}`;
}

window.predict = predict;