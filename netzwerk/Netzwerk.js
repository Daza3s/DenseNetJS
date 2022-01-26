import { Matrix } from "./Matrix.js";

class Netzwerk {
    /**
     * Creates new network with layers specified
     * @param {Array} layers 
     */
     constructor(layers) {

        this.sättigung = 0.01;
        this.lernRate = 0.3;
        this.d = 0.002;

        this.inputCount = layers[0];
        this.outputCount = layers[layers.length-1];
        this.layerCount = layers.length;
        this.layers = layers;
        this.hiddenLayerCount = this.layerCount-2;

        this.weights = [];
        this.biases = [];
        this.actFunc = [];

        this.lastWD = [];
        this.lastBD = [];

        for(let i = 0;i < this.layerCount-1;i++) {
            this.weights.push(new Matrix( this.layers[i] , this.layers[i+1] ));
            this.weights[i].randomize();
            this.biases.push(new Matrix( 1 , this.layers[i+1] ));
            this.biases[i].randomize();
            this.actFunc.push([this.act, this.actA]);

            this.lastWD.push(new Matrix( this.layers[i] , this.layers[i+1] ));
            this.lastWD[i].init();
            this.lastBD.push(new Matrix( 1 , this.layers[i+1] ));
            this.lastBD[i].init();
        }
    }

    setActivations(functions) {
        if(functions.length != this.weights.length) throw "Wrong amount of functions specified";
        
        if(functions[0].length != 2) throw "Did not specify activation and derivativ for each layer";
        
        this.actFunc = functions;
    }

    /**
     * 
     * @param {Number} n 
     * @returns {Number} Sigmoid(n)
     */
     sigmoid = function(n) {
        return 1 / (1 + Math.exp(-n));
    }

    /**
     * 
     * @param {Matrix} m 
     * @returns {Matrix} mapped with f_act
     */
    act = function(m) {
        let sigmoid = function(n) {
            return 1 / (1 + Math.exp(-n));
        }
        let erg = new Matrix(m.iDim, m.jDim);
        erg.werte = m.werte.map( x => sigmoid(x));
        return erg;
    }

    /**
     * 
     * @param {Matrix} m 
     * @returns {Matrix} mapped with f'act
     */
    actA = function(m) {
        let sättigung = this.sättigung || 0.01;
        let sigmoid = function(n) {
            return 1 / (1 + Math.exp(-n));
        }
        let erg = new Matrix(m.iDim, m.jDim);
        erg.werte = m.werte.map( x => sigmoid(x) * (1-sigmoid(x)) + sättigung);
        return erg;
    }
    
    /**
     * 
     * @param {Matrix} output 
     * @param {Matrix} target 
     * @returns {Matrix}
     */
    loss(output, target) {
        let erg = output.sub(target);
        return erg;
    }

    /**
     * 
     * @param {Function} func 
     */
    setLoss(func) {
        this.loss = func;
    }

    /**
     * 
     * @param {Matrix} output 
     * @param {Matrix} target 
     * @returns {Number}
     */
    errorTotal(output, target) {
        let Etotal = 0;

        for(let i = 0;i < output.werte.length;i++) {
            let wert = output.werte[i];
            let ziel = target.werte[i];
            Etotal += (ziel - wert)*(ziel - wert);
        }

        Etotal = Etotal / 2;
        return Etotal;
    }

    /**
     * 
     * @param {Function} func 
     */
    setErrorFunction(func) {
        this.errorTotal = func;
    }

    /**
     * 
     * @param {Matrix} input 
     * @returns {Matrix} output
     */
    predict = function(input) {
        let erg = input;
        for(let i = 0;i < this.layerCount-1;i++) {
            erg = erg.mult(this.weights[i]);
            erg = erg.add(this.biases[i]);
            erg = this.actFunc[i][0](erg);
        }
        return erg;
    }

    getGradients(input, target) {
        let net = [];
        let out = [];

        net[0] = input.mult(this.weights[0]).add(this.biases[0]);
        out[0] = this.actFunc[0][0](net[0]);

        for(let i = 1;i < this.layerCount-1;i++) {
            net.push(out[i-1].mult(this.weights[i]).add(this.biases[i]));
            out.push(this.actFunc[i][0](net[i]));
        }

        let output = out[out.length-1];

        /*Fehlerberechnung*/
        let Etotal = this.errorTotal(output, target);

        /**********/
        /*
            Deltaberechnung (d.h. der schwere Part)
        */
        let oDeltas = [];
        let wDeltas = [];
        let bDeltas = [];

        let l = this.loss(output, target);
        let a = this.actFunc[this.actFunc.length-1][1](net[net.length-1]);
        let d = l.dot(a);
        let deltaO = d;
        let deltaW = (out[out.length-2] || input).transpose().mult( deltaO ).mult(this.lernRate);

        oDeltas.unshift(deltaO);
        wDeltas.unshift(deltaW);
        bDeltas.unshift(deltaO.mult(this.lernRate));

        if(this.hiddenLayerCount > 0) {


            for(let i = 0;i < this.weights.length-2;i++) {
                let evIndex = net.length-1-2-i;

                //nullte 0 delta da das letzte geshifted wurde
                let weightedDelta = oDeltas[0].mult( this.weights[this.weights.length-1-i].transpose() );

                deltaO = this.actFunc[evIndex+1][1](net[evIndex+1]).dot( weightedDelta );  //????????
                deltaW = out[evIndex].transpose().mult( deltaO ).mult(this.lernRate);

                oDeltas.unshift(deltaO);
                wDeltas.unshift(deltaW);
                bDeltas.unshift(deltaO.mult(this.lernRate));
            }

        

            let w = this.weights[1].transpose();

            //nullte 0 delta da das letzte geshifted wurde
            deltaO = this.actFunc[0][1](net[0]).dot( oDeltas[0].mult(w)); //??????
            deltaW = input.transpose().mult(deltaO).mult(this.lernRate);

            oDeltas.unshift(deltaO);
            wDeltas.unshift(deltaW);
            bDeltas.unshift(deltaO.mult(this.lernRate));
        }

        return [wDeltas, bDeltas, output, Etotal];
    }

    train(input, target, options = false) {
        let erg = this.getGradients(input, target);

        let wDeltas = erg[0];
        let bDeltas = erg[1];
        let output = erg[2];
        let Etotal = erg[3];

        if(options) {
            if(options.weightDecay) {
                let lastW = this.weights[this.weights.length-1];
                let sqS = ((a)=>{return a*a;}); //square Sum
                Etotal += lastW.sum(sqS) * options.weightDecay;//might need to implement more
            }
            if(options.loop) {
                for(let i = 0;i < wDeltas.length;i++) {
                    if(options.normClipping) {
                        let wNorm = wDeltas[i].maxNorm();
                        if(wNorm > options.normClipping) {
                            wDeltas[i].werte = wDeltas[i].werte.map(x=>{
                                let erg = options.normClipping * x/wNorm;
                                return erg;
                            });
                        }

                        let bNorm = bDeltas[i].maxNorm();
                        if(bNorm > options.normClipping) {
                            bDeltas[i].werte = bDeltas[i].werte.map(x=>{
                                let erg = options.normClipping * x/bNorm;
                                return erg;
                            });
                        }
                    }

                    if(options.momentum || options.weightDecay) {
                        wDeltas[i].werte = wDeltas[i].werte.map((x,index)=> {
                            let erg = x;
                            if(options.momentum) erg += (1-options.momentum) * this.lastWD[i].werte[index];
                            if(options.weightDecay) erg -= options.weightDecay * this.lastWD[i].werte[index];//?
                            return erg;
                        });

                        bDeltas[i].werte = bDeltas[i].werte.map((x,index)=> {
                            let erg = x;
                            if(options.momentum) erg += (1-options.momentum) * this.lastBD[i].werte[index];
                            if(options.weightDecay) erg -= options.weightDecay * this.lastBD[i].werte[index]; //?
                            return erg;
                        });

                        this.lastWD[i].werte = [...wDeltas[i].werte];
                        this.lastBD[i].werte = [...bDeltas[i].werte];

                    }  
                }
            }
        }

        for(let i = 0;i < this.weights.length;i++) {
            this.weights[i] = this.weights[i].sub(wDeltas[i]);
            this.biases[i] = this.biases[i].sub(bDeltas[i]);
        }

        return Etotal;
    }

    trainSet(inputs, targets, rounds = 1000, opts = false) {
        let rIndex = 0;
        let inpL = inputs[0].length;
        let tarL = targets[0].length;
        let errorSum = 0;

        if(opts.normClipping || opts.momentum || opts.weightDecay) {
            opts.loop = true;
        }

        while(rIndex < rounds) {
            rIndex++;

            for(let i = 0;i < inputs.length;i++) {
                let inp = new Matrix(1, inpL);
                let targ = new Matrix(1, tarL);

                inp.werte = inputs[i];
                targ.werte = targets[i];

                let err = this.train(inp, targ, opts);
                errorSum += err;

                if(i % inputs.length/100 === 0) {
                    console.log(i/100, "von", inputs.length/100,"Zwischenschritten")
                }
            }
            
        }
        return errorSum / (rounds*inputs.length);
    }


    /*JSON TODO: Safe and Extract activation function */

    toJSON(slim = false) {
        let erg = {
            weights: this.weights,
            biases: this.biases,
            actFunc: this.actFunc
        };

        if(slim) return erg;

        erg.loss = this.loss;
        erg.errorTotal = this.errorTotal;
        erg.lernRate = this.lernRate;

        if(this.sättigung) erg.sättigung = this.sättigung;
        if(this.d) erg.d = this.d;

        return erg;
    }

    fromJSON(json) {
        this.weights = json.weights;
        this.biases = json.biases;
        this.actFunc = json.actFunc;
        this.loss = json.loss;
        this.errorTotal = json.errorTotal;
        this.sättigung = json.sättigung;
        this.lernRate = json.lernRate;
        this.d = json.d;
    }
    
}

export { Netzwerk }