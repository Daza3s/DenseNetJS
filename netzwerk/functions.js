import { Matrix } from "./Matrix.js";

let activations = {
    "relu": (m) => {
        let erg = new Matrix(m.iDim, m.jDim);
        erg.werte = m.werte.map(x => Math.max(0,x));
        return erg;
    },
    "reluA": (m) => {
        let erg = new Matrix(m.iDim, m.jDim);
        erg.werte = m.werte.map(x => x <= 0 ? 0 : 1);
        return erg;
    },
    "lrelu": (m) => {
        let erg = new Matrix(m.iDim, m.jDim);
        erg.werte = m.werte.map(x => Math.max(0.01*x,x));
        return erg;
    },
    "lreluA": (m) => {
        let erg = new Matrix(m.iDim, m.jDim);
        erg.werte = m.werte.map(x => x <= 0 ? 0.01 : 1);
        return erg;
    },
    "sigmoid": (m) => {
        let sigmoid = function(n) {
            return 1 / (1 + Math.exp(-n));
        }
        let erg = new Matrix(m.iDim, m.jDim);
        erg.werte = m.werte.map( x => sigmoid(x));
        return erg;
    },
    "sigmoidA": (m, sättigung = 0.1) => {
        let sigmoid = function(n) {
            return 1 / (1 + Math.exp(-n));
        }
        let erg = new Matrix(m.iDim, m.jDim);
        erg.werte = m.werte.map( x => sigmoid(x) * (1-sigmoid(x)) + sättigung);
        return erg;
    },
    "softmax": (m) => {
        let max = m.werte.reduce((a,b)=>Math.max(a,b));
        let exps = 0;
        let erg = new Matrix(m.iDim, m.jDim);
        for(let i = 0;i < m.werte.length;i++) {
            erg.werte[i] = Math.pow(Math.E, m.werte[i] - max);        //scaling for safety
            exps += erg.werte[i];
        }
        erg.werte = erg.werte.map(x => (x/exps) == 1 ? 1-Number.EPSILON : (x/exps) == 0 ? Number.EPSILON : (x/exps)); //safety never 0 or 1
        return erg;
    },
    "softmaxA": (m) => {
        let softmax = (m) => {
            let max = m.werte.reduce((a,b)=>Math.max(a,b));
            let exps = 0;
            let erg = new Matrix(m.iDim, m.jDim);
            for(let i = 0;i < m.werte.length;i++) {
                erg.werte[i] = Math.pow(Math.E, m.werte[i] - max);        //scaling for safety
                exps += erg.werte[i];
            }
            erg.werte = erg.werte.map(x => (x/exps) == 1 ? 1-Number.EPSILON : (x/exps) == 0 ? Number.EPSILON : (x/exps));
            return erg;
        }
        if(m.iDim !== 1) throw "Matrix not in vector form cant compute softmax derivative"
        let jMatrix = new Matrix(m.jDim, m.jDim);
        let soft = softmax(m);
        
        for(let j = 0;j < jMatrix.jDim;j++) {
            for(let i = 0;i < jMatrix.iDim;i++) {
                if(i == j) {
                    let s = soft.at(0,j);
                    let e = s*(1-s);
                    jMatrix.set(i,j, e );
                }else {
                    jMatrix.set(i,j, -soft.at(0,j)*soft.at(0,i) );
                }
            }
        }
        jMatrix.j = true;
        return jMatrix; //experiment

        let erg = new Matrix(1,m.jDim);
        for(let j=0;j<m.jDim;j++) {
            let sum = 0;
            for(let i = 0;i < jMatrix.iDim;i++) {
                sum += jMatrix.at(i, j);
            }
            erg.set(0, j, sum);
        }

        return erg;
    }

}

let loss = {
    "crossEntropy": (output, target) => {
        let loss = 0;
        for(let i = 0;i < output.werte.length;i++) {
            loss += Math.log(output.werte[i])*target.werte[i];
        }
        loss = -loss;
        return loss;
    },
    "crossEntropyA": (output, target) => {
        output.werte = output.werte.map(x=> {            
            return x < Number.EPSILON ? Number.EPSILON : x > 1-Number.EPSILON ? 1-Number.EPSILON : x; //Safety
        });
        let erg = new Matrix(output.iDim, output.jDim);
        for(let i=0;i < erg.werte.length;i++) {
            erg.werte[i] = - target.werte[i]/output.werte[i] + (1-target.werte[i])/(1-output.werte[i]);
        }
        return erg;
    },
    "mse": (output, target) => {
        let Etotal = 0;

        for(let i = 0;i < output.werte.length;i++) {
            let wert = output.werte[i];
            let ziel = target.werte[i];
            Etotal += (ziel - wert)*(ziel - wert);
        }

        Etotal = Etotal / 2;
        return Etotal;
    },
    "mseA": (output, target) => {
        return output.sub(target);
    }
    
}

export { activations, loss }