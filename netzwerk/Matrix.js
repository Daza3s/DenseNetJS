class Matrix {
    constructor(iDim,jDim) {
        this.werte = new Array(iDim*jDim);
        
        this.j = false; //entfernen wenn batch training implementiert
        this.iDim = iDim;
        this.jDim = jDim;
    }

    init() {
        for(let index = 0;index < this.werte.length;index++) {
            this.werte[index] = 0;
        }
    }

    randomize(func) {
        func = func || (() => { return Math.random()*2-1; });
        for(let index = 0;index < this.werte.length;index++) {
            this.werte[index] = func();
        }
    }

    gleicheDim(m) {
        if(this.iDim === m.iDim && this.jDim === m.jDim) return true;

        return false;
    }

    at(i,j) {
        return this.werte[i * this.jDim + j];
    }

    set(i, j, wert) {
        this.werte[i * this.jDim + j] = wert;
    }

    add(m) {
        let erg = new Matrix(this.iDim, this.jDim);
        if(typeof m === 'number') {
            erg.werte = this.werte.map(x=>x+m);
            return erg;
        }
        if(!this.gleicheDim(m)) throw new Error("Wrong scale in matrix manipulation (addition)");

        erg.werte = this.werte.map((x,i)=> x + m.werte[i]);

        return erg;
    }

    sub(m) {
        let erg = new Matrix(this.iDim, this.jDim);
        if(typeof m === 'number') {
            erg.werte = this.werte.map(x=>x-m);
            return erg;
        }
        if(!this.gleicheDim(m)) throw new Error("Wrong scale in matrix manipulation (subtraktion)");

        erg.werte = this.werte.map((x,i)=> x - m.werte[i]);

        return erg;
    }

    dot(m) {
        if(m.j) return this.mult(m);
        let erg = new Matrix(this.iDim, this.jDim);
        if(typeof m === 'number') {
            erg.werte = this.werte.map(x=>x*m);
            return erg;
        }
        if(!this.gleicheDim(m)) throw new Error("Wrong scale in matrix manipulation (dotProdukt)");

        erg.werte = this.werte.map((x,i)=> x * m.werte[i]);

        return erg;
    }

    div(m) {
        let erg = new Matrix(this.iDim, this.jDim);
        if(typeof m === 'number') {
            erg.werte = this.werte.map(x=>x/m);
            return erg;
        }
        if(!this.gleicheDim(m)) throw new Error("Wrong scale in matrix manipulation (division)");

        erg.werte = this.werte.map((x,i)=> x / m.werte[i]);

        return erg;
    }

    mult(m) {
        if(typeof m === 'number') {
            let erg = new Matrix(this.iDim, this.jDim);
            erg.werte = this.werte.map(x=>x*m);
            return erg;
        }

        if(this.jDim != m.iDim) throw new Error("Wrong scale in matrix manipulation (crossProdukt)");

        let erg = new Matrix(this.iDim, m.jDim);
        for(let i = 0;i < this.iDim;i++) {
            for(let j = 0;j < m.jDim;j++) {
                let sum = 0;
                for(let index = 0;index < this.jDim;index++) {
                    sum += this.at(i,index) * m.at(index, j);
                }
                erg.set(i, j, sum);
            }
        }

        return erg;
    }

    costumMult(m, self) {
        if(self.jDim != m.iDim) throw new Error("Wrong scale in matrix manipulation (crossProdukt)");

        let erg = new Matrix(self.iDim, m.jDim);
        for(let i = 0;i < self.iDim;i++) {
            for(let j = 0;j < m.jDim;j++) {
                let sum = 0;
                for(let index = 0;index < this.jDim;index++) {
                    sum += self.at(i,index) * m.at(index, j);
                }
                erg.set(i, j, sum);
            }
        }
        return erg;
    }

    transpose() {
        let erg = new Matrix(this.jDim, this.iDim);
        for(let i = 0;i < this.iDim;i++) {
            for(let j = 0;j < this.jDim;j++) {
                erg.set( j , erg.jDim - 1 - i , this.at(i,j));
            }
        }
        return erg;
    }

    nest() {
        let erg = [];
        for(let i = 0;i < this.iDim;i++) {
            erg.push([]);
            for(let j = 0;j < this.jDim;j++) {
                erg[i].push(this.at(i,j));
            }
        }
        return erg;
    }

    toString() {
        let erg = '';
        let nested = this.nest();
        for(let i = 0;i < nested.length;i++) {
            erg += JSON.stringify(nested[i]) + '\n';
        }        
        return erg.slice(0,-1);
    }

    maxNorm() {
        let max = 0;

        for(let i = 0;i < this.iDim;i++) {
            let sum = 0;
            for(let j = 0;j < this.jDim;j++) {
                sum += this.at(i,j)**2;
            }

            sum = Math.sqrt(sum);

            if(sum > max) max = sum;
        }

        return max;
    }

    sum(func) {
        func = func || ((a)=>{ return a });
        let erg = 0;

        for(let i = 0;i < this.werte.length;i++) {
            erg += func(this.werte[i]);
        }
        
        return erg;
    }
}


export { Matrix }