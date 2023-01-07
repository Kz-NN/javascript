export default class Matrix {
  rows;
  cols;
  data;

  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.data = new Array(rows).fill().map(() => Array(cols).fill(null));
  }

  static zeros(rows, cols) {
    let m = new Matrix(rows, cols);
    m.data = new Array(rows).fill().map(() => Array(cols).fill(0));

    return m;
  }

  static random(rows, cols) {
    let m = new Matrix(rows, cols);
    m.data = new Array(rows)
      .fill()
      .map(() => Array(cols).fill(Math.random() * 2 - 1));

    return m;
  }

  static from(data) {
    let m = new Matrix(data.length, data[0].length);
    m.data = data;

    return m;
  }

  static desirialize(serialized) {
    let data = JSON.parse(serialized);
    let m = new Matrix(data.rows, data.cols);

    m.data = data.data;

    return m;
  }

  serialized() {
    return JSON.stringify(this);
  }

  multiply(other) {
    if (this.cols !== other.rows)
      throw new Error(
        "Attempted to multiply by matrix of incorrect dimensions"
      );

    let res = Matrix.zeros(this.rows, other.cols);

    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < other.cols; j++) {
        let sum = 0.0;
        for (let k = 0; k < this.cols; k++) {
          sum += this.data[i][k] * other.data[k][j];
        }

        res.data[i][j] = sum;
      }
    }

    return res;
  }

  add(other) {
    if (this.rows !== other.rows || this.cols !== other.cols)
      throw new Error("Attempted to add matrix of incorrect dimensions");

    let res = Matrix.zeros(this.rows, this.cols);

    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        res.data[i][j] = this.data[i][j] + other.data[i][j];
      }
    }

    return res;
  }

  dotMultiply(other) {
    if (this.rows !== other.rows || this.cols !== other.cols)
      throw new Error(
        "Attempted to dot multiply matrix of incorrect dimensions"
      );

    let res = Matrix.zeros(this.rows, this.cols);

    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        res.data[i][j] = this.data[i][j] * other.data[i][j];
      }
    }

    return res;
  }

  subtract(other) {
    if (this.rows !== other.rows || this.cols !== other.cols)
      throw new Error("Attempted to subtract matrix of incorrect dimensions");

    let res = Matrix.zeros(this.rows, this.cols);

    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        res.data[i][j] = this.data[i][j] - other.data[i][j];
      }
    }

    return res;
  }

  map(func) {
    return Matrix.from(
      this.data.slice().map((row) => row.map((value) => func(value)))
    );
  }

  transpose() {
    let res = Matrix.zeros(this.cols, this.rows);

    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        res.data[j][i] = this.data[i][j];
      }
    }

    return res;
  }
}
