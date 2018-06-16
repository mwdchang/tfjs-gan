// Input params
const BATCH = 200;
const SIZE = 28;
const INPUT_SIZE = SIZE*SIZE;
const SEED_SIZE = 40;
const ONES = tf.ones([BATCH, 1]);
const ONES_PRIME = tf.ones([BATCH, 1]).mul(tf.scalar(0.98));
const ZEROS = tf.zeros([BATCH, 1]);


// Generator and discrimantor params
const DISCRIMINATOR_LEARNING_RATE = 0.006;
const GENERATOR_LEARNING_RATE = 0.009;
const dOptimizer = tf.train.sgd(DISCRIMINATOR_LEARNING_RATE);
const gOptimizer = tf.train.sgd(GENERATOR_LEARNING_RATE);


// Helper functions
const varInitNormal = (shape, mean=0, std=0.1) => tf.variable(tf.randomNormal(shape, mean, std));
const seed  = (s=BATCH) => tf.randomNormal([s, SEED_SIZE], 0, 3.0);


// Network arch for generator
const G1w = varInitNormal([SEED_SIZE, 140]);
const G1b = varInitNormal([140]);
const G2w = varInitNormal([140, 80]);
const G2b = varInitNormal([80]);
const G3w = varInitNormal([80, INPUT_SIZE]);
const G3b = varInitNormal([INPUT_SIZE]);

// Network arch for discriminator
const D1w = varInitNormal([INPUT_SIZE, 200]);
const D1b = varInitNormal([200]);
const D2w = varInitNormal([200, 90]);
const D2b = varInitNormal([90]);
const D3w = varInitNormal([90, 1]);
const D3b = varInitNormal([1]);




////////////////////////////////////////////////////////////////////////////////
// 
////////////////////////////////////////////////////////////////////////////////
function gen(xs) {
  const l1 = tf.leakyRelu(xs.matMul(G1w).add(G1b));
  const l2 = tf.leakyRelu(l1.matMul(G2w).add(G2b));
  const l3 = tf.tanh(l2.matMul(G3w).add(G3b));
  return l3;
}

function disReal(xs) {
  const l1 = tf.leakyRelu(xs.matMul(D1w).add(D1b));
  const l2 = tf.leakyRelu(l1.matMul(D2w).add(D2b));
  const logits = l2.matMul(D3w).add(D3b);
  const output = tf.sigmoid(logits);
  return [logits, output];
}

function disFake(xs) {
  return disReal(gen(xs));
}


// Copied from tensorflow
function sigmoidCrossEntropyWithLogits(target, output) {
    return tf.tidy(function () {
        var maxOutput = tf.maximum(output, tf.zerosLike(output));
        var outputXTarget = tf.mul(output, target);
        var sigmoidOutput = tf.log(tf.add(tf.scalar(1.0), tf.exp(tf.neg(tf.abs(output)))));
        var result = tf.add(tf.sub(maxOutput, outputXTarget), sigmoidOutput);
        return result;
    });
}


async function trainBatch(realBatch, fakeBatch) {
  const dcost = dOptimizer.minimize(() => {
    const [logitsReal, outputReal] = disReal(realBatch);
    const [logitsFake, outputFake] = disFake(fakeBatch);


    const lossReal = sigmoidCrossEntropyWithLogits(ONES_PRIME, logitsReal);
    const lossFake = sigmoidCrossEntropyWithLogits(ZEROS, logitsFake);
    return lossReal.add(lossFake).mean();
  }, true, [D1w, D1b, D2w, D2b, D3w, D3b]);
  await tf.nextFrame();

  const gcost = gOptimizer.minimize(() => {
    const [logitsFake, outputFake] = disFake(fakeBatch);

    const lossFake = sigmoidCrossEntropyWithLogits(ONES, logitsFake);
    return lossFake.mean();
  }, true, [G1w, G1b, G2w, G2b, G3w, G3b]);
  await tf.nextFrame();
  return [dcost, gcost];
}


