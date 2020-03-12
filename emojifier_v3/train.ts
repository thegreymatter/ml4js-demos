import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';
import * as use from '@tensorflow-models/universal-sentence-encoder';
import * as csv from 'csvtojson';

async function loadTrainingData() {
    return  (await csv().fromFile('Train.csv'))
      .map(x=>({input:x.TEXT,output: parseInt(x.Label)}))
      .map(x=>({input:x.input,output: x.output})).slice(0,5000);
}

async function loadSentenceEncoder() {
    return use.load()
}

async function encodeData(model, data ){
    const sentences = data.map(comment => comment.toLowerCase());
    return await model.embed(sentences);
}

function createModel(){
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [512], activation: 'sigmoid', units: 16}));
    model.add(tf.layers.dense({ inputShape: [16], activation: 'sigmoid', units: 16}));
    model.add(tf.layers.dense({inputShape: [16], activation: 'sigmoid', units: 16}));

    model.compile({
        loss: 'meanSquaredError',
        optimizer: tf.train.adam(.06),
    });
    return model;
}


async function train() {
    const rawdata = await loadTrainingData();
    const sentenceEncoder = await loadSentenceEncoder();
    const model = createModel();
    //encodes every sentence to 512 long array 
    const input = await encodeData(sentenceEncoder, rawdata.map(x=>x.input));
    //encode every emoji id to a one-hot array 3 -> [0,0,0,1,0,0,...]
    const output = tf.oneHot(rawdata.map(x=>x.output),16);

    await model.fit(input as tf.Tensor2D, output, { epochs: 100 });
    model.save('file://./modelx')
};

train(); 