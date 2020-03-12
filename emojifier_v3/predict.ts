import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';
import * as use from '@tensorflow-models/universal-sentence-encoder';
import * as readlineSync from 'readline-sync';

const emojiMapping = ['😜','📸','😍','😂','😉','🎄','📷','🔥','😘','♥️','😁'];

async function loadSentenceEncoder() {
    return use.load()
}

async function encodeSentence(model, data ){
    const sentence = data.toLowerCase();
    return model.embed(sentence);
};

function loadModel(){
    return tf.loadLayersModel('file://./modelx/model.json')
}

function getEmoji(emoji){
    const emojiCode = tf.argMax(emoji, 1);
    
    console.log(emojiMapping[emojiCode.dataSync()[0]]);
}

async function predict() {
    const sentenceEncoder = await loadSentenceEncoder();
    const model = await loadModel();
    
    while(true){
        let answer = readlineSync.question('say something ');
        const input = await encodeSentence(sentenceEncoder, answer);
        const result = await model.predict(input as tf.Tensor2D);
        getEmoji(result);
    }

};

predict(); 