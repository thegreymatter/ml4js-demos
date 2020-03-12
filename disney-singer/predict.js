const brain = require('brain.js');
const fs = require('fs');
const readlineSync = require('readline-sync');

const json = JSON.parse(fs.readFileSync('trained-net.json'))

const network = new brain.recurrent.LSTM();
network.fromJSON(json)

while (true) {
    let answer = readlineSync.question('say something ');
    const output = network.run(answer,true,0.8);
    console.log(`-----------`);
    console.log(`${answer} ${output}`);
    console.log(`-----------`);

}
