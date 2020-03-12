const brain = require('brain.js');
const fs = require('fs');

const net = new brain.recurrent.LSTM();
const letItGo = `The snow glows white on the mountain tonight
Not a footprint to be seen
A kingdom of isolation
And it looks like I'm the queen
The wind is howling like this swirling storm inside
Couldn't keep it in, heaven knows I've tried
Don't let them in, don't let them see
Be the good girl you always have to be
Conceal, don't feel, don't let them know
Well, now they know
Let it go, let it go
Can't hold it back anymore
Let it go, let it go
Turn away and slam the door
I don't care what they're going to say
Let the storm rage on
The cold never bothered me anyway
Let it go, let it go
Can't hold it back anymore
Let it go, let it go
Turn away and slam the door
Let it go
Let it go
Let it go
Let it go
It's funny how some distance makes everything seem small
And the fears that once controlled me can't get to me at all
It's time to see what I can do
To test the limits and break through
No right, no wrong, no rules for me
I'm free
Let it go, let it go
I am one with the wind and sky
Let it go, let it go
You'll never see me cry
Here I stand and here I stay
My power flurries through the air into the ground
My soul is spiraling in frozen fractals all around
And one thought crystallizes like an icy blast
I'm never going back, the past is in the past
Let it go
The cold never bothered me anyway
Let it go, let it go
And I'll rise like the break of dawn
Let it go, let it go
That perfect girl is gone
Here I stand in the light of day
Let the storm rage on
the cold never bothered me anyway`

function createSamples(input) {
  input = input.toLowerCase();
  let i = 0;
  const segmentSize = 128;
  const samples = [];

  while (i + segmentSize < input.length) {
    let sample = input.slice(i, segmentSize + i);
    samples.push(sample);
    i = i + 10;
  }
  return samples;
}

net.train(
  createSamples(letItGo),
  {
    logPeriod: 1,
    iterations: 30,
    hiddenLayers: [3],
    log: details => console.log(details),

  });
net.maxPredictionLength = 300;

const json = net.toJSON()
fs.writeFileSync('trained-net_v2.json', JSON.stringify(json));
