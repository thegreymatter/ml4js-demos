let featureExtractor;
let classifier;
let loss;

function setup() {
  console.log('hi');

  featureExtractor = ml5.featureExtractor('MobileNet');
  const options = { numLabels: 2 };
  classifier = featureExtractor.classification(options);
  setupButtons();
}

function modelReady() {
  select('#modelStatus').html('MobileNet Loaded!');
}

async function classify() {
  const img = select('#classify_image');
console.log(img.elt)
  let file = await readFileAsync(img.elt.files[0]);
  const preview = select('#image_preview');
  preview.elt.src=file.src;

  classifier.classify(file,gotResults);
}

function setupButtons() {

  train = select('#train');
  train.mousePressed(function() {
    classifier.train(function(lossValue) {
      if (lossValue) {
        loss = lossValue;
        select('#loss').html('Loss: ' + loss);
      } else {
        select('#loss').html('Done Training! Final Loss: ' + loss);
      }
    }).then(x=>{select('.train').toggleClass('hidden');
    select('.predict').toggleClass('hidden');
  });
  });

  // Predict Button
  buttonPredict = select('#buttonPredict');
  buttonPredict.mousePressed(classify);

  timg = select('#tasty_images');
  timg.elt.addEventListener('change',async (e)=>{
    if (e.target.files) {
        for(let i=0;i<e.target.files.length;i++){
        let file = await readFileAsync(e.target.files[i]);
        let x = await classifier.addImage(file,'tasty');
        console.log('loaded tasty')
    }
}
  });

  nimg = select('#nasty_images');
  nimg.elt.addEventListener('change',async (e)=>{
    if (e.target.files) {
        for(let i=0;i<e.target.files.length;i++){
       
            let reader = new FileReader();
            let file = await readFileAsync(e.target.files[i]);
            let x = await classifier.addImage(file,'nasty');
            console.log('loaded nasty')
    }
    }
  });

}

function readFileAsync(file) {
    return new Promise((resolve, reject) => {
      let reader = new FileReader();
  
      reader.onload = () => {
        let img = new Image()
        img.src = reader.result;
        resolve(img);
      };
  
      reader.onerror = reject;
  
      reader.readAsDataURL(file);
    })
  }

  // Show the results
function gotResults(err, results) {
    // Display any error
    if (err) {
      console.error(err);
    }
    if (results && results[0]) {
      select('#result').html(results[0].label);
      select('#confidence').html(results[0].confidence.toFixed(2) * 100 + '%');
    }
  }

  setup()