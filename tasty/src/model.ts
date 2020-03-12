import * as ml5 from 'ml5';

export class TastyClassifier {

  classifier: any;

  constructor() {

  }

  async load(){
    const featureExtractor = await ml5.featureExtractor('MobileNet');
    const options = { numLabels: 2 };
    this.classifier = featureExtractor.classification(options);
  }

  async train(onStatus) {
    return this.classifier.train(onStatus);
  }
  
  async classify(file) {
    return new Promise((resolve, reject) => {
      this.classifier.classify(file,(err,result)=>{
        if (err)
          reject(err);
        else
          resolve(result);
      });
    });
  }

  async addImage(file:any , tag:string){
    return this.classifier.addImage(file,tag);
  }

}