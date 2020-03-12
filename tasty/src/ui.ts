import { TastyClassifier } from './model';
import $ from 'jquery';

let predictFile= undefined;

function readFileAsync(file) {
    return new Promise((resolve, reject) => {
        let reader = new FileReader();

        reader.onload = () => {
            let img = new Image()
            img.src = reader.result as string;
            resolve(img);
        };

        reader.onerror = reject;
        reader.readAsDataURL(file);
    })
}

const classifier = new TastyClassifier();

async function setup() {
    await classifier.load();
    $('.train').show();
    setupButtons();
}

function status(state) {
    $('#train-status').text('error rate' + state);
 }

function setupButtons() {

    $('#train').click(async function () {
      //  $('#train').prop('disabled', true);
        $('#train').text('Training');
        await classifier.train(status);
        $('.train').hide();
        $('.predict').show();
        
    });

    $('#tasty_images').change(async (e) => {
        if (e.target.files) {
            for (let i = 0; i < e.target.files.length; i++) {
                const file = await readFileAsync(e.target.files[i]);
                await classifier.addImage(file, 'tasty');
            }
        }
    });

    $('#nasty_images').change(async (e) => {
        if (e.target.files) {
            for (let i = 0; i < e.target.files.length; i++) {
                const file = await readFileAsync(e.target.files[i]);
                await classifier.addImage(file, 'nasty');
            }
        }
    });

    $('#classify_image').change(async (e) => {
        if (e.target.files) {
            const file = (await readFileAsync(e.target.files[0])) as any;
            predictFile = file;
            $('#image_preview').attr('src',file.src);
            $('#predict').attr('disabled',false);
        }
    });

    $('#predict').click(async function () {
       const result = await classifier.classify(predictFile);
       $('#result').text(result[0].label);
       $('#confidence').text(result[0].confidence.toFixed(2) * 100 + '%');
    });
}

setup();