//ea3.js

/************************************************************************ event listeners************************************************/
//submit chat button
document.getElementById('send-button').addEventListener('click', sendMessage);

//submit chat via Enter (not Shift+Enter)
document.getElementById('chat-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

/*************************sending a message into chat**********************************/
function sendMessage() {
    var input = document.getElementById('chat-input');
    var chatSpacing = document.getElementById('chat-spacing');

    // Only send the message if the input is not empty
    if (input.value.trim() !== '') {
        // Create a new chat-history element
        var newChatHistory = document.createElement('div');
        newChatHistory.id = 'chat-history';
        newChatHistory.innerHTML = '<p>' + input.value + '</p>';

        // Add the new chat-history to the beginning of the chat-spacing
        chatSpacing.prepend(newChatHistory);

        // Scroll to the bottom of the chat-spacing
        chatSpacing.scrollTop = chatSpacing.scrollHeight;

        // Clear the input field
        input.value = '';

        // Set focus back to the input field
        input.focus();
    }
}

/*********************************************************************** begin: models ************************************************/
//ea3.js
console.log(`Die TensorFlow.js-Version ist ${tf.version.tfjs}`);

let textData = loadTextData(); //textdaten: oliver twist (von unten)

//restoreSpecialCharacters
/*function restoreSpecialCharacters(text) {
    let replacements = {
        '<ae>': 'ä',
        '<oe>': 'ö',
        '<ue>': 'ü',
        '<ss>': 'ß'
    };

    for (let seq in replacements) {
        let regex = new RegExp(seq, 'g');
        text = text.replace(regex, replacements[seq]);
    }

    return text;
}
textData = restoreSpecialCharacters(textData);*/

//check for special characters
function checkSpecialCharactersInText(text) {
    let specialCharacters = ['<ae>', '<oe>', '<ue>', '<ss>'];
    let examples = { '<ae>': [], '<oe>': [], '<ue>': [], '<ss>': [] };

    let words = text.split(' ');

    for (let word of words) {
        for (let char of specialCharacters) {
            if (word.includes(char) && examples[char].length < 3) {
                examples[char].push(word);
            }
        }
    }

    for (let char of specialCharacters) {
        if (examples[char].length > 0) {
            console.log(`Es gibt Wörter mit "${char}": ${examples[char].join(', ')}`);
        } else {
            console.log(`Es gibt keine Wörter mit "${char}" im Text.`);
        }
    }
}
checkSpecialCharactersInText(textData);

// Textbereinigung
function cleanTextData(textData){
    textData = textData.trim().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g,""); //Sonderzeichen
    textData = textData.replace(/[\r\n]+/g, ' '); //Zeilenumbrüche
    console.log('Bereinigte Textdaten:', textData);
}
cleanTextData(textData);

// Tokenisierung
function tokenizer(text) {
    text = text.trim().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g,""); //Sonderzeichen entfernen
    text = text.replace(/[\r\n]+/g, ' ');     //Zeilenumbrüche wie \n oder \rersetzen durch  ''
    let texttokens = text.split(' ');
    texttokens = texttokens.filter(texttoken => texttoken !== ''); //leere Tokens entfernen
    return texttokens;
}
let tokens = tokenizer(textData);
console.log('Tokenisierung abgeschlossen.');
console.log('Tokenisierte Daten:', tokens);

//Datenmenge reduzieren
tokens = tokens.slice(0, 10000);
console.log('Reduzierte tokenisierte Daten:', tokens);

// Wörterbuch
let wordIndex = {};
tokens.forEach((token, i) => {
    if (!(token in wordIndex)) {
        wordIndex[token] = i + 1;
    }
});
console.log('Wörterbuch:', wordIndex);
console.log('Wörterbuch erstellt.');

// Überprüfen, ob Wörter mit Sonderzeichen im Wörterbuch vorhanden sind
/*function checkSpecialCharacters(dictionary) {
    let examples = { 'ä': [], 'ö': [], 'ü': [], 'ß': [] };
    for (let word in dictionary) {
        for (let char in examples) {
            if (word.includes(char) && examples[char].length < 3) {
                examples[char].push(word);
            }
        }
    }
    for (let char in examples) {
        console.log(`Wörter mit "${char}": ${examples[char].join(', ')}`);
    }
}
checkSpecialCharacters(wordIndex);*/

//vocabSize
let vocabSize = Object.keys(wordIndex).length;
console.log('unique words:', vocabSize);

// Sequenzbildung
let sequences = tokens.map(token => wordIndex[token]);
console.log('Sequenzen:', sequences);
let xs = [];
let ys = [];
for (let i = 0; i < sequences.length - 1; i++) {
    xs.push(sequences[i]);
    ys.push(sequences[i + 1]);
}
console.log('xs:', xs);
console.log('ys:', ys);
console.log('Sequenzbildung und One-Hot-Encoding abgeschlossen.');
console.log('Daten sind bereit für das Modell.');
console.log('Länge von xs:', xs.length);


/***************************************************************************** RNN ******************************************************************************/
// Umwandlung in One-Hot-Vektoren
let xsOneHot = tf.oneHot(tf.tensor1d(xs, 'int32'), vocabSize);
let ysOneHot = tf.oneHot(tf.tensor1d(ys, 'int32'), vocabSize);

// Hinzufügen einer Dimension für Zeitschritte - wie viele Wörter sich angeschaut werden in der Sequenz, um das nächste vorherzusagen
xsOneHot = xsOneHot.reshape([xsOneHot.shape[0], 1, xsOneHot.shape[1]]);

// Modell erstellen
let RNNmodel = tf.sequential();
RNNmodel.add(tf.layers.simpleRNN({units: 32, inputShape: [1, vocabSize]}));
RNNmodel.add(tf.layers.dense({units: vocabSize, activation: 'softmax'}));
console.log('Initiales RNN Modell deklariert und erstellt');

// Modell kompilieren
RNNmodel.compile({loss: 'categoricalCrossentropy', optimizer: 'adam'});
console.log('Initiales RNN Modell kompiliert');

// Modell trainieren
async function trainRNNModel() {
    const batchSize = 128;
    const epochs = 10;
    RNNmodel.fit(xsOneHot, ysOneHot, {
        batchSize, 
        epochs,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
            }
        }
    }).then(() => {
        console.log('RNN Modelltraining abgeschlossen.');
    });
}

// RNN Vorhersagefunktion
function predictRNN(seedText, timestep) {
    let seedTokens = tokenizer(seedText);
    console.log('text zur RNN prediction erkannt:', seedTokens);
    let seedSequences = seedTokens.map(token => wordIndex[token]);
    let seedOneHot = tf.oneHot(tf.tensor1d(seedSequences, 'int32'), vocabSize);
    seedOneHot = seedOneHot.reshape([seedOneHot.shape[0], timestep, seedOneHot.shape[1]]);
    let prediction = RNNmodel.predict(seedOneHot);
    let topn = 3;
    let topk = tf.topk(prediction, topn);
    let predictedIndices = Array.from(topk.indices.dataSync());
    let predictedProbabilities = Array.from(topk.values.dataSync());
    return predictedIndices.map((index, i) => ({
        word: Object.keys(wordIndex).find(key => wordIndex[key] === index),
        confidence: predictedProbabilities[i]
    }));
}

/***************************************************************************** FFNN *****************************************************************************/


// Umwandlung in One-Hot-Vektoren
xsOneHotF = tf.oneHot(tf.tensor1d(xs, 'int32'), vocabSize);
ysOneHotF = tf.oneHot(tf.tensor1d(ys, 'int32'), vocabSize);

// Modell erstellen
let FFNNmodel = tf.sequential();
FFNNmodel.add(tf.layers.dense({units: 32, inputShape: [vocabSize], activation: 'relu'}));
FFNNmodel.add(tf.layers.dense({units: vocabSize, activation: 'softmax'}));
console.log('Initiales FFNN Modell deklariert und erstellt');

// Modell kompilieren
FFNNmodel.compile({loss: 'categoricalCrossentropy', optimizer: 'adam'});
console.log('Initiales FFNN Modell kompiliert');

// Modell trainieren
async function trainFFNNModel() {
    const batchSizeFFNN = 128;
    const epochsFFNN = 10;
    FFNNmodel.fit(xsOneHotF, ysOneHotF, {
        batchSize: batchSizeFFNN, 
        epochs: epochsFFNN,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
            }
        }
    }).then(() => {
        console.log('FFNN Modelltraining abgeschlossen.');
    });
}

// FFNN Vorhersagefunktion
function predictFFNN(seedText) {
    let seedTokens = tokenizer(seedText);
    console.log('text zur FFNN prediction erkannt:', seedTokens);
    let seedSequences = seedTokens.map(token => wordIndex[token]).filter(index => index !== undefined);
    let seedOneHot = tf.oneHot(tf.tensor1d(seedSequences, 'int32'), vocabSize);
    let prediction = FFNNmodel.predict(seedOneHot);
    // Ermittlung der Indizes der n wahrscheinlichsten Vorhersagen
    let topn = 3;
    let topk = tf.topk(prediction, topn);
    let predictedIndices = Array.from(topk.indices.dataSync());
    let predictedProbabilities = Array.from(topk.values.dataSync());
    return predictedIndices.map((index, i) => ({
        word: Object.keys(wordIndex).find(key => wordIndex[key] === index),
        confidence: predictedProbabilities[i]
    }));
}


/*********************************************************************** dev functions ************************************************/
//RNN Modell trainieren
document.getElementById('train-RNNmodel-btn').addEventListener('click', trainRNNModel);
document.getElementById('train-FFNNmodel-btn').addEventListener('click', trainFFNNModel);

/************************************************************************ save model **************************************************/
async function saveModel(model) {
  // Save the model in the usual way
  const saveResult = await model.save('downloads://bestmodel');
  console.log('Model saved successfully:', saveResult);

  // Get the weights of the model
  const weights = model.getWeights();

  // Convert the weights to their underlying data synchronously
  const weightsData = weights.map(tensor => tensor.dataSync());

  // Convert the weights data to a Uint8Array
  const weightsUint8Array = weightsData.map(data => new Uint8Array(data.buffer));

  // Create a Blob from the Uint8Array
  const blob = new Blob([weightsUint8Array], {type: 'application/octet-stream'});

  // Create a URL for the Blob
  const url = URL.createObjectURL(blob);

  // Create a downloadable link for the file
  const link = document.createElement('a');
  link.href = url;
  link.download = 'weights.bin';

  // Append the link to the body
  document.body.appendChild(link);

  // Programmatically click the link to start the download
  link.click();

  // Remove the link from the body
  document.body.removeChild(link);
}


// Speichern des Wörterbuchs
function saveDictionary(dictionary) {
     // Umwandlung des Wörterbuchs in einen JSON-String
    const dictionaryString = JSON.stringify(dictionary);

    // Umwandlung des JSON-Strings in einen Uint8Array
    let encoder = new TextEncoder();
    let uint8Array = encoder.encode(dictionaryString);

    // Erstellung eines Blobs aus dem Uint8Array
    const blob = new Blob([uint8Array], {type: 'application/json'});


    // Erstellung einer URL für den Blob
    const url = URL.createObjectURL(blob);

    // Erstellung eines herunterladbaren Links für die Datei
    const link = document.createElement('a');
    link.href = url;
    link.download = 'dictionary.json';

    // Anhängen des Links an den Body
    document.body.appendChild(link);

    // Programmatisches Klicken auf den Link, um den Download zu starten
    link.click();

    // Entfernen des Links aus dem Body
    document.body.removeChild(link);
}

// Button-Event hinzufügen
document.getElementById('save-data-btn').addEventListener('click', function() {
    //saveModel(RNNmodel);
    saveModel(FFNNmodel);
    saveDictionary(wordIndex);
});

/************************************************************************ load model **************************************************/
// Funktion zum Aktualisieren des Wörterbuchs
function updateDictionary() {
    wordIndex = {...dictionarySave};
    console.log('Wörterbuch aktualisiert:', wordIndex);
}

// Funktion zum Laden des Modells
async function loadModel(modelName) {
    return new Promise(async (resolve, reject) => {
      try {
        // Holen Sie das ausgewählte Modell aus modelsEA3
        const selectedModel = modelsEA3[modelName];
        if (selectedModel) {
          // Erstellen Sie einen Blob aus dem Uint8Array
          const blob = new Blob([selectedModel.bin], {type: 'application/octet-stream'});

          // Erstellen Sie ein neues File-Objekt aus dem Blob
          const binFile = new File([blob], 'model.bin');

          // Ändern Sie das weightsManifest in den JSON-Daten des Modells
          selectedModel.json.weightsManifest[0].paths = ['model.bin'];

          // Erstellen Sie einen Blob aus dem JSON
          const jsonBlob = new Blob([JSON.stringify(selectedModel.json)], {type: 'application/json'});

          // Erstellen Sie ein neues File-Objekt aus dem Blob
          const jsonFile = new File([jsonBlob], 'model.json');

          // Laden Sie das Modell
          if (modelName.startsWith('RNN')) {
            RNNmodel = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, binFile]));
            console.log('RNN-Modell erfolgreich geladen');
            resolve(RNNmodel);
            //return RNNmodel;
          } else if (modelName.startsWith('FFNN')) {
            FFNNmodel = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, binFile]));
            console.log('FFNN-Modell erfolgreich geladen');
            resolve(FFNNmodel);
            //return FFNNmodel;
          }
          //const model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, binFile]));
          //console.log('Modell erfolgreich geladen');
          //return model;
        } else {
          console.error('Modell nicht gefunden:', modelName);
        }
      } catch (error) {
        console.error('Fehler beim Laden des Modells:', error);
        reject(error);
      }
    });
}

// load model Button mit Funktion verknüpfen
document.getElementById('load-model-btn').addEventListener('click', async function() {
    updateDictionary();
    try {
        /*await loadModel('RNNmodel1');
        RNNmodel.compile({loss: 'categoricalCrossentropy', optimizer: 'adam'});
        console.log('RNN Modell kompiliert');*/
        await loadModel('FFNNmodel1');
        FFNNmodel.compile({loss: 'categoricalCrossentropy', optimizer: 'adam'});
        console.log('FFNN Modell kompiliert');
    } catch (error) {
        console.error('Fehler beim Laden oder Kompilieren des Modells:', error);
    }
});

//initial beide Modelle Laden
async function loadingBestModels() {
    await loadModel('RNNmodel1');
    await loadModel('FFNNmodel1');
}

document.addEventListener('DOMContentLoaded', loadingBestModels);

/*********************************************************************** display predictions ************************************************/
function translateSpecialCharacters(word) {
    let specialCharMapping = {
        '<ae>': 'ä',
        '<ue>': 'ü',
        '<oe>': 'ö',
        '<ss>': 'ß',
        '<AE>': 'Ä',
        '<UE>': 'Ü',
        '<OE>': 'Ö',
        '<SS>': 'ẞ'
    };

    return word.replace(/<ae>|<ue>|<oe>|<ss>|<AE>|<UE>|<OE>|<SS>/g, function(match) {
        return specialCharMapping[match];
    });
}

function preprocessInput(input) {
    // Create a mapping for the special characters
    let specialCharMapping = {
        'ä': '<ae>',
        'ü': '<ue>',
        'ö': '<oe>',
        'ß': '<ss>',
        'Ä': '<AE>',
        'Ü': '<UE>',
        'Ö': '<OE>',
        'ẞ': '<SS>'
    };

    // Replace the special characters
    let preprocessedInput = input.replace(/[äüößÄÜÖẞ]/g, function(match) {
        return specialCharMapping[match];
    });

    // Tokenize the preprocessed input
    let tokens = tokenizer(preprocessedInput);

    // Get the last word
    let lastWord = tokens.pop();

    // Log the preprocessed word if it's different from the original
    if (preprocessedInput !== input) {
        console.log('The input "' + input + '" contains special characters and was translated to "' + preprocessedInput + '".');
    }

    return lastWord;
}

//plot predictions
function plotPredictions(predictions, elementId) {
    // Überprüfen Sie, ob das Element existiert
    let element = document.getElementById(elementId);
    if (!element) {
        console.error('Element with id ' + elementId + ' does not exist');
        return;
    }
    
    // Prepare the labels and confidences
    let labels = predictions.map(p => translateSpecialCharacters(p.word) + '  ').reverse();
    let confidences = predictions.map(p => p.confidence).reverse();

    // Prepare the data for the bar chart
    let data = [{
        y: labels,
        x: confidences,
        type: 'bar',
        orientation: 'h',
        text: confidences.map(c => c.toFixed(2)),
        textposition: 'auto',
        textfont: { 
            size: 20
        },
        marker: {
            color: ['rgb(237,240,243)', 'rgb(237,240,243)', 'rgb(37,80,83)'],
            opacity: 0.9,
            line: {
                color: ['rgb(8,48,107)', 'rgb(8,48,107)', 'rgb(255,255,255)'],
                width: 1.5
            }
        },
        hoverinfo: 'none'
    }];

    // Create the bar chart
    Plotly.newPlot(elementId, data, {
        xaxis: {
            automargin: true
        },
        yaxis: {
            automargin: true,
            tickfont: {
                size: 20
            },
            tickpadding: 15
        }
    }, {
        displayModeBar: false
    });
}

//prediction in der Konsole und Grafiken in den predicitons-Regionen
document.getElementById('chat-input').addEventListener('keyup', function(e) {
    if (e.key === ' ') {
        let preprocessedInput = preprocessInput(e.target.value);
        let wordExistsInDictionary = preprocessedInput in wordIndex;
        if (!wordExistsInDictionary) {
            console.log('The word "' + preprocessedInput + '" is not in the dictionary.');
        }
        let predictionsRNN = predictRNN(preprocessedInput, 1);
        console.log('RNN predicts: ' + predictionsRNN.map(p => p.word + ' (' + p.confidence.toFixed(2) + ')').join(', '));
        plotPredictions(predictionsRNN, 'RNN-predictions');
        if (wordExistsInDictionary) {
            let predictionsFFNN = predictFFNN(preprocessedInput);
            console.log('FFNN predicts: ' + predictionsFFNN.map(p => p.word + ' (' + p.confidence.toFixed(2) + ')').join(', '));
            plotPredictions(predictionsFFNN, 'FFNN-predictions');
        } else {
            console.log('The word "' + preprocessedInput + '" is not in the dictionary.');
            console.log('FFNN cannot predict the next word.');
        }
    }
});