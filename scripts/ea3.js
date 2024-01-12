//ea3.js

/**********************event listeners************************************************/
//submit chat button
document.getElementById('send-button').addEventListener('click', sendMessage);

//submit chat via Enter (not Shift+Enter)
document.getElementById('chat-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

//prediction in der Konsole
document.getElementById('chat-input').addEventListener('keyup', function(e) {
    if (e.key === ' ') {
        let predictions = predictRNN(e.target.value);
        //let predictions = predictFFNN(e.target.value);
        console.log(predictions);
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
console.log('RNN Modell erstellt');

// Modell kompilieren
RNNmodel.compile({loss: 'categoricalCrossentropy', optimizer: 'adam'});
console.log('RNN Modell kompiliert');

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
function predictRNN(seedText) {
    // Tokenisieren des eingegebenen Textes
    let seedTokens = tokenizer(seedText);
    console.log('text zur prediction:', seedTokens);

    // Umwandlung der Tokens in numerische Werte mit dem Wörterbuch
    let seedSequences = seedTokens.map(token => wordIndex[token]);

    // Umwandlung der Sequenzen in One-Hot-Vektoren und Hinzufügen einer Dimension für die Zeitschritte
    let seedOneHot = tf.oneHot(tf.tensor1d(seedSequences, 'int32'), vocabSize);
    seedOneHot = seedOneHot.reshape([seedOneHot.shape[0], 1, seedOneHot.shape[1]]);

    // Vorhersage mit dem RNN-Modell
    let prediction = RNNmodel.predict(seedOneHot);

    // Ermittlung der Indizes der n wahrscheinlichsten Vorhersagen
    let topn = 1;
    let topIndices = tf.topk(prediction, topn).indices;
    let predictedIndices = Array.from(topIndices.dataSync());

    // Rückübersetzung der vorhergesagten Indizes in Wörter
    return predictedIndices.map(index => Object.keys(wordIndex).find(key => wordIndex[key] === index));
}

/***************************************************************************** FFNN *****************************************************************************/

/*
// Umwandlung in One-Hot-Vektoren
xsOneHot = tf.oneHot(tf.tensor1d(xs, 'int32'), vocabSize);
ysOneHot = tf.oneHot(tf.tensor1d(ys, 'int32'), vocabSize);

// Modell erstellen
let FFNNmodel = tf.sequential();
FFNNmodel.add(tf.layers.dense({units: 10, inputShape: [vocabSize], activation: 'relu'}));
FFNNmodel.add(tf.layers.dense({units: vocabSize, activation: 'softmax'}));
console.log('FFNN Modell erstellt');

// Modell kompilieren
FFNNmodel.compile({loss: 'categoricalCrossentropy', optimizer: 'adam'});
console.log('FFNN Modell kompiliert');

// Modell trainieren
const batchSizeFFNN = 128;
const epochsFFNN = 10;
FFNNmodel.fit(xsOneHot, ysOneHot, {batchSize: batchSizeFFNN, epochs: epochsFFNN}).then(() => {
    console.log('FFNN Modelltraining abgeschlossen.');
});

// FFNN Vorhersagefunktion
function predictFFNN(seedText) {
    let seedTokens = tokenizer(seedText);
    let seedSequences = seedTokens.map(token => wordIndex[token] ? wordIndex[token] : Unbekannt); // 0 als Standardwert
    let seedOneHot = tf.oneHot(tf.tensor1d(seedSequences, 'int32'), vocabSize);
    let prediction = FFNNmodel.predict(seedOneHot);
    let predictedIndex = tf.argMax(prediction, 1).dataSync()[0];
    return Object.keys(wordIndex).find(key => wordIndex[key] === predictedIndex);
}
*/

/*********************************************************************** dev functions ************************************************/
//RNN Modell trainieren
document.getElementById('train-model-btn').addEventListener('click', trainRNNModel);

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
    saveModel(RNNmodel);
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

// Button mit Funktion verknüpfen
document.getElementById('load-model-btn').addEventListener('click', async function() {
    updateDictionary();
    try {
        await loadModel('RNNmodel1');
        RNNmodel.compile({loss: 'categoricalCrossentropy', optimizer: 'adam'});
        console.log('RNN Modell kompiliert');
    } catch (error) {
        console.error('Fehler beim Laden oder Kompilieren des Modells:', error);
    }
});