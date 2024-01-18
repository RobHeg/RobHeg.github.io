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

/********************************************************************************************* words ***************************/

// Tokenisierung
function tokenizer(text) {
    // Überprüfen, ob der Text undefined ist
    if (typeof text === 'undefined') {
        console.log('Tokenizer: Kein Text zur Verarbeitung vorhanden.');
        return [];
    }

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

//initially create the newdictionary as a copy of the dictionary, ordered by the frequency of how often words appear in the original Text as orderedWords_RNN list
let newDictionary = {};
let initialOrderedWordList = [];
let orderedWords_RNN = [];
let orderedWords_FFNN = [];
// Erstellen Sie eine Frequenzkarte
let frequencyMap = {};
let totalWords = tokens.length;
tokens.forEach(token => {
    if (!(token in frequencyMap)) {
        frequencyMap[token] = 1;
    } else {
        frequencyMap[token]++;
    }
});

// Sortieren Sie die Frequenzkarte
initialOrderedWordList = Object.keys(frequencyMap).sort((a, b) => frequencyMap[b] - frequencyMap[a]);

// Erstellen Sie das neue Wörterbuch und die Konfidenzliste
newDictionary = {};
let initialWordConfidence = [];
initialOrderedWordList.forEach((token, i) => {
    newDictionary[token] = wordIndex[token];
    initialWordConfidence[i] = frequencyMap[token] / totalWords;
});

console.log('initial frequency List: ', initialOrderedWordList);
console.log('initial confidences: ', initialWordConfidence);

/***************************************************************************** chars ***************************************************************************/

/*// Buchstabentokenisierung
function charTokenizer(text) {
    text = translateSpecialCharacters(text);  //Spezielle Zeichen übersetzen
    text = text.toLowerCase();  // Umwandlung in Kleinbuchstaben
    let chartokens = text.split('');
    chartokens = chartokens.filter(chartoken => {
        // Behalte nur alphanumerische Zeichen, Leerzeichen und spezielle deutsche Zeichen
        return /[a-z0-9äöüß]/.test(chartoken);
    });
    return chartokens;
}
let charTokens = charTokenizer(textData);
console.log('Buchstabentokenisierung abgeschlossen.');
console.log('Tokenisierte Buchstabendaten:', charTokens);

// Wörterbuch für Buchstaben
let charIndex = {};
charTokens.forEach((token, i) => {
    if (!(token in charIndex)) {
        charIndex[token] = i + 1;
    }
});
console.log('Wörterbuch für Buchstaben:', charIndex);
console.log('Wörterbuch für Buchstaben erstellt.');
let charVocabSize = Object.keys(charIndex).length;
console.log('unique characters:', charVocabSize);

// Sequenzbildung für Buchstaben
let charSequences = charTokens.map(token => charIndex[token]);
console.log('Sequenzen für Buchstaben:', charSequences);
let charXs = [];
let charYs = [];
for (let i = 0; i < charSequences.length - 1; i++) {
    charXs.push(charSequences[i]);
    charYs.push(charSequences[i + 1]);
}
console.log('xs für Buchstaben:', charXs);
console.log('ys für Buchstaben:', charYs);
console.log('Sequenzbildung und One-Hot-Encoding für Buchstaben abgeschlossen.');
console.log('Daten für Buchstaben sind bereit für das Modell.');
console.log('Länge von xs für Buchstaben:', charXs.length);*/

/***************************************************************************** RNN words ******************************************************************************/
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

/***************************************************************************** FFNN words *****************************************************************************/


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

/*********************************************************************** RNN chars ***************************************************/
/*// Umwandlung in One-Hot-Vektoren für Buchstaben
let charXsOneHot = tf.oneHot(tf.tensor1d(charXs, 'int32'), charVocabSize);
let charYsOneHot = tf.oneHot(tf.tensor1d(charYs, 'int32'), charVocabSize);

// Hinzufügen einer Dimension für Zeitschritte - wie viele Buchstaben sich angeschaut werden in der Sequenz, um das nächste vorherzusagen
charXsOneHot = charXsOneHot.reshape([charXsOneHot.shape[0], 1, charXsOneHot.shape[1]]);

// Modell erstellen für Buchstaben
let charRNNmodel = tf.sequential();
charRNNmodel.add(tf.layers.simpleRNN({units: 5, inputShape: [1, charVocabSize]}));
charRNNmodel.add(tf.layers.dense({units: charVocabSize, activation: 'softmax'}));
console.log('Initiales RNN Modell für Buchstaben deklariert und erstellt');

// Modell kompilieren für Buchstaben
charRNNmodel.compile({loss: 'categoricalCrossentropy', optimizer: 'adam'});
console.log('Initiales RNN Modell für Buchstaben kompiliert');

// Modell trainieren für Buchstaben
async function trainCharRNNModel() {
    const batchSize = 128;
    const epochs = 10;
    charRNNmodel.fit(charXsOneHot, charYsOneHot, {
        batchSize, 
        epochs,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
            }
        }
    }).then(() => {
        console.log('RNN Modelltraining für Buchstaben abgeschlossen.');
    });
}

// RNN Vorhersagefunktion für Buchstaben
function predictCharRNN(seedText, timestep) {
    let seedCharTokens = charTokenizer(seedText);
    console.log('text zur RNN prediction für Buchstaben erkannt:', seedCharTokens);
    let seedCharSequences = seedCharTokens.map(token => charIndex[token]);
    let seedCharOneHot = tf.oneHot(tf.tensor1d(seedCharSequences, 'int32'), charVocabSize);
    seedCharOneHot = seedCharOneHot.reshape([seedCharOneHot.shape[0], timestep, seedCharOneHot.shape[1]]);
    let prediction = charRNNmodel.predict(seedCharOneHot);
    let topn = 3;
    let topk = tf.topk(prediction, topn);
    let predictedIndices = Array.from(topk.indices.dataSync());
    let predictedProbabilities = Array.from(topk.values.dataSync());
    return predictedIndices.map((index, i) => ({
        char: Object.keys(charIndex).find(key => charIndex[key] === index),
        confidence: predictedProbabilities[i]
    }));
}
*/
/*********************************************************************** FFNN characters **********************************************/
/*
// Umwandlung in One-Hot-Vektoren für Buchstaben
let charXsOneHotF = tf.oneHot(tf.tensor1d(charXs, 'int32'), charVocabSize);
let charYsOneHotF = tf.oneHot(tf.tensor1d(charYs, 'int32'), charVocabSize);

// Modell erstellen für Buchstaben
let charFFNNmodel = tf.sequential();
charFFNNmodel.add(tf.layers.dense({units: 5, inputShape: [charVocabSize], activation: 'relu'}));
charFFNNmodel.add(tf.layers.dense({units: charVocabSize, activation: 'softmax'}));
console.log('Initiales FFNN Modell für Buchstaben deklariert und erstellt');

// Modell kompilieren für Buchstaben
charFFNNmodel.compile({loss: 'categoricalCrossentropy', optimizer: 'adam'});
console.log('Initiales FFNN Modell für Buchstaben kompiliert');

// Modell trainieren für Buchstaben
async function trainCharFFNNModel() {
    const batchSizeFFNN = 128;
    const epochsFFNN = 10;
    charFFNNmodel.fit(charXsOneHotF, charYsOneHotF, {
        batchSize: batchSizeFFNN, 
        epochs: epochsFFNN,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
            }
        }
    }).then(() => {
        console.log('FFNN Modelltraining für Buchstaben abgeschlossen.');
    });
}

// FFNN Vorhersagefunktion für Buchstaben
function predictCharFFNN(seedText) {
    let seedCharTokens = charTokenizer(seedText);
    console.log('text zur FFNN prediction für Buchstaben erkannt:', seedCharTokens);
    let seedCharSequences = seedCharTokens.map(token => charIndex[token]).filter(index => index !== undefined);
    let seedCharOneHot = tf.oneHot(tf.tensor1d(seedCharSequences, 'int32'), charVocabSize);
    let prediction = charFFNNmodel.predict(seedCharOneHot);
    // Ermittlung der Indizes der n wahrscheinlichsten Vorhersagen
    let topn = 3;
    let topk = tf.topk(prediction, topn);
    let predictedIndices = Array.from(topk.indices.dataSync());
    let predictedProbabilities = Array.from(topk.values.dataSync());
    return predictedIndices.map((index, i) => ({
        char: Object.keys(charIndex).find(key => charIndex[key] === index),
        confidence: predictedProbabilities[i]
    }));
}
*/

/*********************************************************************** dev functions ************************************************/
//RNN Modell trainieren
document.getElementById('train-RNNmodel-btn').addEventListener('click', function() {
    //let modelType = document.getElementById('model-type').checked;
    //if (modelType) {
        trainRNNModel(); // Trainiert das Wortmodell
    /*} else {
        trainCharRNNModel(); // Trainiert das Buchstabenmodell
    }*/
});

document.getElementById('train-FFNNmodel-btn').addEventListener('click', function() {
    let modelType = document.getElementById('model-type').checked;
    //if (modelType) {
        trainFFNNModel(); // Trainiert das Wortmodell
    /*} else {
        trainCharFFNNModel(); // Trainiert das Buchstabenmodell
    }*/
});

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
    //let modelType = document.getElementById('model-type').checked;
    //if (modelType) {
        // Speichern des RNN- oder FFNN-Wortmodells und des Wörterbuchs
        saveModel(RNNmodel, 'RNNmodel');
        saveModel(FFNNmodel, 'FFNNmodel');
        saveDictionary(wordIndex, 'wordIndex');
    //} else {
        // Speichern des RNN- oder FFNN-Buchstabenmodells und des Buchstabenwörterbuchs
        //saveModel(charRNNmodel, 'charRNNmodel');
        //saveModel(charFFNNmodel, 'charFFNNmodel');
    //    saveDictionary(charIndex, 'charIndex');
    //}
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

function preprocessChar(input) {
    // Holen Sie sich den letzten Buchstaben
    let lastChar = input.slice(-1);
    return lastChar;
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
    let labels = predictions.map(p => p.word ? translateSpecialCharacters(p.word) + '  ' : '').reverse();
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


let orderedConfidences_RNN = [];

function createNewDictionary_RNN(predictions) {
    // Erstellen Sie ein Array basierend auf den Vorhersagen
    orderedWords_RNN = predictions.map(p => p.word);
    orderedConfidences_RNN = predictions.map(p => p.confidence);

    // Fügen Sie alle Wörter aus dem alten Wörterbuch hinzu, die nicht in den Vorhersagen waren
    for (let word in wordIndex) {
        if (!orderedWords_RNN.includes(word)) {
            orderedWords_RNN.push(word);
            orderedConfidences_RNN.push(0); // Setzen Sie die Wahrscheinlichkeit auf 0 für Wörter, die nicht vorhergesagt wurden
        }
    }

    // Erstellen Sie ein neues Wörterbuch aus dem geordneten Array
    for (let i = 0; i < orderedWords_RNN.length; i++) {
        newDictionary[orderedWords_RNN[i]] = wordIndex[orderedWords_RNN[i]];
    }
}

let orderedConfidences_FFNN = [];

function createNewDictionary_FFNN(predictions) {
    // Erstellen Sie ein Array basierend auf den Vorhersagen
    orderedWords_FFNN = predictions.map(p => p.word);
    orderedConfidences_FFNN = predictions.map(p => p.confidence);

    // Fügen Sie alle Wörter aus dem alten Wörterbuch hinzu, die nicht in den Vorhersagen waren
    for (let word in wordIndex) {
        if (!orderedWords_FFNN.includes(word)) {
            orderedWords_FFNN.push(word);
            orderedConfidences_FFNN.push(0); // Setzen Sie die Wahrscheinlichkeit auf 0 für Wörter, die nicht vorhergesagt wurden
        }
    }

    // Erstellen Sie ein neues Wörterbuch aus dem geordneten Array
    for (let i = 0; i < orderedWords_FFNN.length; i++) {
        newDictionary[orderedWords_FFNN[i]] = wordIndex[orderedWords_FFNN[i]];
    }
}

/*function completeWord(partialWord) {
    // Durchlaufen Sie das neue Wörterbuch
    for (let word in newDictionary) {
        // Überprüfen Sie, ob das Wort mit dem teilweise eingegebenen Wort beginnt
        if (word.startsWith(partialWord)) {
            return word;
        }
    }

    // Wenn kein passendes Wort gefunden wurde, geben Sie null zurück
    return null;
}

function completeFirstWord(partialWord) {
    // Durchlaufen Sie alle Wörter im Wörterbuch
    for (let word in wordIndex) {
        // Überprüfen Sie, ob das Wort mit dem teilweise eingegebenen Wort beginnt
        if (word.startsWith(partialWord)) {
            return word;
        }
    }

    // Wenn kein passendes Wort gefunden wurde, geben Sie null zurück
    return null;
}*/

function completeWord(partialWord, orderedWords, orderedConfidences, numWords) {
    // Erstellen Sie ein Array, um die passenden Wörter zu speichern
    let matchingWords = [];

    // Durchlaufen Sie die geordnete Wortliste
    for (let i = 0; i < orderedWords.length; i++) {
        // Überprüfen Sie, ob das Wort mit dem teilweise eingegebenen Wort beginnt
        if (orderedWords[i].startsWith(partialWord)) {
            matchingWords.push({word: orderedWords[i], confidence: orderedConfidences[i]});

            // Überprüfen Sie, ob die gewünschte Anzahl von Wörtern erreicht wurde
            if (matchingWords.length === numWords) {
                break;
            }
        }
    }

    // Füllen Sie das Array mit 'null', wenn weniger Wörter gefunden wurden
    while (matchingWords.length < numWords) {
        matchingWords.push({word: null, confidence: 0});
    }

    return matchingWords;
}

function findWordPositionInList(word, list) {
    // Überprüfen Sie, ob das Wort im Wörterbuch existiert
    if (word in newDictionary) {
        // Finden Sie die Position des Wortes in orderedWords_RNN
        let position = list.indexOf(word);

        // Da die Indizes in JavaScript bei 0 beginnen, addieren Sie 1 zur Position
        return position + 1;
    }

    // Wenn das Wort nicht im Wörterbuch gefunden wurde, geben Sie null zurück
    return null;
}

function countWordsInInput(input) {
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

    // Ersetzen Sie die Sonderzeichen
    let preprocessedInput = input.replace(/[äüößÄÜÖẞ]/g, function(match) {
        return specialCharMapping[match];
    });

    // Tokenisieren Sie die vorverarbeitete Eingabe
    let tokens = tokenizer(preprocessedInput);

    // Geben Sie die Anzahl der Tokens zurück
    return tokens.length;
}

function writeStatementToElement(word, list, elementId) {
    // Überprüfen Sie, ob das Element existiert
    let element = document.getElementById(elementId);
    if (!element) {
        console.error('Element mit der ID ' + elementId + ' existiert nicht');
        return;
    }

    // Finden Sie die Position des Wortes in der Liste
    let position = findWordPositionInList(word, list);

    // Bereiten Sie die Aussage vor
    let statement;
    if (position !== null) {
        if (list === initialOrderedWordList) {
            statement = 'Das Wort "' + translateSpecialCharacters(word) + '" war auf Platz Nummer ' + position + ' in der nach Häufigkeit geordneten Liste aller Wörter des Ausgangstextes.';
        } else if (list === orderedWords_RNN) {
            statement = 'Das Wort "' + translateSpecialCharacters(word) + '" war guess Nummer ' + position + ' von RNN.';
        } else if (list === orderedWords_FFNN) {
            statement = 'Das Wort "' + translateSpecialCharacters(word) + '" war guess Nummer ' + position + ' von FFNN.';
        }
    } else {
        statement = 'Das Wort "' + translateSpecialCharacters(word) + '" ist nicht im Wörterbuch vorhanden.';
    }

    // Löschen Sie den aktuellen Inhalt des Elements
    element.innerHTML = '';

    // Schreiben Sie die Aussage in das Element
    element.innerHTML = statement;
}

//prediction in der Konsole und Grafiken in den predicitons-Regionen
document.getElementById('chat-input').addEventListener('keyup', function(e) {
    let preprocessedInput = preprocessInput(e.target.value);
    //let preprocessedChar = preprocessChar(e.target.value);
    let wordExistsInDictionary = preprocessedInput in wordIndex;
    //let charExistsInDictionary = preprocessedChar in charIndex;
    if (e.key === ' ' && typeof preprocessedInput !== 'undefined' && preprocessedInput.length > 0) {
        let wordCount = countWordsInInput(e.target.value);
        let listToUse_RNN = wordCount === 1 ? initialOrderedWordList : orderedWords_RNN;
        let listToUse_FFNN = wordCount === 1 ? initialOrderedWordList : orderedWords_FFNN;
        let position = findWordPositionInList(preprocessedInput, listToUse_RNN);
        if (position !== null) {
            if (listToUse_RNN === initialOrderedWordList) {
                console.log('Das Wort "' + translateSpecialCharacters(preprocessedInput) + '" war auf Platz Nummer ' + position + ' in der nach Häufigkeit geordneten Liste aller Wörter des Ausgangstextes.');
            } else {
                console.log('Das Wort "' + translateSpecialCharacters(preprocessedInput) + '" war guess Nummer ' + position + ' von RNN.');
            }
        } else {
            console.log('Das Wort "' + translateSpecialCharacters(preprocessedInput) + '" ist nicht im Wörterbuch vorhanden.');
        }
        let predictionsRNN = predictRNN(preprocessedInput, 1);
        console.log('RNN predicts: ' + predictionsRNN.map(p => p.word + ' (' + p.confidence.toFixed(2) + ')').join(', '));
        plotPredictions(predictionsRNN, 'RNN-predictions');
        writeStatementToElement(preprocessedInput, listToUse_RNN, 'RNN-placement-evaluation');
        createNewDictionary_RNN(predictionsRNN);
        console.log('RNN guess list: ', orderedWords_RNN);
        if (wordExistsInDictionary) {
            let predictionsFFNN = predictFFNN(preprocessedInput);
            console.log('FFNN predicts: ' + predictionsFFNN.map(p => p.word + ' (' + p.confidence.toFixed(2) + ')').join(', '));
            plotPredictions(predictionsFFNN, 'FFNN-predictions');
            writeStatementToElement(preprocessedInput, listToUse_FFNN, 'FFNN-placement-evaluation');
            createNewDictionary_FFNN(predictionsFFNN);
            console.log('FFNN guess list: ', orderedWords_FFNN);
        } else {
            console.log('The word "' + preprocessedInput + '" is not in the dictionary.');
            console.log('FFNN cannot predict the next word.');
            Plotly.purge('FFNN-predictions');
            writeStatementToElement(preprocessedInput, listToUse_FFNN, 'FFNN-placement-evaluation');
        }
    } else {
        // Überprüfen, ob mindestens ein Buchstabe eingegeben wurde
        if (typeof preprocessedInput !== 'undefined'){
            if (preprocessedInput.length > 0) {
            
                //Wörter vervollständigen - RNN
                let completion_RNN = completeWord(preprocessedInput, orderedWords_RNN, orderedConfidences_RNN, 3);
                if (completion_RNN[0].word) {
                    console.log('RNN suggests: ' + completion_RNN[0].word);
                    plotPredictions(completion_RNN, 'RNN-predictions');
                } else {
                    let firstWordCompletion = completeWord(preprocessedInput, initialOrderedWordList, initialWordConfidence, 3);
                    if (firstWordCompletion) {
                        console.log('First word completion RNN: ' + firstWordCompletion[0].word);
                        plotPredictions(firstWordCompletion, 'RNN-predictions');
                    }else{
                        Plotly.purge('RNN-predictions');
                    }
                }
               
                //Wörter vervollständigen - FFNN
                let completion_FFNN = completeWord(preprocessedInput, orderedWords_FFNN, orderedConfidences_FFNN, 3);
                if (completion_FFNN[0].word) {
                    console.log('FFNN suggests: ' + completion_FFNN[0].word);
                    plotPredictions(completion_FFNN, 'FFNN-predictions');
                } else {
                    let firstWordCompletion = completeWord(preprocessedInput, initialOrderedWordList, initialWordConfidence, 3);
                    if (firstWordCompletion) {
                        console.log('First word completion FFNN: ' + firstWordCompletion[0].word);
                        plotPredictions(firstWordCompletion, 'FFNN-predictions');
                    }else{
                        Plotly.purge('FFNN-predictions');
                    }
                }

                // Nur den letzten Buchstaben zur Vorhersage verwenden
                /*let lastChar = preprocessedInput.slice(-1);
                if (!charExistsInDictionary) {
                    console.log('The character "' + lastChar + '" is not in the dictionary.');
                }
                let predictionsCharRNN = predictCharRNN(lastChar, 1);
                console.log('Char RNN predicts: ' + predictionsCharRNN.map(p => p.char + ' (' + p.confidence.toFixed(2) + ')').join(', '));
                plotPredictions(predictionsCharRNN, 'RNN-char-predictions');
                if (charExistsInDictionary) {
                    let predictionsCharFFNN = predictCharFFNN(lastChar);
                    console.log('Char FFNN predicts: ' + predictionsCharFFNN.map(p => p.char + ' (' + p.confidence.toFixed(2) + ')').join(', '));
                    plotPredictions(predictionsCharFFNN, 'FFNN-char-predictions');
                } else {
                    console.log('The character "' + lastChar + '" is not in the dictionary.');
                    console.log('Char FFNN cannot predict the next character.');
                }*/
            }
        }
    }
});