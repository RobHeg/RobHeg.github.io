// JavaScript source code
console.log('Hello TensorFlow');


/************************************************************** run ************************************************************************************************************************/
let data, model, tensorData;
let testdata;
let modelJson = null;
let modelWeights = null;
let globalOptimizer = 'adam';
let globalLearningRate = 0.01;
let globalBatchSize = 32; 
let globalEpochs = 50;
let globalTrainingMSE;
let globalTrainingLoss;
let globalTestMSE;
let globalTestLoss;

//whole program logic here in a nutshell
async function run() {
    const funcString = 'return (x + 0.8) * (x - 0.2) * (x - 0.3) * (x - 0.6);';
    const func = new Function('x', funcString);
    data = await loadData(100, [-1,1], func, 0.1);
    model = createModelDisplay();
    const trainResults = await train(model, data);
    predict(trainResults);
}

//get data to be trained on   
async function loadData(N, interval, func, noiseVariance) {
  data = await getFunctionData(N, interval, func, noiseVariance);
  if (!data) {
    console.error('No data returned from getFunctionData()');
    return;
  }

  const values = data.map(d => ({
    x: d.x,
    y: d.y,
  }));

  if (!values || !values.length) {
    console.error('Values array is empty');
    return;
  }

  displayData(values);  // Display the data
  
  return data;
}


//split Data
function splitData(splitRatio) {
  // Initialize data and testdata as empty arrays if they are undefined
  data = data || [];
  testdata = testdata || [];

  // Combine data and testdata
  let combinedData = [...data, ...testdata];

  // Check if all the datapoints from data and testdata are in the combined dataset
  let isAllDataPointsPresent = data.every(val => combinedData.includes(val)) && testdata.every(val => combinedData.includes(val));

  if (!isAllDataPointsPresent) {
    console.error("Not all data points from data and testdata are present in the combined dataset.");
    return;
  }

  // Check if combinedData has only the datapoints from data and testdata
  let hasOnlyDataPoints = combinedData.every(val => data.includes(val) || testdata.includes(val));

  if (!hasOnlyDataPoints) {
    console.error("combinedData has extra data points not present in data or testdata.");
    return;
  }

  // Shuffle the array
  tf.util.shuffle(combinedData);

  // Calculate the split index
  const splitIdx = Math.floor(combinedData.length * splitRatio);

  // Split the array into training data and test data
  testdata = combinedData.slice(0, splitIdx);
  data = combinedData.slice(splitIdx);
}

//display data
function displayData(trainingValues, testValues) {
  // show data scatterplot
  const data_surface = document.getElementById('datadisplay');
  if (!data_surface) {
    console.error('No HTML element found with id "datadisplay"');
    return;
  }
  
  // Prepare the data for tfvis
  const data = {values: [], series: []};
  if (trainingValues && trainingValues.length > 0) {
    data.values.push(trainingValues);
    data.series.push('Training Data');
  }
  if (testValues && testValues.length > 0) {
    data.values.push(testValues);
    data.series.push('Test Data');
  }
  //console.log('Training Values:', trainingValues);
  //console.log('Test Values:', testValues);

  tfvis.render.scatterplot(
    data_surface,
    data,
    {
      xLabel: 'X',
      yLabel: 'Y(X)',
      height: 300
    }
  );
}

//creates and displays model
function createModelDisplay() {
  model = createModel();
  displayModel(model);  
  return model;
}

//train data
async function train(model, data) {
  // Convert the data to a form we can use for training.
  tensorData = convertToTensor(data);
  const {inputs, labels} = tensorData;

  // Pass the parameters to the trainModel function
  await trainModel(model, inputs, labels, globalOptimizer, globalLearningRate, globalBatchSize, globalEpochs);
  console.log('Done Training');
  
  return {model, data, tensorData};
}

//calls predictions
function predict({model, data, tensorData}) {
  testModel(model, data, tensorData);
}

//save model to download file
/*async function saveModel(model) {
  const saveResult = await model.save('downloads://bestmodel');
  console.log('Model saved successfully:', saveResult);
}*/
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

//load model from save file
async function loadModel(files) {
  try {
    model = await tf.loadLayersModel(tf.io.browserFiles(files));
    console.log('Model loaded successfully');
    return model;
  } catch (error) {
    console.error('Error loading model:', error);
  }
}


document.addEventListener('DOMContentLoaded', run);

/**************************************************************get data for file. have to adjust labels to make it work again though***************************************************************/
/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
async function getData() {
  const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
  const carsData = await carsDataResponse.json();
  const cleaned = carsData.map(car => ({
    mpg: car.Miles_per_Gallon,
    horsepower: car.Horsepower,
  }))
  .filter(car => (car.mpg != null && car.horsepower != null));

  return cleaned;
}

/**************************************************************create own data from function********************************************************************************************************/

function y(x) {
  // Define the function here
  //return Math.sin(x);
  return (x + 0.8) * (x - 0.2) * (x - 0.3) * (x - 0.6);
}

async function getFunctionData(N, interval, func, noiseVariance) {
  const data = [];
  const step = (interval[1] - interval[0]) / N;
  const sampleMethod = document.getElementById('sample-method').value;

  for(let i = 0; i < N; i++) {
    let x;
    if (sampleMethod === 'randomEqual') {
      const xMin = interval[0] + i * step;
      const xMax = xMin + step;
      x = randomInRange(xMin, xMax);
    } else if (sampleMethod === 'random') {
      x = randomInRange(interval[0], interval[1]);
    } else { // equal
      x = interval[0] + i * step;
    }

    let yValue = func(x); //y(x)
    yValue += gaussianNoise(noiseVariance); // Add Gaussian noise to y
    data.push({x: x, y: yValue});
  }
  return data;
}

function randomInRange(min, max) {
  return Math.random() * (max - min) + min;
}

// Function to generate Gaussian noise
function gaussianNoise(variance = 0.1) {
  let u = 0, v = 0;
  while(u === 0) u = Math.random(); // Converting [0,1) to (0,1)
  while(v === 0) v = Math.random();
  return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v ) * Math.sqrt(variance);
}


//gets data from input html fields
function useHtmlValues() {
  // Get the values from the input fields
  const funcString = document.getElementById('function').value;
  const noiseVariance = parseFloat(document.getElementById('variance').value);
  const N = parseInt(document.getElementById('samples').value);
  const interval = [
    parseFloat(document.getElementById('intervalStart').value),
    parseFloat(document.getElementById('intervalEnd').value)
  ];

  // Convert the function string to a function
  const func = new Function('x', 'return ' + funcString);

  // Call loadData with the values from the input fields
  loadData(N, interval, func, noiseVariance);
}

/************************************************************** create model ************************************************************************************************************************/
function createModel() {
  // Create a sequential model
  const model = tf.sequential();

  // Add a single input layer
  model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));

  // Add an output layer
  model.add(tf.layers.dense({units: 1, useBias: true}));

  return model;
}
/************************************************************** display model ************************************************************************************************************************/
//display currently used model
function displayModel(model) {
  const model_surface = document.getElementById('modeldisplay');
  if (!model_surface) {
    console.error('No HTML element found with id "modeldisplay"');
    return;
  }
  model_surface.innerHTML = ''; // Clear the model display

  // Get the layers of the model
  const layers = model.layers;

  // Create a table to display the layer information
  const table = document.createElement('table');

  // Add table headers
  const headers = document.createElement('tr');
  headers.innerHTML = '<th>Layer</th><th>Units</th><th>Activation</th><th>Use Bias</th>';
  table.appendChild(headers);

  // Add a row for each layer
  layers.forEach((layer, i) => {
    const row = document.createElement('tr');

    // Get the layer config
    const config = layer.getConfig();

    // Add the layer information to the row
    row.innerHTML = `<td>Layer ${i+1}</td><td>${config.units}</td><td>${config.activation}</td><td>${config.useBias}</td>`;

    // Add the row to the table
    table.appendChild(row);
  });

  // Add the table to the model display
  model_surface.appendChild(table);
}

/************************************************************** convert to tensor ************************************************************************************************************************/
/**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
function convertToTensor(data) {
  // Wrapping these calculations in a tidy will dispose any
  // intermediate tensors.

  return tf.tidy(() => {
    // Step 1. Shuffle the data
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor
    const inputs = data.map(d => d.x); //horsepower
    const labels = data.map(d => d.y); //mpg

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    }
  });
}



/************************************************************** train model ************************************************************************************************************************/
async function trainModel(model, inputs, labels, optimizerName = 'adam', learningRate = 0.01, batchSize = 32, epochs = 50) {
  console.log(`Optimizer: ${optimizerName}`);
  console.log(`Learning Rate: ${learningRate}`);
  console.log(`Batch Size: ${batchSize}`);
  console.log(`Epochs: ${epochs}`);  

   // Display the training parameters
  displayTrainingParameters(optimizerName, learningRate, batchSize, epochs);

    const optimizers = {
        'sgd': tf.train.sgd,
        'adam': tf.train.adam,
        'adagrad': tf.train.adagrad,
        'adadelta': tf.train.adadelta,
        'adamax': tf.train.adamax,
        'rmsprop': tf.train.rmsprop,
        // Add any other optimizers you want to use here
    };

  const optimizer = optimizers[optimizerName];
  if (!optimizer) {
    console.error(`Optimizer not recognized: ${optimizerName}`);
    return;
  }

  // Prepare the model for training.
  model.compile({
    optimizer: optimizer(learningRate),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });

  const training_surface = document.getElementById('trainingdisplay');
  if (!training_surface) {
    console.error('No HTML element found with id "trainingdisplay"');
    return;
  }

  // After training the model, get the final MSE and loss
  const predictions = model.predict(inputs);
  const loss = tf.losses.meanSquaredError(labels, predictions);
  const mse = tf.metrics.meanSquaredError(labels, predictions);

  // Convert MSE and loss to regular JavaScript numbers
  globalTrainingLoss = await loss.dataSync()[0];
  globalTrainingMSE = await mse.dataSync()[0];

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks( //show training performance
        training_surface,  
        ['loss', 'mse'],
        { height: 200, callbacks: ['onEpochEnd'] }
    )
  });
}
/************************************************************** display training parameters *******************************************************************************************************/
function displayTrainingParameters(optimizer, learningRate, batchSize, epochs) {
  const training_surface = document.getElementById('trainingdisplay');
  if (!training_surface) {
    console.error('No HTML element found with id "trainingdisplay"');
    return;
  }
  training_surface.innerHTML = ''; // Clear the training display

  // Create a table to display the training parameters
  const table = document.createElement('table');

  // Add table headers
  const headers = document.createElement('tr');
  headers.innerHTML = '<th>Parameter</th><th>Value</th>';
  table.appendChild(headers);

  // Add a row for each parameter
  const parameters = {optimizer, learningRate, batchSize, epochs};
  for (const [key, value] of Object.entries(parameters)) {
    const row = document.createElement('tr');
    row.innerHTML = `<td>${key}</td><td>${value}</td>`;
    table.appendChild(row);
  }

  // Add the table to the training display
  training_surface.appendChild(table);
}

/************************************************************** test model - training ***********************************************************************************************************/
function testModel(model, inputData, normalizationData) {
  const {originalPoints, predictedPoints} = generatePredictions(model, inputData, normalizationData);

  // Call the new function to display the data
  displayDataGraph(originalPoints, predictedPoints);

  // Display the final MSE and loss
  displayResults(globalTrainingMSE, globalTrainingLoss);
}

// displaying the data graph for training
function displayDataGraph(originalPoints, predictedPoints) {
  const predictions_surface = document.getElementById('predictionsdisplay');
  if (!predictions_surface) {
    console.error('No HTML element found with id "predictionsdisplay"');
    return;
  }
  tfvis.render.scatterplot(
    predictions_surface,
    //{name: 'Model Predictions vs Original Data'},
    {values: [originalPoints, predictedPoints], series: ['original', 'predicted']},
    {
      xLabel: 'X', //Horsepower
      yLabel: 'Y(X)', //MPG
      height: 300
    }
  );
}

//displaying the result table for training
function displayResults(mse, loss) {
  const results_surface = document.getElementById('resultsdisplay');
  if (!results_surface) {
    console.error('No HTML element found with id "resultsdisplay"');
    return;
  }
  results_surface.innerHTML = ''; // Clear the results display

  // Create a table to display the results
  const table = document.createElement('table');

  // Add table headers
  const headers = document.createElement('tr');
  headers.innerHTML = '<th>Parameter</th><th>Value</th>';
  table.appendChild(headers);

  // Add a row for each result
  const results = {mse, loss};
  for (const [key, value] of Object.entries(results)) {
    const row = document.createElement('tr');
    row.innerHTML = `<td>${key}</td><td>${value.toExponential(2)}</td>`; // Convert value to scientific notation with 2 decimal places
    table.appendChild(row);
  }

  // Add the table to the results display
  results_surface.appendChild(table);
}
/************************************************************** test model - testing ***********************************************************************************************************/

//basic logic to do testdata evaluations, and display both the graph and the result table
async function evaluateTestdata(trainingValues, testValues){
    // Generate predictions for the test data
    const testResults = generatePredictions(model, testdata, tensorData);

    // Calculate the MSE and loss for the test data
    const testLabels = tf.tensor(testdata.map(d => d.y));
    const testPredictions = tf.tensor(testResults.predictedPoints.map(p => p.y));
    const testLoss = tf.losses.meanSquaredError(testLabels, testPredictions);
    const testMSE = tf.metrics.meanSquaredError(testLabels, testPredictions);
    console.log('Test Loss:', testLoss);

    // Assign the MSE and loss to the global variables
    globalTestLoss = await testLoss.dataSync()[0];
    globalTestMSE = await testMSE.dataSync()[0];
    console.log('Global Test Loss:', globalTestLoss);

    // Display the results in a table
    displayTestResultsTable();

    // Display the test data and its predictions
    displayTestDataGraph(trainingValues, testValues, testResults.predictedPoints);
}


//generate predictions (for test dataset)
function generatePredictions(model, inputData, normalizationData) {
  const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

  const [xs, preds] = tf.tidy(() => {
    const xsNorm = tf.tensor(inputData.map(d => d.x))  // use actual x-values
      .sub(inputMin)
      .div(inputMax.sub(inputMin));

    const predictions = model.predict(xsNorm.reshape([inputData.length, 1]));

    const unNormXs = xsNorm
      .mul(inputMax.sub(inputMin))
      .add(inputMin);

    const unNormPreds = predictions
      .mul(labelMax.sub(labelMin))
      .add(labelMin);

    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    return {x: val, y: preds[i]}
  });

  const originalPoints = inputData.map(d => ({
    x: d.x, y: d.y,
  }));

  return {originalPoints, predictedPoints};
}


//display result graph, but for test data
function displayTestDataGraph(trainingPoints, testPoints, predictedPoints) {
  const tpredictions_surface = document.getElementById('test-predictionsdisplay');
  if (!tpredictions_surface) {
    console.error('No HTML element found with id "test-predictionsdisplay"');
    return;
  }
  tfvis.render.scatterplot(
    tpredictions_surface,
    {values: [trainingPoints, testPoints, predictedPoints], series: ['Training Data', 'Test Data', 'Test Predictions']},
    {
      xLabel: 'X',
      yLabel: 'Y(X)',
      height: 300
    }
  );
}

//displaying the result table for testing
function displayTestResultsTable() {
  const results_surface = document.getElementById('test-resultsdisplay');
  if (!results_surface) {
    console.error('No HTML element found with id "test-resultsdisplay"');
    return;
  }
  results_surface.innerHTML = ''; // Clear the results display

  // Create a table to display the results
  const table = document.createElement('table');

  // Add table headers
  const headers = document.createElement('tr');
  headers.innerHTML = '<th>Parameter</th><th>Value</th><th>Percentage Change </br> from Training Results</th>';
  table.appendChild(headers);

  // Add a row for each result
  const results = {MSE: globalTestMSE, Loss: globalTestLoss};
  for (const [key, value] of Object.entries(results)) {
    const row = document.createElement('tr');
    const trainingValue = key === 'MSE' ? globalTrainingMSE : globalTrainingLoss;
    const percentageChange = ((value - trainingValue) / trainingValue) * 100;
    row.innerHTML = `<td>${key}</td><td>${value.toExponential(2)}</td><td>${percentageChange.toFixed(2)}%</td>`; // Convert value and percentageChange to scientific notation with 2 decimal places
    table.appendChild(row);
  }

  // Add the table to the results display
  results_surface.appendChild(table);
}

/************************************************************** html buttons ************************************************************************************************************************/
document.getElementById('data-btn').addEventListener('click', async () => {
  //data = await loadData();
  await useHtmlValues();
});

document.getElementById('model-btn').addEventListener('click', () => {
  model = createModelDisplay();
});

/***********************************************************upload/download data**********************************************/
document.getElementById('save-data-btn').addEventListener('click', saveDataToFile);
document.getElementById('load-data-btn').addEventListener('click', loadDataFromFile);

async function saveDataToFile() {
  // Save data
  let a = document.createElement('a');
  a.download = 'data.json';
  a.href = URL.createObjectURL(new Blob([JSON.stringify(data)], {type: 'application/json'}));
  a.click();
  console.log('Data saved successfully');

  // Get the checkbox
  const checkbox = document.getElementById('testdata-apply-checkbox');

  // Check if testdata is initialized, not empty, and the checkbox is checked
  if (testdata && testdata.length > 0 && checkbox.checked) {
    // Save testdata
    a = document.createElement('a');
    a.download = 'testdata.json';
    a.href = URL.createObjectURL(new Blob([JSON.stringify(testdata)], {type: 'application/json'}));
    a.click();
    console.log('Test data saved successfully');
  }
}

function loadDataFromFile() {
  // Get the checkbox
  const checkbox = document.getElementById('testdata-apply-checkbox');

  // Load data
  const input = document.createElement('input');
  input.type = 'file';
  input.onchange = function(event) {
    const file = event.target.files[0];
    const reader = new FileReader();
    reader.onload = function(event) {
      data = JSON.parse(event.target.result);
      console.log('Data loaded successfully');
      
      // Prepare the data values
      const dataValues = data.map(d => ({
        x: d.x,
        y: d.y,
      }));

      // Check if the checkbox is checked
      if (checkbox.checked) {
        // Load testdata
        const testdataInput = document.createElement('input');
        testdataInput.type = 'file';
        testdataInput.onchange = function(event) {
          const file = event.target.files[0];
          const reader = new FileReader();
          reader.onload = function(event) {
            testdata = JSON.parse(event.target.result);
            console.log('Test data loaded successfully');

            // Prepare the testdata values
            const testdataValues = testdata.map(d => ({
              x: d.x,
              y: d.y,
            }));

            // Display both data and testdata
            displayData(dataValues, testdataValues);
          };
          reader.readAsText(file);
        };
        testdataInput.click();
      } else {
        // Display only data
        displayData(dataValues);
      }
    };
    reader.readAsText(file);
  };
  input.click();
}
/************************************************************************* upload/download models ******************************/
document.getElementById('save-btn').addEventListener('click', async () => {
  await saveModel(model);
});

document.getElementById('upload-btn').addEventListener('click', async () => {
  // Prompt the user to select a .json file
  const jsonUpload = document.createElement('input');
  jsonUpload.type = 'file';
  jsonUpload.accept = '.json';
  jsonUpload.click();

  jsonUpload.onchange = async () => {
    modelJson = jsonUpload.files[0];
    console.log('JSON file selected:', modelJson);

    // Prompt the user to select a .bin file
    const binUpload = document.createElement('input');
    binUpload.type = 'file';
    binUpload.accept = '.bin';
    binUpload.click();

    binUpload.onchange = async () => {
      modelWeights = binUpload.files;
      console.log('BIN file selected:', modelWeights);

      // Load the model using the selected .json and .bin files
      model = await loadModel([modelJson, ...modelWeights]);
      displayModel(model); // Display the loaded model
      console.log('Model display updated'); // Confirm that the model display has been updated

      // Clear the training section
      const trainingSection = document.getElementById('trainingdisplay');
      if (trainingSection) {
        trainingSection.innerHTML = '';
        console.log('Training section cleared');
      }

      // Update the predictions
      predict({model, data, tensorData});
      console.log('Predictions updated');
    };
  };
});


/************************************************************************** model creator *************************************/
function addLayer(units = 10, activation = 'relu', useBias = true) {
  // Get the layer table
  const layerTable = document.getElementById('layer-table');

  // If there's at least one row, copy the values from the last row
  if (layerTable.rows.length > 1) {
    const lastRow = layerTable.rows[layerTable.rows.length - 1];
    units = parseInt(lastRow.cells[0].firstChild.value);
    activation = lastRow.cells[1].firstChild.value;
    useBias = lastRow.cells[2].firstChild.firstChild.checked; 
  }

  // Create a new row
  const row = document.createElement('tr');

  // Create the 'units' field
  const unitsCell = document.createElement('td');
  const unitsInput = document.createElement('input');
  unitsInput.type = 'number';
  unitsInput.min = '1';
  unitsInput.value = units;
  unitsCell.appendChild(unitsInput);
  row.appendChild(unitsCell);

  // Create the 'activation' field
  const activationCell = document.createElement('td');
  const activationSelect = document.createElement('select');
  ['elu', 'hardSigmoid', 'linear', 'relu', 'relu6', 'selu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh'].forEach(act => {
    const option = document.createElement('option');
    option.value = act;
    option.text = act;
    if (act === activation) {
      option.selected = true;
    }
    activationSelect.appendChild(option);
  });
  activationCell.appendChild(activationSelect);
  row.appendChild(activationCell);

  // Create the 'use bias' field
  const useBiasCell = document.createElement('td');
  const useBiasLabel = document.createElement('label');
  useBiasLabel.className = 'switch';
  const useBiasCheckbox = document.createElement('input');
  useBiasCheckbox.type = 'checkbox';
  useBiasCheckbox.checked = useBias;
  useBiasLabel.appendChild(useBiasCheckbox);
  const useBiasSlider = document.createElement('span');
  useBiasSlider.className = 'slider round';
  useBiasLabel.appendChild(useBiasSlider);
  useBiasCell.appendChild(useBiasLabel);
  row.appendChild(useBiasCell);

  // Create the 'remove' button
  const removeCell = document.createElement('td');
  const removeButton = document.createElement('button');
  removeButton.textContent = 'Remove';
  removeButton.addEventListener('click', () => {
    layerTable.removeChild(row);
  });
  removeCell.appendChild(removeButton);
  row.appendChild(removeCell);

  // Add the new row to the table
  layerTable.appendChild(row);
}

document.getElementById('add-layer-btn').addEventListener('click', () => {
  addLayer();
});

// Add two layers at the beginning
addLayer();
addLayer();

//***************************************************************** use the model from creator ****************************************/
document.getElementById('use-model-btn').addEventListener('click', async () => {
  // Create a new sequential model
  const newModel = tf.sequential();

  // Get the layer table
  const layerTable = document.getElementById('layer-table');

  // Iterate over each row in the table
  for (let i = 1; i < layerTable.rows.length; i++) {
    const row = layerTable.rows[i];

    // Get the layer parameters from the row
    let units = parseInt(row.cells[0].firstChild.value);
    const activation = row.cells[1].firstChild.value;
    const useBias = row.cells[2].firstChild.checked;

    // If this is the last row, set the units to 1
    if (i === layerTable.rows.length - 1) {
      units = 1;
      row.cells[0].firstChild.value = 1;  // Update the table
    }

    // Add a dense layer to the model with the given parameters
    if (i === 1) {
      // For the first layer, specify the input shape
      newModel.add(tf.layers.dense({units, activation, useBias, inputShape: [1]}));
    } else {
      newModel.add(tf.layers.dense({units, activation, useBias}));
    }
  }

  // Use the new model
  model = newModel;
  console.log('New model created:', model);


  displayModel(model); // Display the loaded model
  console.log('Model display updated'); // Confirm that the model display has been updated

  // Clear the training section
  const trainingSection = document.getElementById('trainingdisplay');
  if (trainingSection) {
    trainingSection.innerHTML = '';
    console.log('Training section cleared due to new model');
  }

  //clear the training predictions section
  const predictionsSection = document.getElementById('predictionsdisplay');
  if (predictionsSection) {
    predictionsSection.innerHTML = '';
    console.log('prediction section cleared due to new model');
  }
  const resultsSection = document.getElementById('resultsdisplay');
  if (resultsSection) {
    resultsSection.innerHTML = '';
    console.log('result section cleared due to new model');
  }

  //clear the test section
  const testSection = document.getElementById('test-predictionsdisplay');
  if (testSection) {
    testSection.innerHTML = '';
    console.log('test graph section cleared due to new model');
  }
  const tresultsSection = document.getElementById('test-resultsdisplay');
  if (tresultsSection) {
    tresultsSection.innerHTML = '';
    console.log('test result section cleared due to new model');
  }

  // Check if the 'auto-train-checkbox' is checked
  const autoTrainCheckbox = document.getElementById('auto-train-checkbox');
  if (autoTrainCheckbox.checked) {
    // If it's checked, automatically train the model and display the predictions
    const trainResults = await train(model, data);
    model = trainResults.model;
    tensorData = trainResults.tensorData;
    predict({model, data, tensorData});
    console.log('Model trained and predictions displayed');
  }
});

/**************************************************************** parse values from html to data set ***********************************/
document.getElementById('use-values-btn').addEventListener('click', async () => {
  await useHtmlValues();

    // Clear the data display section
  const trainingSection = document.getElementById('trainingdisplay');
  if (trainingSection) {
    trainingSection.innerHTML = '';
    console.log('Training section cleared due to new model');
  }

  //clear the training section
  const predictionsSection = document.getElementById('predictionsdisplay');
  if (predictionsSection) {
    predictionsSection.innerHTML = '';
    console.log('prediction section cleared due to new model');
  }
  const resultsSection = document.getElementById('resultsdisplay');
  if (resultsSection) {
    resultsSection.innerHTML = '';
    console.log('result section cleared due to new model');
  }

  //clear the test section
  const testSection = document.getElementById('test-predictionsdisplay');
  if (testSection) {
    testSection.innerHTML = '';
    console.log('test graph section cleared due to new model');
  }
  const tresultsSection = document.getElementById('test-resultsdisplay');
  if (tresultsSection) {
    tresultsSection.innerHTML = '';
    console.log('test result section cleared due to new model');
  }

  // Check if the 'auto-train-checkbox' is checked
  const autoTrainCheckbox = document.getElementById('auto-train-checkbox2');
  if (autoTrainCheckbox.checked) {
    // If it's checked, automatically train the model and display the predictions
    const trainResults = await train(model, data);
    model = trainResults.model;
    tensorData = trainResults.tensorData;
    predict({model, data, tensorData});
    console.log('Model trained and predictions displayed');
  }
});


/************************************************************** model training *************************************/
document.getElementById('train-btn').addEventListener('click', async () => {
    // Get the values from the HTML fields
    globalOptimizer = document.getElementById('optimizer').value;
    globalLearningRate = parseFloat(document.getElementById('learningRate').value);
    globalBatchSize = parseInt(document.getElementById('batchSize').value);
    globalEpochs = parseInt(document.getElementById('epochs').value);

    console.log(`Button Click - Optimizer: ${globalOptimizer}`);
    console.log(`Button Click - Learning Rate: ${globalLearningRate}`);
    console.log(`Button Click - Batch Size: ${globalBatchSize}`);
    console.log(`Button Click - Epochs: ${globalEpochs}`);

   //clear the test section
  const testSection = document.getElementById('test-predictionsdisplay');
  if (testSection) {
    testSection.innerHTML = '';
    console.log('test graph section cleared due to new model');
  }
  const tresultsSection = document.getElementById('test-resultsdisplay');
  if (tresultsSection) {
    tresultsSection.innerHTML = '';
    console.log('test result section cleared due to new model');
  }



    // Pass the values to the train function
    const trainResults = await train(model, data, globalOptimizer, globalLearningRate, globalBatchSize, globalEpochs);

    // Use the trained model to make predictions
    predict(trainResults);
});


/*********************************** test data training data split ********************************************************/
function updateLabel(value) {
  const label = document.getElementById('split-label');
  label.textContent = `Test Dataset: ${value}%, Training Dataset: ${100-value}%`;
}

// Update the label when the page loads
window.onload = function() {
  updateLabel(document.getElementById('split-slider').value);
};

document.getElementById('split-slider').addEventListener('input', function() {
  updateLabel(this.value);
});

document.getElementById('apply-split-btn').addEventListener('click', async function() {
  
  //clear the test section
  const testSection = document.getElementById('test-predictionsdisplay');
  if (testSection) {
    testSection.innerHTML = '';
    console.log('test graph section cleared due to new split');
  }
  const tresultsSection = document.getElementById('test-resultsdisplay');
  if (tresultsSection) {
    tresultsSection.innerHTML = '';
    console.log('test result section cleared due to new split');
  }


  const splitRatio = document.getElementById('split-slider').value / 100;  // Convert to a value between 0 and 1
  
  // Split data using the splitRatio
  splitData(splitRatio);

  // Prepare the values for displayData
  const trainingValues = data.map(d => ({x: d.x, y: d.y}));
  const testValues = testdata.map(d => ({x: d.x, y: d.y}));

  // Display the data
  displayData(trainingValues, testValues);

  // Check the state of the checkbox
  const autoTrainCheckbox = document.getElementById('auto-train-checkbox3');
  if (autoTrainCheckbox.checked) {
    // Pass the values to the train function
    const trainResults = await train(model, data, globalOptimizer, globalLearningRate, globalBatchSize, globalEpochs);

    // Use the trained model to make predictions
    predict(trainResults);
  }
  
  //auto-test
  const autoTestCheckbox = document.getElementById('auto-test-checkbox');
    if (autoTestCheckbox.checked) {
        await evaluateTestdata(trainingValues, testValues);   
    }

});

/**************************************** test data button    ****************************/
document.getElementById('predict-btn').addEventListener('click', async () => {
    // Check if testdata is defined and has data
    if (testdata && Array.isArray(testdata) && testdata.length > 0) {
        // Prepare the values for displayData
        const trainingValues = data.map(d => ({x: d.x, y: d.y}));
        const testValues = testdata.map(d => ({x: d.x, y: d.y}));
        
        // Call evaluateTestdata
        await evaluateTestdata(trainingValues, testValues);   
    }
});

/***************************************************************** using examples *********************************************************************/

// model definition
const models = {
  'model1': {
    json: {"modelTopology":{"class_name":"Sequential","config":{"name":"sequential_2","layers":[{"class_name":"Dense","config":{"units":1,"activation":"linear","use_bias":true,"kernel_initializer":{"class_name":"VarianceScaling","config":{"scale":1,"mode":"fan_avg","distribution":"normal","seed":null}},"bias_initializer":{"class_name":"Zeros","config":{}},"kernel_regularizer":null,"bias_regularizer":null,"activity_regularizer":null,"kernel_constraint":null,"bias_constraint":null,"name":"dense_Dense3","trainable":true,"batch_input_shape":[null,1],"dtype":"float32"}}]},"keras_version":"tfjs-layers 2.0.0","backend":"tensor_flow.js"},"format":"layers-model","generatedBy":"TensorFlow.js tfjs-layers v2.0.0","convertedBy":null,"weightsManifest":[{"paths":["./bestmodel.weights.bin"],"weights":[{"name":"dense_Dense3/kernel","shape":[1,1],"dtype":"float32"},{"name":"dense_Dense3/bias","shape":[1],"dtype":"float32"}]}]},
    bin: new Uint8Array([190,35,206,61,69,75,193,62]) // replace with your BIN data
  },
  'model2': {
    json: {"modelTopology":{"class_name":"Sequential","config":{"name":"sequential_20","layers":[{"class_name":"Dense","config":{"units":8,"activation":"sigmoid","use_bias":true,"kernel_initializer":{"class_name":"VarianceScaling","config":{"scale":1,"mode":"fan_avg","distribution":"normal","seed":null}},"bias_initializer":{"class_name":"Zeros","config":{}},"kernel_regularizer":null,"bias_regularizer":null,"activity_regularizer":null,"kernel_constraint":null,"bias_constraint":null,"name":"dense_Dense69","trainable":true,"batch_input_shape":[null,1],"dtype":"float32"}},{"class_name":"Dense","config":{"units":4,"activation":"sigmoid","use_bias":true,"kernel_initializer":{"class_name":"VarianceScaling","config":{"scale":1,"mode":"fan_avg","distribution":"normal","seed":null}},"bias_initializer":{"class_name":"Zeros","config":{}},"kernel_regularizer":null,"bias_regularizer":null,"activity_regularizer":null,"kernel_constraint":null,"bias_constraint":null,"name":"dense_Dense70","trainable":true}},{"class_name":"Dense","config":{"units":3,"activation":"sigmoid","use_bias":true,"kernel_initializer":{"class_name":"VarianceScaling","config":{"scale":1,"mode":"fan_avg","distribution":"normal","seed":null}},"bias_initializer":{"class_name":"Zeros","config":{}},"kernel_regularizer":null,"bias_regularizer":null,"activity_regularizer":null,"kernel_constraint":null,"bias_constraint":null,"name":"dense_Dense71","trainable":true}},{"class_name":"Dense","config":{"units":1,"activation":"softplus","use_bias":true,"kernel_initializer":{"class_name":"VarianceScaling","config":{"scale":1,"mode":"fan_avg","distribution":"normal","seed":null}},"bias_initializer":{"class_name":"Zeros","config":{}},"kernel_regularizer":null,"bias_regularizer":null,"activity_regularizer":null,"kernel_constraint":null,"bias_constraint":null,"name":"dense_Dense72","trainable":true}}]},"keras_version":"tfjs-layers 2.0.0","backend":"tensor_flow.js"},"format":"layers-model","generatedBy":"TensorFlow.js tfjs-layers v2.0.0","convertedBy":null,"weightsManifest":[{"paths":["./bestmodel.weights.bin"],"weights":[{"name":"dense_Dense69/kernel","shape":[1,8],"dtype":"float32"},{"name":"dense_Dense69/bias","shape":[8],"dtype":"float32"},{"name":"dense_Dense70/kernel","shape":[8,4],"dtype":"float32"},{"name":"dense_Dense70/bias","shape":[4],"dtype":"float32"},{"name":"dense_Dense71/kernel","shape":[4,3],"dtype":"float32"},{"name":"dense_Dense71/bias","shape":[3],"dtype":"float32"},{"name":"dense_Dense72/kernel","shape":[3,1],"dtype":"float32"},{"name":"dense_Dense72/bias","shape":[1],"dtype":"float32"}]}]}, // replace with your JSON data
    bin: new Uint8Array([195,152,241,192,51,206,83,64,86,221,247,192,236,188,199,192,26,16,3,193,218,189,119,192,150,164,156,192,56,201,2,193,221,214,51,63,209,127,51,192,57,176,184,62,197,2,76,191,235,223,118,62,218,14,90,64,70,3,128,64,171,25,161,63,109,213,158,191,229,222,133,192,245,8,20,192,106,213,106,63,221,130,30,191,55,109,242,191,119,27,47,64,165,175,25,192,132,220,19,192,230,64,143,192,26,190,10,192,8,112,209,63,148,172,158,191,78,166,97,192,174,92,198,190,156,111,9,63,64,140,71,192,251,66,141,192,235,21,27,192,91,130,14,64,195,62,22,64,232,231,201,61,131,134,104,192,95,131,252,191,39,181,27,64,206,179,193,63,48,184,153,192,145,158,9,192,136,191,130,192,19,46,187,192,131,97,14,192,57,87,43,64,70,18,97,63,5,255,58,61,179,215,216,62,122,55,3,192,225,100,16,64,123,20,38,64,195,246,200,191,114,98,203,191,96,205,31,191,170,212,35,64,7,174,99,192,234,203,138,192,33,10,59,64,155,252,71,191,163,144,231,191,36,60,253,62,235,6,9,191,170,179,144,190,74,217,62,190,1,180,151,191,83,235,161,191,60,252,208,63,16,244,177,62]) // replace with your BIN data
  },
  'model3': {
    json: {"modelTopology":{"class_name":"Sequential","config":{"name":"sequential_11","layers":[{"class_name":"Dense","config":{"units":5,"activation":"sigmoid","use_bias":true,"kernel_initializer":{"class_name":"VarianceScaling","config":{"scale":1,"mode":"fan_avg","distribution":"normal","seed":null}},"bias_initializer":{"class_name":"Zeros","config":{}},"kernel_regularizer":null,"bias_regularizer":null,"activity_regularizer":null,"kernel_constraint":null,"bias_constraint":null,"name":"dense_Dense49","trainable":true,"batch_input_shape":[null,1],"dtype":"float32"}},{"class_name":"Dense","config":{"units":7,"activation":"sigmoid","use_bias":true,"kernel_initializer":{"class_name":"VarianceScaling","config":{"scale":1,"mode":"fan_avg","distribution":"normal","seed":null}},"bias_initializer":{"class_name":"Zeros","config":{}},"kernel_regularizer":null,"bias_regularizer":null,"activity_regularizer":null,"kernel_constraint":null,"bias_constraint":null,"name":"dense_Dense50","trainable":true}},{"class_name":"Dense","config":{"units":4,"activation":"sigmoid","use_bias":true,"kernel_initializer":{"class_name":"VarianceScaling","config":{"scale":1,"mode":"fan_avg","distribution":"normal","seed":null}},"bias_initializer":{"class_name":"Zeros","config":{}},"kernel_regularizer":null,"bias_regularizer":null,"activity_regularizer":null,"kernel_constraint":null,"bias_constraint":null,"name":"dense_Dense51","trainable":true}},{"class_name":"Dense","config":{"units":1,"activation":"sigmoid","use_bias":true,"kernel_initializer":{"class_name":"VarianceScaling","config":{"scale":1,"mode":"fan_avg","distribution":"normal","seed":null}},"bias_initializer":{"class_name":"Zeros","config":{}},"kernel_regularizer":null,"bias_regularizer":null,"activity_regularizer":null,"kernel_constraint":null,"bias_constraint":null,"name":"dense_Dense52","trainable":true}}]},"keras_version":"tfjs-layers 2.0.0","backend":"tensor_flow.js"},"format":"layers-model","generatedBy":"TensorFlow.js tfjs-layers v2.0.0","convertedBy":null,"weightsManifest":[{"paths":["./bestmodel.weights.bin"],"weights":[{"name":"dense_Dense49/kernel","shape":[1,5],"dtype":"float32"},{"name":"dense_Dense49/bias","shape":[5],"dtype":"float32"},{"name":"dense_Dense50/kernel","shape":[5,7],"dtype":"float32"},{"name":"dense_Dense50/bias","shape":[7],"dtype":"float32"},{"name":"dense_Dense51/kernel","shape":[7,4],"dtype":"float32"},{"name":"dense_Dense51/bias","shape":[4],"dtype":"float32"},{"name":"dense_Dense52/kernel","shape":[4,1],"dtype":"float32"},{"name":"dense_Dense52/bias","shape":[1],"dtype":"float32"}]}]},
    bin: new Uint8Array([120,122,166,192,22,17,170,191,36,161,10,65,102,245,226,192,219,177,126,192,248,44,212,63,168,187,128,192,135,251,20,191,158,150,169,63,202,209,172,63,25,12,128,192,188,229,164,64,253,110,61,192,21,79,147,64,16,166,198,64,249,246,40,192,32,93,201,64,11,3,69,191,234,210,132,63,151,145,37,62,199,123,251,62,16,105,134,63,47,146,173,189,5,255,71,63,20,106,23,64,206,134,115,191,27,184,91,64,97,90,29,191,50,228,153,191,144,243,184,64,46,154,130,191,72,16,139,192,141,169,151,64,52,231,84,192,69,159,111,64,99,114,171,64,194,10,54,192,180,145,187,64,166,53,30,192,182,37,145,64,3,235,61,192,202,223,131,64,214,213,176,64,232,234,186,191,59,238,180,64,39,97,200,63,64,145,32,63,159,89,26,64,60,187,43,191,199,189,207,61,200,214,68,64,218,129,254,62,173,198,236,192,60,255,103,192,99,80,18,64,22,17,112,191,23,110,68,64,178,175,100,63,87,98,143,192,144,120,158,191,217,33,172,192,238,199,77,192,177,204,118,64,165,106,36,191,164,0,101,64,83,214,142,63,232,160,130,192,79,251,175,191,169,70,147,64,151,30,201,63,243,78,199,192,158,142,159,191,156,126,140,192,225,19,228,192,138,121,8,64,109,64,33,191,194,161,60,64,245,136,7,64,160,44,173,192,164,49,174,191,63,44,148,63,93,78,62,190,192,36,161,191,226,160,155,191,78,253,28,192,101,225,14,65,7,149,174,64,129,205,72,62,218,102,30,191]) // replace with your BIN data
  }
};/*,
  'model2': {
    json: {"modelTopology":{...}}, // replace with your JSON data
    bin: new Uint8Array([...]) // replace with your BIN data
  },
  'model3': {
    json: {"modelTopology":{...}}, // replace with your JSON data
    bin: new Uint8Array([...]) // replace with your BIN data
  },
  // add more models as needed
};*/




const datasets = {
  'model1': {
    data: [{"x":-0.3569561471924978,"y":-0.4566153191970793},{"x":-0.8232608447202902,"y":0.1016313959613033},{"x":-0.9177654023836698,"y":0.09515062188071133},{"x":0.8716125149664308,"y":0.026877594014358325},{"x":0.860118490693147,"y":0.26631049442608773},{"x":0.0485116740028681,"y":-0.050443188365735116},{"x":-0.5540528868094459,"y":0.1264337481895475},{"x":-0.8025473785411161,"y":-1.2438087940045754},{"x":0.9144056262118792,"y":0.5282517821096819},{"x":0.02113143587845309,"y":-0.030711208796738197},{"x":-0.045118372544499546,"y":-0.04305393893242224},{"x":-0.07047673071728006,"y":-0.004841444216596387},{"x":-0.021350704758726934,"y":0.09041322732942939},{"x":-0.05593577080246129,"y":0.010967067108398731},{"x":0.7028398405994579,"y":0.1525975442870924},{"x":-0.6763532372326,"y":-0.15566466139069318},{"x":0.1886786570237271,"y":0.019713706083696405},{"x":0.7819527128598274,"y":-0.16053644443421883},{"x":0.3166156755458053,"y":0.08914057819666409},{"x":-0.974848746569036,"y":0.7375542601918509},{"x":-0.31097946027669654,"y":-0.32006362547165707},{"x":0.0793897195498291,"y":0.1801997695454646},{"x":0.1729146695739048,"y":0.16044717455030494},{"x":-0.395855166487847,"y":0.05190537088225425},{"x":-0.5029909080708597,"y":-0.4092448297966728},{"x":0.31221838527127244,"y":0.20234777141113758},{"x":-0.8502180549230182,"y":-0.035717765955552705},{"x":-0.9750491438070055,"y":1.316739621046965},{"x":-0.8965121838832181,"y":0.1605893202593219},{"x":-0.3861332627506826,"y":0.22275892416895743},{"x":0.8313998734876836,"y":0.19548580133201204},{"x":0.27213771383573604,"y":-0.027857398822600285},{"x":0.98688420927537,"y":0.8151388485782984},{"x":-0.48800535379891896,"y":-0.3984869278253995},{"x":-0.5320767448070578,"y":-0.37304030238694075},{"x":0.4969484080038954,"y":0.09786518380082307},{"x":-0.7983459738324363,"y":-0.3668714032639924},{"x":-0.54393260446353,"y":0.1724762642737123},{"x":0.4383268726204958,"y":0.08370852529660257},{"x":0.43381042956392957,"y":-0.11381237056816637},{"x":0.4643829942313091,"y":0.020131363751134005},{"x":0.39357891635829334,"y":0.04950632593666455},{"x":-0.6902633906355112,"y":-0.13203190516344462},{"x":-0.2994233113020439,"y":0.3566409234564291},{"x":0.21502598873819703,"y":0.16259605317833747},{"x":-0.8571063939945193,"y":0.33721971786556515},{"x":-0.1404672400644657,"y":0.047028464956831134},{"x":0.5327248234813343,"y":0.034368094397254134},{"x":0.36864527630340665,"y":0.059359105729514444},{"x":0.3369631574753617,"y":-0.2698625611928425},{"x":0.9998544419081894,"y":0.5508214394500537},{"x":-0.08746012202085084,"y":0.12370397147020873},{"x":-0.2537435844665275,"y":-0.16954693631465856},{"x":-0.3618053477051045,"y":-0.3442795971287176},{"x":0.3978329665703298,"y":-0.27093444612611955},{"x":0.21195812266746059,"y":0.8227697167149997},{"x":-0.8719136004954047,"y":0.047457569816924144},{"x":-0.9457096792245869,"y":0.4593682526749262},{"x":-0.17896077545670042,"y":-0.1922095236223137},{"x":0.7361481928762093,"y":0.17691169534185455},{"x":0.9428190416433865,"y":0.5134446268699564},{"x":0.1780865861017014,"y":0.06191058277903824},{"x":-0.7095340364245497,"y":0.05047869245354332},{"x":0.2666441585122404,"y":-0.08338645482962362},{"x":-0.9538428462702366,"y":0.4000336311242146},{"x":-0.6241782171081177,"y":-0.37736120821978925},{"x":-0.7210434391393639,"y":-0.4398217517269169},{"x":-0.9851503083542889,"y":0.7602698433213622},{"x":-0.7382197749592121,"y":-0.2127845837541154},{"x":-0.7755979895358749,"y":0.15101746107100636},{"x":-0.9043501064121278,"y":0.23178097634167374},{"x":-0.051662899747672644,"y":0.0003520532349906083},{"x":0.7080482845886149,"y":0.5732282521003017},{"x":-0.8446477300735236,"y":0.008750006717970674},{"x":0.06066330619431754,"y":0.21303594852819346},{"x":-0.13103200625474468,"y":0.6947009730926907},{"x":-0.9970368439232197,"y":0.27214443391202403},{"x":0.6222945743162938,"y":0.23354327790935228},{"x":-0.35091409243043326,"y":-0.19773508143496069},{"x":0.8041736683004043,"y":0.2119149351577025},{"x":0.5350513292528362,"y":0.2801798031718227},{"x":0.9021976987921468,"y":0.17148588096511616},{"x":-0.5800276332196436,"y":-0.02458914476542909},{"x":-0.1994864993799928,"y":-0.14205385770556875},{"x":0.24868152167465757,"y":0.14714735364084683},{"x":-0.01801829051379463,"y":0.011169220325011407},{"x":-0.11059850603333182,"y":-0.013308700336654856},{"x":0.3297718457258202,"y":-0.13925222879558474},{"x":0.5467022982550765,"y":0.2500505406049684},{"x":-0.5297869574379854,"y":-0.2389787621749173},{"x":0.5448825048211959,"y":-0.1293586929490929},{"x":0.5006260957651557,"y":0.20676393075091865},{"x":0.3943106609461837,"y":-0.8188367577500699},{"x":0.9334266529065287,"y":0.3795249632189536},{"x":0.525023418024675,"y":-0.06969873134662476},{"x":-0.2708842398476886,"y":-0.17149158391153282},{"x":0.288609023182849,"y":-0.02938483949953546},{"x":0.8206845839657199,"y":-0.07922626209239696},{"x":-0.8393481651304451,"y":0.16149743270861716},{"x":0.7901359274182105,"y":-0.009326164625266573},{"x":-0.3054256098987067,"y":-0.2605167400594646},{"x":-0.8344710195920247,"y":-0.07252190002107808},{"x":-0.9229461109265661,"y":0.33498410458983274},{"x":-0.21887513886302287,"y":0.2042090235816763},{"x":0.6098023153082861,"y":-0.3033870317688277},{"x":0.760730852636307,"y":0.06980629314157714},{"x":0.6929937423340095,"y":0.2147349101755052},{"x":-0.2797028908567134,"y":-0.06802759544612863},{"x":0.2269914261801227,"y":-0.3987427252954102},{"x":0.7869209070770837,"y":0.09004286926261328},{"x":-0.7873980597104812,"y":-0.07455358484956601},{"x":0.2922759301485283,"y":0.001451768187725928},{"x":-0.23066312353194732,"y":0.2779262460117633},{"x":-0.6408555615914954,"y":-0.2514141411901278},{"x":-0.33998460540642467,"y":-0.22301178686131962},{"x":0.8782593803626377,"y":0.3614531146191078},{"x":-0.6038085628851452,"y":-0.2108225019335851},{"x":-0.5974177387109946,"y":-0.3306083129774557},{"x":-0.24559164369314346,"y":-0.23962257744958754},{"x":-0.8793784049417864,"y":0.07993511027327227},{"x":0.6053530958185727,"y":-0.752741074619679},{"x":0.7324146166377145,"y":-0.11927330107276046},{"x":-0.42871197188813587,"y":-0.18951966928684444},{"x":-0.9427604539194467,"y":0.35948800342104337},{"x":-0.06531095459928427,"y":0.08816345914276164},{"x":-0.11877723438207807,"y":-0.14479328866966487},{"x":-0.6593509669020636,"y":-0.3688406479911286},{"x":-0.8118565538090988,"y":-0.33610480414450866},{"x":0.09146796506887428,"y":-0.030361961247349384},{"x":0.7762654082242122,"y":0.12268611424884604},{"x":0.8114736763663639,"y":-0.12106957365431775},{"x":0.7232259355361966,"y":0.295337874982314},{"x":0.14632967112431078,"y":-0.19960541254266362},{"x":0.378893945547259,"y":-0.24401177442982216},{"x":-0.9130335774372802,"y":0.12198742083168519},{"x":-0.6317467699950966,"y":0.2522623700624469},{"x":0.6958220678532685,"y":-0.12941967170437357},{"x":0.34349301079382566,"y":0.30004545562582036},{"x":0.11821847122784598,"y":-0.1334612853761969},{"x":-0.21260638607231172,"y":-0.13956906699270039},{"x":0.9294394299224495,"y":0.6196479858666938},{"x":-0.5665752938253961,"y":-0.08170574753007886},{"x":0.8816212016638161,"y":-0.0026366650117681245},{"x":-0.4002107478167082,"y":0.045509572021663225},{"x":-0.10913563696573718,"y":0.07402479878590029},{"x":0.04369081735447718,"y":0.0016239406671226045},{"x":0.6924719651167444,"y":0.39504170234141794},{"x":-0.9761634916328416,"y":0.44020653710985247},{"x":0.08878965103526967,"y":-0.24273863437836973},{"x":-0.8291586104800757,"y":-0.16387861566433687},{"x":-0.20492943086255078,"y":-0.03189337725630782},{"x":0.6510757585624781,"y":-0.380191173250516},{"x":0.23756017117337933,"y":-0.1933972508682821},{"x":0.6047181162080963,"y":0.04209235554559913},{"x":0.9220210487130001,"y":0.1210327259073729},{"x":-0.963034887975784,"y":0.3976021921103382},{"x":0.631448904865537,"y":0.20672433950208643},{"x":0.3717452081233314,"y":0.21595627940693043},{"x":-0.495414747070138,"y":0.31730465166709176},{"x":-0.6827259366040065,"y":0.287025410422973},{"x":0.982625761496451,"y":0.3797948954533657},{"x":-0.8887956499270744,"y":0.29108580526215067},{"x":0.661138413031315,"y":0.12998305549364114},{"x":0.10153672569883651,"y":-0.05390905944926491},{"x":0.8923018732262775,"y":0.4659849457542762},{"x":-0.9338790195149619,"y":0.16957870540508935},{"x":-0.9944231105782488,"y":0.04478044167108369},{"x":-0.4375523085438427,"y":-0.25454316394891274},{"x":0.973450383459083,"y":0.04848679585384719},{"x":0.8258826814622329,"y":-0.009993915927546906},{"x":-0.6485625993385368,"y":-0.09026199630103712},{"x":0.12976860256588638,"y":0.1184150090291509},{"x":-0.02946049162386572,"y":0.09232176218584276},{"x":-0.9563360918454231,"y":0.3963888943859431},{"x":0.8393119737563969,"y":0.06773827768548327},{"x":-0.5081043564812137,"y":-0.27517140059493356},{"x":0.6688920958852264,"y":-0.1964873881015512},{"x":-0.12183731908886847,"y":0.16271153689711776},{"x":0.6479707060049175,"y":0.19760808376089398},{"x":-0.6101935195818907,"y":-0.25137911703790744},{"x":0.9080864393707944,"y":-0.08720435511164115},{"x":0.457762015316586,"y":-0.18643504934688573},{"x":-0.4913266952541104,"y":-0.33163547254319137},{"x":-0.731515776752887,"y":-0.22073006320709987},{"x":0.19462659305529262,"y":-0.18279045284798065},{"x":-0.8884204525422285,"y":0.1060517587612302},{"x":-0.5459705990281851,"y":-0.18491941937305184},{"x":-0.3455112440776867,"y":-0.16837046732459532},{"x":0.4109600254394435,"y":0.1922924109646039},{"x":0.1144810433205324,"y":0.39411280602742776},{"x":0.2326142567852831,"y":0.1783856292936202},{"x":0.029632753807918178,"y":0.09484924493910495},{"x":0.7486597769221766,"y":0.05929780400055259},{"x":-0.7503831912926222,"y":-0.14363111143215998},{"x":-0.8623069676335144,"y":0.3885719282130238},{"x":-0.1869365047964674,"y":-0.14441107546890633},{"x":-0.6080940451114859,"y":-0.12515194683387554},{"x":-0.011404227288002234,"y":0.005022585431714924},{"x":0.49470592048679024,"y":0.2941964049495568},{"x":-0.2417257867307039,"y":-0.3504629899069932},{"x":-0.8689175762687444,"y":0.2032024090659565},{"x":-0.31500769871985973,"y":-0.24312263761592856},{"x":0.5661029887117344,"y":-0.029807165621048032},{"x":0.9760287436531234,"y":0.29353024807332984},{"x":-0.4334323811149498,"y":-0.44352023359760523},{"x":0.9935046265588738,"y":0.4505513612641616},{"x":0.5206797769779291,"y":-0.21981828997832514},{"x":-0.5648616215940333,"y":-0.3001325541592593},{"x":0.4023948936688655,"y":0.05288391225823906},{"x":0.6777982663339357,"y":-0.16419704764269158},{"x":0.27854682159184946,"y":-0.1776798922587306},{"x":-0.7103220649932044,"y":-0.07812214172222541},{"x":0.46698495537804574,"y":0.09627478103485149},{"x":0.32114096985729246,"y":0.1376557338563019},{"x":0.686887723598669,"y":0.1355664436502336},{"x":-0.6287317969541817,"y":0.1777643879268037},{"x":-0.5889612197484823,"y":-0.05072347865871332},{"x":0.5949776037810741,"y":-0.07779610973849027},{"x":0.28368710913711126,"y":0.11428995841421673},{"x":-0.5756592413098828,"y":-0.2163464350939276},{"x":-0.969592458411488,"y":0.22312231330617857},{"x":-0.6972932298544896,"y":-0.17820050233446005},{"x":-0.3442341456737573,"y":-0.27970678962206136},{"x":0.5110713013380584,"y":0.003466495326925172},{"x":0.8185837927342167,"y":0.27238082915218953},{"x":-0.7271307696221766,"y":0.059054488772490324},{"x":0.14490147119958854,"y":0.16301071986756174},{"x":-0.08266752024614799,"y":-0.05389380777312461},{"x":0.7565561546929382,"y":-0.016103648474507064},{"x":-0.25856347730182316,"y":0.029517067414144435},{"x":-0.7677874184488159,"y":-0.0796257016478262},{"x":-0.8499567181960993,"y":0.09766912564662082},{"x":-0.774024124548205,"y":-0.41966563390761413},{"x":-0.2529825700443939,"y":-0.643393482931525},{"x":-0.5242407445076108,"y":-0.0971954050627024},{"x":0.06783215637273973,"y":-0.15693429132051412},{"x":0.5319708141488754,"y":0.2922156031651046},{"x":-0.47831382352413815,"y":-0.36468837850009483},{"x":0.7401568268919229,"y":0.2054856676173225},{"x":0.5539646276602367,"y":-0.033398169459939536},{"x":0.8410467294219254,"y":0.27068469563970293},{"x":-0.13400477314539166,"y":0.07889147998125742},{"x":0.4709418540143989,"y":0.10714942808638249},{"x":-0.2667536663163299,"y":0.07681780486233203},{"x":0.5052033463686756,"y":0.0030638938226796395},{"x":0.2233136177976692,"y":0.0352941187036894},{"x":0.5742097184322713,"y":-0.1704991118503878},{"x":0.06574205968863253,"y":0.03440664645508324},{"x":0.6150724849238908,"y":0.2112100117607155},{"x":-0.16289841091074017,"y":-0.07444424554406297},{"x":-0.4572662431034033,"y":0.04910789731376758},{"x":0.6142223882232767,"y":0.1942630963379648},{"x":-0.2941888140730531,"y":-0.5191764888682222},{"x":-0.46126799317290695,"y":-0.2676209711353228},{"x":-0.15260411206738314,"y":-0.0760196215530311},{"x":0.004573669664092687,"y":0.5372751536666753},{"x":0.24059508285444836,"y":-0.17402953167006358},{"x":-0.5171395302343362,"y":-0.20518570060934804},{"x":0.7199694982395859,"y":0.15932657095986572},{"x":0.589547411330495,"y":0.04534252340295075},{"x":-0.8069491405643643,"y":-0.19341960951039583},{"x":-0.8039264272659868,"y":0.2894627180931399},{"x":0.47708418976728695,"y":0.015772718236771295},{"x":0.8978024889415671,"y":0.1661074955600005},{"x":0.4848289183829282,"y":0.24084805364861583},{"x":-0.45592142583306516,"y":0.07939380001399185},{"x":-0.09221476875631432,"y":0.03936288395212434},{"x":-0.03859779280053681,"y":-0.010216590307432851},{"x":-0.07672234949848429,"y":-0.40709229557684035},{"x":0.4168346320674763,"y":0.11118671939640395},{"x":0.3638545863865308,"y":-0.04850961030688347},{"x":0.03563773977275022,"y":-0.011554835005380652},{"x":0.9610852248100918,"y":0.5011039920972523},{"x":-0.008070930158894974,"y":0.20461821475396005},{"x":-0.27916421947252434,"y":-0.1331844586789121},{"x":0.4880556658375073,"y":0.018308366245813532},{"x":-0.5353405653565191,"y":-0.3681085514471806},{"x":-0.7814327969796421,"y":-0.2839873206068831},{"x":0.9451623835809015,"y":0.5488650380277168},{"x":-0.6388558761418568,"y":-0.21116207063669695},{"x":0.26445437906176966,"y":-0.45588706750856817},{"x":0.21372970432048452,"y":-0.134529418934911},{"x":-0.7486007838249716,"y":0.1149842777395831},{"x":-0.884249793315002,"y":0.21261030524280472},{"x":-0.38052591862667673,"y":-0.32729873542928023},{"x":0.18487445088964444,"y":0.2188233598379682},{"x":0.3817006457866358,"y":0.11057359121163567},{"x":0.05391778810208861,"y":0.1474757445662067},{"x":0.008333959366570508,"y":0.023917799288131338},{"x":-0.03147712221338776,"y":-0.17598049655848927},{"x":0.3482248013985658,"y":-0.07601040211201733},{"x":-0.16858820810296052,"y":-0.17434697479840738},{"x":-0.40895434705249584,"y":0.00511059860177801},{"x":0.517215781037767,"y":0.18352629019288258},{"x":-0.4174191532301415,"y":-0.2415105395975084},{"x":-0.7018690045440609,"y":-0.05714295030417166},{"x":-0.652968069519735,"y":-0.24197129883498403},{"x":-0.790420979417478,"y":0.0404866792516488},{"x":-0.9832081911479787,"y":0.41169236418226907},{"x":0.13551165002696905,"y":0.21616825420909117},{"x":-0.6742612284181876,"y":-0.32545669298592295},{"x":0.4205694962196665,"y":0.33937611512345117},{"x":-0.19310396842725208,"y":-0.17227995613603858},{"x":-0.5565369459568926,"y":-0.10954362941608668},{"x":-0.09649669147253034,"y":-0.24731742353274366},{"x":-0.30258713396227027,"y":-0.09434010664385034},{"x":0.9524179874902527,"y":0.049328310980888745},{"x":0.44815459177493566,"y":-0.29685259633793204},{"x":0.057483525408295766,"y":-0.24748012344488798},{"x":-0.4124723058972238,"y":-0.3095240183086939},{"x":0.2991333710457038,"y":-0.024427314592211806},{"x":-0.570317222348514,"y":-0.10391066557832204},{"x":0.09516594759487754,"y":0.13329940667289045},{"x":0.3011365576360423,"y":0.3244696873711593},{"x":-0.9988477038406747,"y":0.6925936972093799},{"x":-0.26370357879237216,"y":-0.16009658131507618},{"x":-0.9274536348181047,"y":0.3037760702778273},{"x":-0.5920871969212046,"y":-0.054317339538930184},{"x":-0.1808387098532108,"y":0.12597887998864365},{"x":0.07221477367799128,"y":0.0643906347988831},{"x":0.8895471446497469,"y":0.2829287170232955},{"x":-0.6898658655528024,"y":-0.43541813016956205},{"x":-0.5121885900456699,"y":-0.28690829185517086},{"x":-0.09742885267975096,"y":-0.5243708591888628},{"x":-0.33249526804722357,"y":-0.40262119838277394},{"x":-0.5867324056548782,"y":-0.1817833546749067},{"x":0.948995802107417,"y":0.8681028699586042},{"x":-0.9322092231242066,"y":1.0461300366434032},{"x":0.03294123453387878,"y":0.07056173800546717},{"x":-0.4481108747742613,"y":-0.33775792187739384},{"x":-0.7420787756302977,"y":-0.09291788966474804},{"x":0.10749232248353967,"y":-0.1723148785760666},{"x":-0.280962136161194,"y":0.5499782420284489},{"x":0.010953617504797378,"y":0.11579586259375554},{"x":0.7111469881384442,"y":0.13416334157180881},{"x":0.1259197718941932,"y":-0.063634747348397},{"x":0.7515932300547497,"y":0.09107804016995774},{"x":-0.7154090150109182,"y":-0.4341093853031986},{"x":0.16100920521780523,"y":0.04158305940984759},{"x":-0.8925628205755614,"y":0.182688886107585},{"x":-0.7614709287056426,"y":-0.03621151770021222},{"x":-0.22288723333039553,"y":-0.21098878978147045},{"x":0.9582460330276497,"y":0.03258978620746833},{"x":-0.2859401669648677,"y":-0.0995991970225836},{"x":-0.33360607156122274,"y":-0.14831411999256536},{"x":-0.9079873679500694,"y":0.5029272247616328},{"x":-0.6691423546872008,"y":-0.1884734123044278},{"x":0.2835482834901598,"y":-0.12395007453896019},{"x":0.1328480579155994,"y":0.11293744835032479},{"x":0.797029419508341,"y":0.06319340549050488},{"x":0.015998618048275626,"y":-0.05777817834179898},{"x":-0.450845897991119,"y":-0.19858810294772417},{"x":0.5986700965199704,"y":-0.02939002378627887},{"x":0.2547196797196894,"y":0.09249660876950835},{"x":-0.6192582769479463,"y":-0.33638856718145904},{"x":0.4309421276066111,"y":-0.09140563686074726},{"x":0.6708121534499442,"y":0.3136691279333845},{"x":0.63572348490368,"y":-0.15995330334552085},{"x":0.9151217428712471,"y":0.16920421274886813},{"x":-0.8199597060943766,"y":0.126546671089132},{"x":0.33455145933879193,"y":0.011369628713578208},{"x":0.4542802891819012,"y":0.010485908971887014},{"x":-0.22916129765285218,"y":-0.1956392625623264},{"x":-0.20906628323178128,"y":-0.4729772576562468},{"x":-0.16484836480434892,"y":-0.0480644673808775},{"x":0.6288677640264239,"y":0.17201229235902984},{"x":0.20840286427858976,"y":0.12308247639960669},{"x":-0.10108813907738347,"y":-0.1033710936449613},{"x":0.2592536594750224,"y":0.1454147858411076},{"x":0.08266047964942756,"y":0.1313810695746637},{"x":0.3532523937887252,"y":-0.32943401483877727},{"x":0.4095748156586395,"y":-0.4010227959742824},{"x":0.5826191410692323,"y":-0.1140921027267316},{"x":-0.3782808855867918,"y":0.010353947824616294},{"x":0.15101794376282587,"y":0.01706257184073667},{"x":0.1984121669982465,"y":-0.022931883760812625},{"x":0.7740329890219677,"y":0.1272774439547143},{"x":-0.06431213237496329,"y":0.08603743229358196},{"x":0.9676534658675068,"y":0.4423314815743753},{"x":0.5631757438902513,"y":0.3139436673569731},{"x":-0.32439666184192606,"y":-0.19175997444742177},{"x":0.7207917507795061,"y":-0.04713853939026674},{"x":0.4438707572094311,"y":-0.03222986809427535},{"x":-0.1472553507795848,"y":0.06538540228479872},{"x":0.6840475053779652,"y":-0.14194239597157393},{"x":-0.7068012883902535,"y":-0.15892326589214015},{"x":-0.4738107919272848,"y":0.15675265464109223}], // example data for model1
    testdata: [{"x":-0.002472150906454418,"y":-0.39229969538479936},{"x":-0.17197778965300034,"y":-0.025847643635440155},{"x":-0.44026607883418106,"y":-0.20623425940569828},{"x":0.20198911957468318,"y":0.1345101167312001},{"x":-0.687867434170197,"y":0.015475820738658441},{"x":-0.3742752852749823,"y":-0.2772038351960896},{"x":-0.13503978233558783,"y":0.21637471210928205},{"x":0.30699345015261004,"y":0.18908618109367556},{"x":-0.4212612420228088,"y":-0.06048878582201439},{"x":0.7276351397811531,"y":-0.16586406563866346},{"x":0.6409973859639865,"y":-0.1417083462085208},{"x":0.4287584637342223,"y":-0.29826529851841227},{"x":0.15612218867877853,"y":0.09970655382750956},{"x":-0.48004997207466066,"y":-0.14070110008235923},{"x":-0.23507113992618672,"y":-0.1427052889805598},{"x":-0.9383224674413654,"y":0.7060743494278482},{"x":-0.12746733891513068,"y":-0.12837209669910846},{"x":0.12042338981255296,"y":0.11101020011185696},{"x":0.8650884152471412,"y":0.12449553151352383},{"x":-0.6608038223109921,"y":-0.1812226747707932},{"x":0.7682197202095591,"y":-0.09886080371490324},{"x":0.6557937415038901,"y":-0.13468054872561996},{"x":-0.044503501180441433,"y":0.22589400396302314},{"x":0.8065652045577963,"y":-0.07560445076038744},{"x":-0.28124913344119606,"y":-0.2207694818430653},{"x":0.8547313429412734,"y":0.06023988651487375},{"x":-0.7561597900374432,"y":-0.3716082346165241},{"x":-0.365330915581251,"y":-0.48711750846776714},{"x":-0.4684843584363619,"y":-0.18093765602650905},{"x":-0.39170241797554173,"y":-0.1855917642419797},{"x":0.5752805877822798,"y":-0.2181942177881693},{"x":-0.32963888409829906,"y":-0.10220437259232909},{"x":0.5587962817124793,"y":-0.1753424641028471},{"x":-0.2294114389838874,"y":-0.40853034927724885},{"x":0.3574077765710547,"y":-0.2039834324841995},{"x":-0.1570794125657296,"y":-0.10475657525597645},{"x":0.16639467537988495,"y":-0.06249885252285448},{"x":-0.34885076058988645,"y":-0.1844471833996844},{"x":0.8567703449198478,"y":0.16926127219186046},{"x":0.849211043322516,"y":-0.0680189894196335},{"x":0.9393735879676925,"y":0.36726921383109956},{"x":0.38764762398187,"y":0.16122285553759455}] // example test data for model1
  },
  'model2':{
    data:[{"x":-0.8504765469827487,"y":0.0396603813400653},{"x":0.3234646988263361,"y":-0.15247537264826175},{"x":0.3264395015961274,"y":0.1061262275365224},{"x":0.19885881418209797,"y":0.001397103216704286},{"x":0.6483017739661339,"y":0.16621643676943035},{"x":0.24960213460505834,"y":0.02838029815541026},{"x":-0.3817278277486541,"y":-0.25616234013554073},{"x":-0.7232414638351983,"y":-0.14671562473675967},{"x":0.03144142512671947,"y":-0.08113793873157017},{"x":-0.3873296373660865,"y":-0.09001611265936761},{"x":0.5937436509428339,"y":-0.13965763595356928},{"x":0.6313254362988863,"y":-0.035800188554214366},{"x":-0.3012542405938292,"y":-0.06288981803437813},{"x":-0.21572657819321742,"y":-0.2471542782946633},{"x":0.8220053579170112,"y":0.14678416545077583},{"x":0.25255300733631314,"y":0.12947059125743404},{"x":0.0662112869817702,"y":-0.11571707268195114},{"x":-0.7779590043993581,"y":0.012727286178055908},{"x":0.8879588472820202,"y":0.2848588784960505},{"x":-0.14717337707277459,"y":-0.18038045775141848},{"x":0.6214943510756424,"y":-0.00319272376901528},{"x":-0.6928475097757844,"y":-0.20848340416586866},{"x":0.7391222273955663,"y":0.026113857550217685},{"x":0.9521394354830022,"y":0.3905582197242177},{"x":0.4115367015673941,"y":0.09794612220472837},{"x":0.2363512016242159,"y":0.06661601375361557},{"x":0.6996063026266335,"y":-0.011155104650693463},{"x":0.46154573939545435,"y":-0.07268426545222956},{"x":0.6075636989231177,"y":-0.0620043606604884},{"x":0.12949942277757417,"y":-0.012383640070016123},{"x":0.573102202241729,"y":-0.05070107762946971},{"x":-0.9988308272947047,"y":0.5414237143466814},{"x":0.9129252880054224,"y":0.46243316503300147},{"x":0.90670497625007,"y":0.30957908191250033},{"x":0.6375737143985704,"y":0.15775444564217245},{"x":0.7650622690638831,"y":0.1020072082740609},{"x":0.28170952737093224,"y":-0.04059849576156209},{"x":-0.5321862919539452,"y":-0.2614464770292029},{"x":-0.00036953425267333573,"y":-0.07474226544736928},{"x":-0.4236940134385111,"y":-0.25749058645141587},{"x":-0.9650803540446513,"y":0.2852023943722855},{"x":0.5018114354446221,"y":0.09960536300435827},{"x":-0.5474810978850787,"y":-0.25561735328308355},{"x":0.43103150684202507,"y":0.07996852311252725},{"x":0.7909926561778039,"y":0.05513347577033223},{"x":0.40626825988166193,"y":-0.09050430394311862},{"x":-0.8072727820217249,"y":0.07110787686106913},{"x":-0.4847325242149543,"y":-0.014573595019537505},{"x":0.6767291902655066,"y":-0.07809158451440615},{"x":0.4790906831953139,"y":0.11100995093338585},{"x":-0.9773846877950536,"y":0.40870678581983955},{"x":0.4563576293610827,"y":-0.10843536989088554},{"x":-0.7411714812817596,"y":-0.10945553786778536},{"x":-0.47196524487058833,"y":-0.06904701742347988},{"x":0.8500893540055262,"y":0.23622934130753598},{"x":0.14671123482792034,"y":-0.14185523997058633},{"x":-0.8825662683519178,"y":-0.0193618840731396},{"x":-0.36922721594237407,"y":-0.13949533346354384},{"x":0.6678529498384373,"y":-0.10048603592491705},{"x":-0.9005292097647333,"y":0.18192029991855632},{"x":-0.129392457204135,"y":-0.1041781548955038},{"x":0.3057899872480645,"y":0.051016644751298884},{"x":0.9396588389025179,"y":0.4451096626983668},{"x":-0.6632599438525688,"y":-0.2078106820131066},{"x":0.01336889574022351,"y":0.011571393551276569},{"x":-0.44130506458392255,"y":-0.0839303538771993},{"x":0.5517190181741035,"y":-0.07242147935342949},{"x":-0.27667278996208416,"y":-0.044523376070488355},{"x":0.39248583716148716,"y":0.022191046911953862},{"x":-0.5197055736318547,"y":-0.16881988630946887},{"x":-0.971342641060475,"y":0.4717133387889396},{"x":-0.9534055753921353,"y":0.3424516133262455},{"x":-0.908186165680819,"y":0.2842997745816517},{"x":-0.5803302742761569,"y":-0.08880815598572964},{"x":0.15254949344048174,"y":0.015590072752839858},{"x":-0.3287545428496787,"y":-0.3563207537939378},{"x":0.38785015593681127,"y":-0.1155030531823214},{"x":0.9578761208609041,"y":0.10077189483196569},{"x":0.7009202407472278,"y":-0.041283547790001665},{"x":-0.6408809164424051,"y":-0.17517465524370116},{"x":-0.1641427811073493,"y":-0.053961246402873975},{"x":0.8366566840313328,"y":-0.011110154790610699},{"x":-0.5796593788081816,"y":-0.24712935411333725},{"x":0.8848043660679237,"y":0.24044915550470908},{"x":-0.26810647644541546,"y":-0.3911302501593349},{"x":0.37448049242021547,"y":0.028130601853383716},{"x":-0.17458991630144802,"y":-0.22439935899012034},{"x":-0.07743117487725261,"y":-0.14866352697502863},{"x":0.9610024938965656,"y":0.4030932116583717},{"x":-0.5133718339606225,"y":-0.17005701907083454},{"x":-0.7736841249213796,"y":0.044313465229895606},{"x":0.7336428120732377,"y":0.05930600868210825},{"x":-0.8787770459267099,"y":0.16915246011231486},{"x":-0.4309323215900639,"y":-0.1038654699273064},{"x":0.039467122442880365,"y":-0.07211700138283665},{"x":-0.8368162343144524,"y":0.05868795111787047},{"x":0.5693111183097187,"y":0.0012161402395900875},{"x":0.11073663688072027,"y":-0.0894538166472739},{"x":-0.7497787422835384,"y":0.007469990887472702},{"x":0.6252984725136189,"y":0.08464461231448647},{"x":-0.3630536686675474,"y":-0.12005127729475647},{"x":-0.04865538036780244,"y":-0.1203010499673398},{"x":-0.3725672020534601,"y":-0.032356731479513456},{"x":-0.23684928531333688,"y":-0.05967231935276296},{"x":0.290743186134008,"y":0.10222042825176288},{"x":0.7236123644416098,"y":0.030129724952156586},{"x":-0.4394582971205952,"y":-0.1023450784042682},{"x":0.5555271620686302,"y":0.11000226016166051},{"x":0.5330280765491912,"y":-0.06313186692465549},{"x":0.8597763293407666,"y":0.2372198076473871},{"x":-0.24338710648840126,"y":-0.021469834332820764},{"x":-0.9440406552840154,"y":0.3500391863407545},{"x":-0.5993313229450948,"y":-0.14356401176873954},{"x":0.6149911035633915,"y":0.026076580481390835},{"x":0.9726677669083399,"y":0.28278006041735315},{"x":0.6159072401286172,"y":-0.05449445465445077},{"x":-0.8976435202637348,"y":0.20670277775823537},{"x":0.4369625319709399,"y":-0.13528994984033194},{"x":0.7887991985566295,"y":0.20754825467644653},{"x":-0.2908215795509905,"y":-0.1107220352660705},{"x":0.5791801968993431,"y":-0.02283321051686095},{"x":0.20844035276890888,"y":-0.23633979547583472},{"x":0.7125783387917883,"y":-0.044863740605722575},{"x":0.19255567725818604,"y":0.05658891518410472},{"x":0.9184678890241885,"y":0.20038288223337644},{"x":-0.7335969064404263,"y":0.12443895506741123},{"x":0.017007491601877315,"y":-0.0035281431680728448},{"x":-0.12078450744758909,"y":0.013587262902121575},{"x":0.3553878337445168,"y":-0.08721917162284157},{"x":0.9412648882222311,"y":0.15036287782825708},{"x":-0.535520280009355,"y":-0.17579358944020412},{"x":0.21697531132406664,"y":0.046729593822604554},{"x":-0.3244147381311881,"y":-0.12484621158684095},{"x":0.5051779505458796,"y":-0.20960566126958405},{"x":-0.37660422702650015,"y":-0.3115056624582883},{"x":0.4450305484518774,"y":0.010854420905137526},{"x":-0.5693021137850464,"y":-0.20155195992219732},{"x":-0.39479016032352077,"y":-0.06563148027299551},{"x":-0.5075967416085266,"y":-0.2943780855957111},{"x":0.7606199757785053,"y":-0.035288556849458294},{"x":-0.6083418955298289,"y":-0.06323618466408237},{"x":0.29516330753098496,"y":0.06709371554751656},{"x":-0.8450726346348795,"y":0.11639248675594957},{"x":-0.3520438661042174,"y":-0.008338632689645048},{"x":0.9815895626735152,"y":0.42964138668890267},{"x":0.4720999864746449,"y":-0.12918034041669835},{"x":-0.5206574760039042,"y":-0.14899140029970123},{"x":0.08951886242991272,"y":-0.14485678470393815},{"x":0.21274939782727165,"y":-0.03584896368843936},{"x":0.8168138524539459,"y":0.06766004013118418},{"x":-0.6754484336471434,"y":-0.2406453840980887},{"x":-0.8117285237280059,"y":0.04569529994139336},{"x":-0.6363508594490329,"y":-0.08390060501633105},{"x":-0.7828235900851896,"y":-0.12302284990602554},{"x":0.80723337174999,"y":0.22250470295786096},{"x":-0.33028376534187726,"y":-0.17135555481829176},{"x":0.6947895242661427,"y":0.07407466255733763},{"x":-0.8931498826649451,"y":0.09384078328839089},{"x":0.9218749353772104,"y":0.236915388765469},{"x":-0.9266594724214777,"y":0.23933425748897663},{"x":0.8144364529057816,"y":0.03772353465287767},{"x":0.8763257051545092,"y":-0.24126968039583235},{"x":-0.8034272260459445,"y":-0.06097924011642785},{"x":-0.5955330901669321,"y":-0.33633103484641413},{"x":0.9855595813120922,"y":0.35685309185176667},{"x":-0.1341667986007298,"y":-0.0946388723060945},{"x":0.9697595798479999,"y":0.17762773654914085},{"x":0.3007950202365586,"y":0.09570674829020702},{"x":0.27087305143859236,"y":-0.09946132713964047},{"x":0.06435426191514299,"y":0.09374286423583592},{"x":0.38491938302259443,"y":-0.07053084096362677},{"x":0.363996626006774,"y":-0.024577774719295876},{"x":0.027138380538921415,"y":-0.02512216759822152},{"x":-0.40836949053340316,"y":-0.1649619802712845},{"x":-0.8341516876739312,"y":0.06507047125942109},{"x":0.18891343286898646,"y":-0.01158089679708742},{"x":-0.033785164069964985,"y":-0.02054231705642889},{"x":0.13099739744396202,"y":-0.0076864581910665464},{"x":-0.7531189838656629,"y":-0.12516472129202635},{"x":-0.6477519728949845,"y":-0.02196762471363692},{"x":-0.41601569056292276,"y":-0.22023206394565353},{"x":-0.5947106347039068,"y":-0.2741943875102189},{"x":0.45369564037006027,"y":0.010586289259573648},{"x":-0.6841622012385228,"y":-0.12819103594687445},{"x":-0.46068907056755704,"y":-0.12434458919390401},{"x":-0.02930604752546713,"y":0.011058399128289681},{"x":-0.3066232108870078,"y":0.026697428325740374},{"x":-0.4140919428757533,"y":-0.32959671091501297},{"x":0.6742714191458663,"y":-0.008427664280444298},{"x":0.6004624085677683,"y":-0.007350851631221446},{"x":0.9012435202800361,"y":0.38000336437010906},{"x":0.2629175133807373,"y":0.12875146089069064},{"x":-0.4516284613315672,"y":-0.12644868834254924},{"x":0.417936267316212,"y":0.09955655377053006},{"x":-0.15653756872659513,"y":-0.06997056965024176},{"x":0.751690218898792,"y":0.09072653614882767},{"x":-0.42805650361987013,"y":-0.12602551974062237},{"x":-0.9486021501795262,"y":0.32243470530810453},{"x":0.49028894681224433,"y":-0.053086995374885926},{"x":-0.09644459595254261,"y":-0.13427401609412398},{"x":-0.05827532082536035,"y":0.00046127181922674815},{"x":-0.494430648159327,"y":-0.12172602796939261},{"x":0.3522397859564275,"y":0.21776779394955056},{"x":-0.9322158194260747,"y":0.24761567564159676},{"x":-0.6963171064913852,"y":0.004790569969899752},{"x":0.8776912546370454,"y":0.22291496137591504},{"x":0.3193302912072775,"y":0.09764796745229888},{"x":-0.06385792215528319,"y":-0.031331413510031644},{"x":-0.23189412743062976,"y":-0.08321485075788962},{"x":-0.5041804277044364,"y":-0.3373040091760324},{"x":0.10817021909132113,"y":0.11636284883819092},{"x":-0.26155113324750334,"y":-0.5877016331629459},{"x":-0.2855757652123915,"y":-0.13610385567578592},{"x":-0.17916877385289123,"y":-0.1468649003675387},{"x":-0.01008386846482899,"y":-0.07414139356380914},{"x":-0.46577924499753726,"y":-0.06565889838007533},{"x":-0.13514465325745081,"y":-0.08611913182497838},{"x":0.8295486388330329,"y":0.18072798472534699},{"x":0.7479241016080003,"y":0.016545518332923413},{"x":-0.3995059772782918,"y":-0.05242381038348108},{"x":-0.5501809114938149,"y":-0.18632852703861538},{"x":0.18046182914937176,"y":-0.10885438387597202},{"x":-0.4048935209913713,"y":-0.13454807582344458},{"x":-0.34606015378463717,"y":-0.15363103657747187},{"x":0.6564243314422314,"y":-0.04579705064532884},{"x":-0.8683890650153993,"y":0.1838822381893372},{"x":-0.5443045711550747,"y":-0.1570119424386203},{"x":0.4663023042450788,"y":0.06511493980478192},{"x":0.7817309055776346,"y":0.23070551714904547},{"x":0.4012230024940554,"y":0.07522637590857628},{"x":0.7192994641336918,"y":0.013507994698675037},{"x":-0.10459393861570923,"y":-0.2857205724188321},{"x":-0.44512230407369097,"y":-0.01987665559917784},{"x":-0.08277364840151485,"y":-0.14917973689253394},{"x":-0.1880735281180548,"y":-0.08751058631369789},{"x":-0.6536477788383165,"y":0.0024408200140632885},{"x":0.12234333173609073,"y":-0.006919430994228321},{"x":0.9275331109475841,"y":0.24003045276011445},{"x":-0.8711316318662919,"y":0.05593854008656779},{"x":-0.9571195580375442,"y":0.42518512753526555},{"x":0.36806499537517345,"y":-0.1071841778378622},{"x":-0.34303639897602733,"y":-0.06939109005594377},{"x":-0.35920391931289136,"y":-0.21712553975473625},{"x":-0.2500154489901558,"y":-0.10376287489951427},{"x":0.5616069908219348,"y":-0.024045131574730184},{"x":0.27991504645941,"y":-0.004133214988790692},{"x":-0.9369655870899614,"y":0.3822462149663338},{"x":0.33504483817578584,"y":-0.07000852462732744},{"x":-0.33537523771986794,"y":-0.14424712465023412},{"x":0.6849523473145038,"y":0.01053024273615733},{"x":0.22478557729601423,"y":0.07748952784236018},{"x":-0.9861897662102949,"y":0.487371670477892},{"x":0.08412629687764937,"y":-0.1238127924129055},{"x":-0.07283109307134605,"y":0.00033745166708289837},{"x":-0.2075870975017263,"y":-0.20131260502690776},{"x":-0.6203224013145127,"y":-0.037515532802059215},{"x":-0.5553051159824726,"y":-0.28076098392483384},{"x":0.6875395416461632,"y":0.045058272277468225},{"x":0.17501542673127143,"y":0.07425663143533531},{"x":-0.5871467980425759,"y":-0.3676948629547566},{"x":-0.7095464305313549,"y":-0.18658402770387844},{"x":0.25586212366381794,"y":-0.05934540552513016},{"x":-0.15178570419363244,"y":-0.02372303442968037},{"x":-0.6111013991547367,"y":0.09457859478042027},{"x":-0.9848213219856464,"y":0.2741055126368321},{"x":0.48455118323749014,"y":0.03990702872333551},{"x":0.2879167801028193,"y":0.15993512669717977},{"x":-0.794264975417572,"y":0.09314472895994737},{"x":-0.29505043606623943,"y":-0.23429271293193182},{"x":-0.06595769760615543,"y":-0.0534460756879676},{"x":0.9987339349920535,"y":0.33820883381441047},{"x":0.3438638145789339,"y":-0.033427869283230785},{"x":0.008291173397354946,"y":0.0888833915869919},{"x":0.3139326126083531,"y":0.0034565628915415253},{"x":-0.6307716277002599,"y":-0.07631094638840907},{"x":0.5433910971260345,"y":0.11321956048517703},{"x":0.20370733351871687,"y":-0.04456485127705722},{"x":-0.8633401397253189,"y":0.17221274214748428},{"x":0.04103810162545073,"y":-0.021306258998843374},{"x":0.9760291227373259,"y":0.4462655976084623},{"x":0.8455230726052494,"y":0.2800630067429054},{"x":-0.5716010067430826,"y":-0.1386703184036962},{"x":0.8600959337085335,"y":0.24862962607364153},{"x":-0.0225701599877997,"y":-0.07217428329730588},{"x":-0.8193060457895429,"y":0.18667710364669363},{"x":-0.19694161375422367,"y":-0.15862942181580875},{"x":0.15529247181008995,"y":0.20830235035256323},{"x":0.7731495143937647,"y":-0.01795429755586976},{"x":0.9493596587400954,"y":-0.00005738451233872066},{"x":0.045980216572293106,"y":-0.06363399376009699},{"x":-0.4967666161990152,"y":0.0072328386601754335},{"x":-0.1697778765125187,"y":-0.12304158540833335},{"x":-0.7605394087307891,"y":-0.07049484458638076},{"x":0.7574037961515712,"y":0.11606244804742999},{"x":-0.8400692766792538,"y":-0.007432229005289059},{"x":-0.2278296115862803,"y":-0.015350345841759996},{"x":0.833270143391577,"y":0.06297607936843859},{"x":0.34560518853347616,"y":0.047832580967117344},{"x":0.5888351770778083,"y":0.11908411331277875},{"x":-0.7279985591954864,"y":-0.09675334873078069},{"x":-0.27126240021148523,"y":-0.253943785171314},{"x":-0.9930910812016277,"y":0.5082001878250731},{"x":-0.1416722293766269,"y":-0.08669129491308133},{"x":-0.822143772708515,"y":-0.015017409729709892},{"x":0.13511817116409972,"y":-0.14128796987108186},{"x":0.7767558447942908,"y":0.3330625088046362},{"x":-0.26484855209985214,"y":-0.08716025531443969},{"x":0.05970353216278252,"y":-0.023842148173076852},{"x":0.07366622190174096,"y":-0.03971505652412277},{"x":-0.0442161065397434,"y":-0.049584665602337284},{"x":0.5468392921007368,"y":-0.13934076766720535},{"x":0.9342065872990348,"y":0.33710992840888715},{"x":0.0793275248955761,"y":0.07139903677053297},{"x":0.16337128657387695,"y":0.14229357838981826},{"x":0.3964832715573389,"y":0.10315599084872497},{"x":0.9932156668156904,"y":0.5147956336252384},{"x":-0.00840247312621602,"y":-0.06094112311336225},{"x":-0.6173513686421778,"y":-0.26602657456033896},{"x":-0.2569815274641982,"y":-0.21498403187970344},{"x":0.8707684731737897,"y":0.28816483371163165},{"x":-0.7671345251350703,"y":-0.19109068013616154},{"x":-0.7358089769828657,"y":-0.173349414475842},{"x":0.7997892718789233,"y":0.11629005917907907},{"x":-0.6036321048718509,"y":-0.11713950811333615},{"x":-0.03692773151453279,"y":-0.011066474253734401},{"x":0.33141787755076585,"y":0.06223482497615436},{"x":0.17209571348240865,"y":0.15038717469663118},{"x":0.7052226150908056,"y":-0.08796864200096294},{"x":-0.05391295403691908,"y":0.027144372195745774},{"x":0.5357542054874135,"y":-0.009084246332292687},{"x":0.8045234369592384,"y":-0.1511243172692941},{"x":0.5274299285046851,"y":0.022746687053010663},{"x":0.1440514885723924,"y":0.022979340055123667},{"x":0.867639747882438,"y":0.10462972697683544},{"x":-0.6898965312565483,"y":-0.004512824389253017},{"x":-0.8876175809446073,"y":0.16236935240411712},{"x":-0.2009162814233079,"y":-0.0918513984054736},{"x":0.3758382128628184,"y":-0.09971685600068558},{"x":-0.08627257741125102,"y":-0.19311083774394147},{"x":-0.7166435808509884,"y":-0.13758414317127848},{"x":0.11553960447513895,"y":0.11823211585888802},{"x":0.4982308016639079,"y":-0.12849840732037554},{"x":-0.48701879309399687,"y":-0.28801192909838186},{"x":-0.7036720564933605,"y":-0.12667126672201545},{"x":0.8400103636306577,"y":-0.002852775237634081},{"x":-0.019366660766305756,"y":-0.08764312448735331},{"x":-0.6687569746917829,"y":-0.060563849497912395},{"x":0.09833033600223984,"y":-0.2329448629604868},{"x":0.240066018427136,"y":-0.02251251887348199},{"x":-0.24556345046441141,"y":0.09269574142637478},{"x":-0.6583565966226501,"y":-0.23140741303314616},{"x":0.6616488169759666,"y":-0.038314078706784056},{"x":-0.21317548520022137,"y":-0.05019465532592867},{"x":0.7420225557027832,"y":0.06556860512568195},{"x":-0.8566502908070299,"y":0.024070829400533608},{"x":-0.8294015767939927,"y":0.1355558953370548},{"x":0.1690955758704685,"y":-0.007055325966575145},{"x":0.426353077118507,"y":-0.04606388691871383},{"x":-0.11844208252977374,"y":-0.05675532778934817},{"x":-0.09402614353988441,"y":-0.14419986723580513},{"x":-0.7892289028783833,"y":0.09168449859810274},{"x":-0.9246489778632899,"y":0.3171635848446727},{"x":0.10496020824265544,"y":0.1116077584883315}],
    testdata: [{"x":-0.11198211648022822,"y":0.04078812413260999},{"x":0.6420612307836995,"y":-0.07716979753658473},{"x":-0.627205913519444,"y":-0.20955859713220543},{"x":-0.7598775535339596,"y":-0.020870881625132905},{"x":-0.673513346101807,"y":-0.318978722608142},{"x":-0.7996358426613487,"y":0.15186850853928127},{"x":0.2259429058205237,"y":0.00009292891991039996},{"x":0.4859972784671403,"y":0.045441914680959115},{"x":-0.9109917352650495,"y":0.28578634993987123},{"x":0.23203370135588663,"y":-0.030737883538801223},{"x":-0.4798778547207352,"y":-0.04006608665951389},{"x":-0.915441783870084,"y":0.3378263016482487},{"x":-0.19086531434515414,"y":0.012228278987019156},{"x":0.022989196854607437,"y":0.018053979285921534},{"x":0.5226609299883805,"y":-0.12773591322064276},{"x":0.5831103855292991,"y":0.13327618218485918},{"x":0.6525569700000827,"y":-0.06904896072790445},{"x":0.004412133416963644,"y":-0.12151882705908328},{"x":-0.7144371601230275,"y":-0.10990943269122481},{"x":-0.3103779134994861,"y":-0.27250385976658575},{"x":0.26629330805953,"y":-0.013461750078005659},{"x":0.8911921575285473,"y":0.32859908838374874},{"x":0.8953929472848663,"y":0.09542439835263672},{"x":0.09035520849028653,"y":0.02576875487726614},{"x":0.5996538521396673,"y":-0.06803782161433938},{"x":0.7265768766260126,"y":-0.04006128926575127},{"x":0.053423000382374373,"y":0.040822619099742524},{"x":0.5112129457448021,"y":-0.06192023246308129},{"x":-0.31659827906808846,"y":-0.31670649368716575},{"x":-0.2801003447222504,"y":-0.07291835640080799},{"x":-0.18330087869822345,"y":-0.07527683751220922},{"x":0.517013403059642,"y":-0.04131677313779871},{"x":-0.22466517940920822,"y":-0.1810701085269641},{"x":0.4223761272925564,"y":0.020508798727707035},{"x":-0.5636337010425752,"y":-0.11070957819638046},{"x":-0.9616187318614684,"y":0.33599119972458086},{"x":-0.5288192170781544,"y":-0.2100478078980662},{"x":-0.45810803638730235,"y":-0.142041834651767},{"x":-0.10800187585152843,"y":-0.0092636344425693},{"x":0.4437438705903359,"y":-0.05482291229283393}]  
  },
  'model3':{
    data:[{"x":-0.7424888478735002,"y":-0.08141805478211094},{"x":-0.047212756541447964,"y":-0.05675148890072814},{"x":-0.9649711407816506,"y":0.4005310417936649},{"x":0.32872303233436373,"y":0.003360672802962851},{"x":0.8076940084473928,"y":0.1225520564983626},{"x":0.6559836259498317,"y":0.09052465865658438},{"x":-0.4459013385723712,"y":-0.21831564854909158},{"x":0.6922901721666495,"y":0.03621292861301948},{"x":-0.45650582387186284,"y":-0.1585942716790096},{"x":-0.7602804614468417,"y":-0.03572035778024845},{"x":0.11626171020319159,"y":-0.005867914565535042},{"x":0.732481898795436,"y":0.06834168634401833},{"x":-0.9697409109035411,"y":0.3983639461512049},{"x":0.5762890278747836,"y":-0.012680067418339274}],
    testdata: [{"x":0.40903778265668067,"y":-0.0004756804395573055},{"x":-0.16289841091074017,"y":-0.07444424554406297},{"x":0.06574205968863253,"y":0.03440664645508324},{"x":-0.8797288490744982,"y":0.1442141323501804},{"x":-0.32600567346673803,"y":-0.14942386649317124},{"x":0.5327248234813343,"y":0.034368094397254134},{"x":-0.9340656053679752,"y":0.2773532824167149},{"x":0.8281432406480166,"y":0.11373438579921308},{"x":-0.5867324056548782,"y":-0.1817833546749067},{"x":0.05848951077406781,"y":-0.03236827980098315},{"x":-0.33360607156122274,"y":-0.14831411999256536},{"x":0.4736439016603548,"y":-0.006704414498170258},{"x":-0.48278534237740267,"y":-0.16880772243309974},{"x":0.18849953381741308,"y":0.02968856748994852}]
    }
};

async function loadExampleData(exampleName) {
  // Get the checkbox
  const checkbox = document.getElementById('example-data-checkbox');

  // Check if the checkbox is checked
  if (checkbox.checked) {
    // Get the selected dataset from the examples
    const selectedDataset = datasets[exampleName];
    if (selectedDataset) {
      // Load the data and testdata
      data = selectedDataset.data;
      testdata = selectedDataset.testdata;
      console.log('Example datasets loaded successfully');
    } else {
      console.error('Example dataset not found:', exampleName);
    }
  }
}

document.getElementById('example-btn').addEventListener('click', async function() {
  // Get the selected example from the dropdown
  const selectedExample = document.getElementById('model-select').value;

  // Load the selected example model
  await loadExampleModel(selectedExample);

  // Load the example datasets
  await loadExampleData(selectedExample);

  displayModel(model); // Display the loaded model
  console.log('Model display updated'); // Confirm that the model display has been updated

  // Clear the training section
  const predictionsSection = document.getElementById('predictionsdisplay');
  if (predictionsSection) {
    predictionsSection.innerHTML = '';
    console.log('prediction section cleared due to new model');
  }
  const resultsSection = document.getElementById('resultsdisplay');
  if (resultsSection) {
    resultsSection.innerHTML = '';
    console.log('result section cleared due to new model');
  }

  // Clear the test section
  const testSection = document.getElementById('test-predictionsdisplay');
  if (testSection) {
    testSection.innerHTML = '';
    console.log('test graph section cleared due to new split');
  }
  const tresultsSection = document.getElementById('test-resultsdisplay');
  if (tresultsSection) {
    tresultsSection.innerHTML = '';
    console.log('test result section cleared due to new split');
  }
});

async function loadExampleModel(exampleName) {
  try {
    // Get the selected model from the examples
    const selectedModel = models[exampleName];
    if (selectedModel) {
      // Create a Blob from the Uint8Array
      const blob = new Blob([selectedModel.bin], {type: 'application/octet-stream'});

      // Create a new File object from the Blob
      const binFile = new File([blob], 'model.bin');

      // Modify the weightsManifest in the model's JSON data
      selectedModel.json.weightsManifest[0].paths = ['model.bin'];

      // Create a Blob from the JSON
      const jsonBlob = new Blob([JSON.stringify(selectedModel.json)], {type: 'application/json'});

      // Create a new File object from the Blob
      const jsonFile = new File([jsonBlob], 'model.json');

      // Load the model
      model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, binFile]));
      console.log('Model loaded successfully');
      return model;
    } else {
      console.error('Example not found:', exampleName);
    }
  } catch (error) {
    console.error('Error loading model:', error);
  }
}

async function loadExampleData(exampleName) {
  // Get the selected dataset from the examples
  const selectedDataset = datasets[exampleName];
  if (selectedDataset) {
    // Load the data
    data = selectedDataset.data;
    console.log('Data loaded successfully');

    // Prepare the data values
    const dataValues = data.map(d => ({
      x: d.x,
      y: d.y,
    }));

    // Check if testdata exists
    if (selectedDataset.testdata && selectedDataset.testdata.length > 0) {
      // Load the testdata
      testdata = selectedDataset.testdata;
      console.log('Test data loaded successfully');

      // Prepare the testdata values
      const testdataValues = testdata.map(d => ({
        x: d.x,
        y: d.y,
      }));

      // Display both data and testdata
      displayData(dataValues, testdataValues);
    } else {
      // Display only data
      displayData(dataValues);
    }
  } else {
    console.error('Example dataset not found:', exampleName);
  }
}