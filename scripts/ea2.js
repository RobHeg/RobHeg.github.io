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
    json: {"modelTopology":{"class_name":"Sequential","config":{"name":"sequential_20","layers":[{"class_name":"Dense","config":{"units":8,"activation":"sigmoid","use_bias":true,"kernel_initializer":{"class_name":"VarianceScaling","config":{"scale":1,"mode":"fan_avg","distribution":"normal","seed":null}},"bias_initializer":{"class_name":"Zeros","config":{}},"kernel_regularizer":null,"bias_regularizer":null,"activity_regularizer":null,"kernel_constraint":null,"bias_constraint":null,"name":"dense_Dense69","trainable":true,"batch_input_shape":[null,1],"dtype":"float32"}},{"class_name":"Dense","config":{"units":4,"activation":"sigmoid","use_bias":true,"kernel_initializer":{"class_name":"VarianceScaling","config":{"scale":1,"mode":"fan_avg","distribution":"normal","seed":null}},"bias_initializer":{"class_name":"Zeros","config":{}},"kernel_regularizer":null,"bias_regularizer":null,"activity_regularizer":null,"kernel_constraint":null,"bias_constraint":null,"name":"dense_Dense70","trainable":true}},{"class_name":"Dense","config":{"units":3,"activation":"sigmoid","use_bias":true,"kernel_initializer":{"class_name":"VarianceScaling","config":{"scale":1,"mode":"fan_avg","distribution":"normal","seed":null}},"bias_initializer":{"class_name":"Zeros","config":{}},"kernel_regularizer":null,"bias_regularizer":null,"activity_regularizer":null,"kernel_constraint":null,"bias_constraint":null,"name":"dense_Dense71","trainable":true}},{"class_name":"Dense","config":{"units":1,"activation":"softplus","use_bias":true,"kernel_initializer":{"class_name":"VarianceScaling","config":{"scale":1,"mode":"fan_avg","distribution":"normal","seed":null}},"bias_initializer":{"class_name":"Zeros","config":{}},"kernel_regularizer":null,"bias_regularizer":null,"activity_regularizer":null,"kernel_constraint":null,"bias_constraint":null,"name":"dense_Dense72","trainable":true}}]},"keras_version":"tfjs-layers 2.0.0","backend":"tensor_flow.js"},"format":"layers-model","generatedBy":"TensorFlow.js tfjs-layers v2.0.0","convertedBy":null,"weightsManifest":[{"paths":["./bestmodel.weights.bin"],"weights":[{"name":"dense_Dense69/kernel","shape":[1,8],"dtype":"float32"},{"name":"dense_Dense69/bias","shape":[8],"dtype":"float32"},{"name":"dense_Dense70/kernel","shape":[8,4],"dtype":"float32"},{"name":"dense_Dense70/bias","shape":[4],"dtype":"float32"},{"name":"dense_Dense71/kernel","shape":[4,3],"dtype":"float32"},{"name":"dense_Dense71/bias","shape":[3],"dtype":"float32"},{"name":"dense_Dense72/kernel","shape":[3,1],"dtype":"float32"},{"name":"dense_Dense72/bias","shape":[1],"dtype":"float32"}]}]}, // replace with your JSON data
    bin: new Uint8Array([195,152,241,192,51,206,83,64,86,221,247,192,236,188,199,192,26,16,3,193,218,189,119,192,150,164,156,192,56,201,2,193,221,214,51,63,209,127,51,192,57,176,184,62,197,2,76,191,235,223,118,62,218,14,90,64,70,3,128,64,171,25,161,63,109,213,158,191,229,222,133,192,245,8,20,192,106,213,106,63,221,130,30,191,55,109,242,191,119,27,47,64,165,175,25,192,132,220,19,192,230,64,143,192,26,190,10,192,8,112,209,63,148,172,158,191,78,166,97,192,174,92,198,190,156,111,9,63,64,140,71,192,251,66,141,192,235,21,27,192,91,130,14,64,195,62,22,64,232,231,201,61,131,134,104,192,95,131,252,191,39,181,27,64,206,179,193,63,48,184,153,192,145,158,9,192,136,191,130,192,19,46,187,192,131,97,14,192,57,87,43,64,70,18,97,63,5,255,58,61,179,215,216,62,122,55,3,192,225,100,16,64,123,20,38,64,195,246,200,191,114,98,203,191,96,205,31,191,170,212,35,64,7,174,99,192,234,203,138,192,33,10,59,64,155,252,71,191,163,144,231,191,36,60,253,62,235,6,9,191,170,179,144,190,74,217,62,190,1,180,151,191,83,235,161,191,60,252,208,63,16,244,177,62]) // replace with your BIN data
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


document.getElementById('example-btn').addEventListener('click', async function() {
    // Get the selected example from the dropdown
    const selectedExample = document.getElementById('model-select').value;

    // Load the selected example model
    await loadExampleModel(selectedExample);

    displayModel(model); // Display the loaded model
    console.log('Model display updated'); // Confirm that the model display has been updated

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
    console.log('test graph section cleared due to new split');
    }
    const tresultsSection = document.getElementById('test-resultsdisplay');
    if (tresultsSection) {
    tresultsSection.innerHTML = '';
    console.log('test result section cleared due to new split');
    }

    // Update the predictions
    //predict({model, data, tensorData});
    //console.log('Predictions updated');
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