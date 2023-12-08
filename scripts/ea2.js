// JavaScript source code
console.log('Hello TensorFlow');


/************************************************************** run ************************************************************************************************************************/
let data, model, tensorData;

async function run() {
  data = await loadData();
  model = createModelDisplay();
  const trainResults = await train(model, data);
  predict(trainResults);
}

async function loadData() {
  data = await getData();
  if (!data) {
    console.error('No data returned from getData()');
    return;
  }

  const values = data.map(d => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  if (!values || !values.length) {
    console.error('Values array is empty');
    return;
  }

  // show data scatterplot
  const data_surface = document.getElementById('datadisplay');
  if (!data_surface) {
    console.error('No HTML element found with id "datadisplay"');
    return;
  }
  tfvis.render.scatterplot(
    data_surface,
    {values: values, series: ['original']},
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  );
  
  return data;
}

function createModelDisplay() {
  model = createModel();
  const model_surface = document.getElementById('modeldisplay');
  if (!model_surface) {
    console.error('No HTML element found with id "modeldisplay"');
    return;
  }
  tfvis.show.modelSummary(model_surface, model);
  
  return model;
}

async function train(model, data) {
  // Convert the data to a form we can use for training.
  tensorData = convertToTensor(data);
  const {inputs, labels} = tensorData;

  await trainModel(model, inputs, labels);
  console.log('Done Training');
  
  return {model, data, tensorData};
}

function predict({model, data, tensorData}) {
  testModel(model, data, tensorData);
}

async function saveModel(model) {
  const saveResult = await model.save('downloads://bestmodel');
  console.log(saveResult);
}

async function loadModel() {
  const model = await tf.loadLayersModel('https://robheg.github.io/modelsave/bestmodel.json');
  return model;
}


document.addEventListener('DOMContentLoaded', run);

/**************************************************************get data************************************************************************************************************************/
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
    const inputs = data.map(d => d.horsepower)
    const labels = data.map(d => d.mpg);

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
async function trainModel(model, inputs, labels) {
  // Prepare the model for training.
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });

  const batchSize = 32;
  const epochs = 50;

  const training_surface = document.getElementById('trainingdisplay');
  if (!training_surface) {
    console.error('No HTML element found with id "trainingdisplay"');
    return;
  }

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks( //show training performance
        training_surface,  
        //{ name: 'Training Performance' },
      ['loss', 'mse'],
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });
}


/************************************************************** test model ************************************************************************************************************************/
function testModel(model, inputData, normalizationData) {
  const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {

    const xsNorm = tf.linspace(0, 1, 100);
    const predictions = model.predict(xsNorm.reshape([100, 1]));

    const unNormXs = xsNorm
      .mul(inputMax.sub(inputMin))
      .add(inputMin);

    const unNormPreds = predictions
      .mul(labelMax.sub(labelMin))
      .add(labelMin);

    // Un-normalize the data
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });


  const predictedPoints = Array.from(xs).map((val, i) => {
    return {x: val, y: preds[i]}
  });

  const originalPoints = inputData.map(d => ({
    x: d.horsepower, y: d.mpg,
  }));

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
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  );
}


/************************************************************** html buttons ************************************************************************************************************************/
document.getElementById('data-btn').addEventListener('click', async () => {
  data = await loadData();
});

document.getElementById('model-btn').addEventListener('click', () => {
  model = createModelDisplay();
});

document.getElementById('train-btn').addEventListener('click', async () => {
  const trainResults = await train(model, data);
  model = trainResults.model;
  tensorData = trainResults.tensorData;
});

document.getElementById('predict-btn').addEventListener('click', () => {
  predict({model, data, tensorData});
});

document.getElementById('save-btn').addEventListener('click', async () => {
  await saveModel(model);
});

document.getElementById('load-btn').addEventListener('click', async () => {
  model = await loadModel();
});