$('.ui.checkbox')
  .checkbox();

var app = {}

// ********************************** Import Data **********************************

$(document).ready(function () {
  $("#uploaded-file").change(handleFileSelect);
});

const handleFileSelect = evt => {

  //make upload file ready
  $('#extract-data').removeClass('disabled');

  let file = evt.target.files[0];
  Papa.parse(file, {
    header: true,
    skipEmptyLines: true,
    complete: function (results) {
      // pass results
      app.Data = results;
      updateSelectCol(Object.keys(results.data[0]))
    }
  });
  console.log(app)
}

const updateSelectCol = arr => {
  // clear dropdowns
  $('#select-data').empty();
  $('#select-time').empty();
  $.each(arr, function (index, value) {
    $('#select-data').append($('<option/>', {
      value: value,
      text: value
    }));
    $('#select-time').append($('<option/>', {
      value: value,
      text: value
    }));
  });
}

const forwardFill = arr => {
  let temp = [];
  let lastindex = 0;
  for (var i = 0; i < arr.length; i++) {
    if (arr[i] != null) {
      lastindex = i
      break;
    }
  }

  for (var i = 0; i < arr.length; i++) {
    if (arr[i] == null) {
      arr[i] = arr[lastindex]
    }
    lastindex = i
  }
};

$("#extract-data").click(
  function () {
    console.log("g")
    let target = [],
      date = [];

    app.Data.data.forEach((element, index) => {
      target.push(element[$("#select-data").val()])
      date.push(element[$("#select-time").val()])
    });
    console.log(date);

    date = date.map(d => d.replace(' UTC', ''));
    console.log(date)

    const GRAPH = document.getElementById('graph-extracted');
    Plotly.purge(GRAPH);
    Plotly.plot(GRAPH, [{
      x: date,
      y: target,
      type: 'scatter'
    }], {
      responsive: true,
    });

    app.date = date
    app.target = target
  }
)

// ********************************** Training & Validation **********************************

function normalize(min, max) {
  var delta = max - min;
  return function (val) {
      return (val - min) / delta;
  };
}

function unnormalize(min, max){
  var delta = max - min;
  return function(val){
    return (val * delta) + min
  }
}

const prepareData = (arr, lookback, steps) => {
  console.log(arr)
  console.log($("#predict-steps").val())

  var inputs = arr.map(d => parseFloat(d))

  app.min = Math.min(...inputs)
  app.max = Math.max(...inputs)

  inputs = inputs.map(normalize(app.min, app.max))

  let x = []
  let y = []
  console.log(inputs)
  console.log(x, y)
  for (var i = 0; i < (inputs.length - lookback - steps); i++) {
    x.push(inputs.slice(i, i + lookback))
    y.push(inputs.slice(i + lookback, i + lookback + steps))
  };

  let preds_input = inputs.slice(-21,)

  app.xlength = x[0].length
  app.ylength = y[0].length

  return tf.tidy(() => {

    const input = tf.tensor(x, [x.length, 7, 3])

    const inputForecast = tf.tensor(preds_input, [1, 7, 3])
    const output = tf.tensor2d(y, [y.length, y[0].length])

    return {
      "inputTensor": input,
      "outputTensor": output,
      "forecastTensor": inputForecast
    }
  })
};


function createModel() {
  // Create a sequential model
  const model = tf.sequential();

  // Add a single lstm layer
  model.add(tf.layers.lstm({
    inputShape: [7, 3],
    units: 64
  }));

  // Add an output layer
  model.add(tf.layers.dense({units: 32, activation: "relu"}))
  model.add(tf.layers.dense({
    units: app.ylength,
    activation: 'sigmoid'
  }));
  return model;
}

const trainModel = async (model, inputs, labels) => {
  // Prepare the model for training.  
  const learningRate = 4e-3;
  //const optimizer = tf.train.rmsprop(learningRate);
  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse', 'mape'],
  });
  Plotly.purge("graph-loss");
  Plotly.purge("graph-mape");
  Plotly.plot('graph-loss', [{
    y: [],
    mode: 'lines',
    name: "Train Loss",
    line: {color: '#80CAF6'}
  }, {
    y: [],
    name: "Validation Loss",
    mode: 'lines',
    line: {color: '#DF56F1'}
  }]);

  Plotly.plot('graph-mape', [ {
    y: [],
    name: "Validation MAPE",
    mode: 'lines',
    line: {color: '#DF56F1'}
  }]);

  console.log(inputs.shape, labels.shape)
  const fitOutput = await model.fit(
    inputs, labels, {
      batchSize: parseInt($("#batch-size").val()),
      epochs: parseInt($("#epochs").val()), // change
      validationSplit: parseFloat($("#val-size").val()), //change
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          Plotly.extendTraces('graph-loss', {
            y: [[logs.loss], [logs.val_loss]]
          }, [0, 1])
          Plotly.extendTraces('graph-mape', {
            y: [[logs.val_mape]]
          }, [0])
        }
      }
    });
  inputs.dispose();
  labels.dispose();
  console.log(fitOutput);
  app.Model = model;
  return model
}

const Forecast = (model, inputData) => {
  return tf.tidy(() => {
    const preds = model.predict(inputData);

    return preds.dataSync();
  });
}

$("#train").click(function () {

  var lookback_steps = 21

  let tensors = prepareData(app.target, lookback_steps, parseInt($("#predict-steps").val()));
  app.forecastTensor = tensors.forecastTensor

  let model = createModel()

  app.Model = trainModel(model, tensors.inputTensor, tensors.outputTensor)
  
})


$("#forecast").click(function (){
  let preds = Forecast(app.Model, app.forecastTensor)

  console.log(preds.map(unnormalize(app.min, app.max)))
  const GRAPH_forecast = document.getElementById('graph-forecast');

  Plotly.purge(GRAPH_forecast);
  Plotly.plot(GRAPH_forecast, [{
    y: preds.map(unnormalize(app.min, app.max)),
    type: 'lines'
  }], {
    responsive: true,
  });

})