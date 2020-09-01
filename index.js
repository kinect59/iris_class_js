
// Tiny TFJS train / predict example.
async function laod_and_test() {
const test_data = tf.tensor2d([[6.7, 2.5, 5.8, 1.8]]); 
  
  model = await tf.loadLayersModel('web_model/model.json').then(model => {
  
  model.summary();
  y = model.predict(tf.tensor2d([[6.7, 2.5, 5.8, 1.8]]))
  y_pred = y.argMax(axis=-1)
  document.getElementById('out').innerHTML = y_pred.dataSync()
  //document.getElementById('micro-out-div').innerText = model.predict(tf.tensor2d([[6.7, 2.5, 5.8, 1.8]])).argMax().dataSync();
  //y = model.predict(tf.zeros([1,2])) 
  //document.getElementById('out').innerHTML = y.dataSync()[0]
  });
}

laod_and_test();
