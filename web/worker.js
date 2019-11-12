
this.window = this;
if (typeof OffscreenCanvas !== 'undefined') {
    self.document = {
        createElement: () => {
            return new OffscreenCanvas(640, 480);
        }
    };
    self.HTMLVideoElement = function() {}
    self.HTMLImageElement = function() {}
    self.HTMLCanvasElement = function() {}
}
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.0.0/dist/tf.min.js');
tf.setBackend('cpu')
p_model = tf.loadLayersModel('tfjs_model/model.json')

var word_index = {};

var xhttp = new XMLHttpRequest();
xhttp.onreadystatechange = function() {
    if (this.readyState == 4){
        if (this.status == 200) {
            word_index = JSON.parse(xhttp.responseText);
        }
        else{
            console.log("Error loading word index")
        }
    }
};
xhttp.open("GET", "word_index.json", true);
xhttp.send();

onmessage = function(e) {
    c = e.data[0];
    d = e.data.slice(1);
    if (c == 'run'){
        toks = d[0];
        input = [];
        for (var i = 0; i<= toks.length-1; i++){
            index = word_index[toks[i]];
            if (index === undefined){
                index = word_index[toks[i].toLowerCase()];
            }
            if (index === undefined){
                index = 0;
            }
            input.push(index);
        }
        input_windows = []
        for (var i = 0; i<= toks.length-1; i++){
            input_window = [];
            for (var w = -2; w<= 2; w++){
                val = input[i+w]
                if (val === undefined){ val = 0; }
                input_window.push(val);
            }
            input_windows.push(input_window)
        }
        p_model.then((model)=>{
            prediction = model.predictOnBatch(tf.tensor(input_windows));
            prediction.data().then((v)=>{
                result = []
                for (var i = 0; i<= toks.length-1; i++){
                    result.push(Math.round(v[i]));
                }
                self.postMessage(['nn', toks, result])
            });
        });
        
    }
}
