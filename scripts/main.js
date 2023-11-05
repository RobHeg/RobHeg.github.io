/*
take a look at the ml5-tutorial here: https://learn.ml5js.org/#/tutorials/hello-ml5
*/

console.log('ml5-version:',  ml5.version);

let classifier;
let img;
let canvas;

function preload() {    //called at the beginning, first
  classifier = ml5.imageClassifier('MobileNet');
  img = loadImage('https://th.bing.com/th/id/OIP.Hp7DTN8RxWUVwFIcfeb0SwHaFj?w=228&h=180&c=7&r=0&o=5&dpr=1.5&pid=1.7'); 
}

function setup() {      //called at the beginning, second
  img = resizeImg(img);
  canvas = createCanvas(img.width, img.height);
  canvas.parent('canvas-container');
  image(img, 0, 0);
  classifyImg();
}

function classifyImg(){
    classifier.classify(img, gotResult);
}

function gotResult(error, results) {
    if (error) {
        console.error(error);
    } else {
        console.log(results);

        // Set the text of the result-text element
        document.getElementById('result-text').innerHTML = 
            '<span style="color:  rgb(0,0,0);">Best guess: ' + '</span>' + results[0].label + " <br>" +
            '<span style="color:  rgb(0,0,0);">Confidence: ' + '</span>' + results[0].confidence.toFixed(2);

        // add a line break every three words of the descriptions
        let labels = results.slice(0, 3).reverse().map(r => {
            let words = r.label.split(' '); // Split the label into words
            let label = '';
            for (let i = 0; i < words.length; i++) {
                if (i > 0 && i % 3 === 0) {
                    label += '<br>'; // Add a line break every 3 words
                }
                label += words[i] + ' ';
            }
            return label;
        });

        //prepare graph visualizations
        let confidences = results.slice(0, 3).reverse().map(r => r.confidence); // Reverse the results (best first)
        let colors = ['rgb(237,240,243)', 'rgb(237,240,243)', 'rgb(37,80,83)']; // Use a different color for the first bar
        let data = [{
            y: labels,
            x: confidences,
            type: 'bar',
            orientation: 'h', // This makes the chart horizontal
            text: confidences.map(c => c.toFixed(2)), // Convert confidences to strings with 2 decimal places
            textposition: 'auto',
            textfont: { 
                size: 20 // Use bigger fonts for the confidence values
            },
            marker: {
                color: colors,
                opacity: 0.9,
                line: {
                    color: ['rgb(8,48,107)', 'rgb(8,48,107)', 'rgb(255,255,255)'],
                    width: 1.5
                }
            },
            hoverinfo: 'none'
        }];

        // Create the bar chart
        Plotly.newPlot('classifier-results', data, {
            xaxis: {
                automargin: true
            },
            yaxis: {
                automargin: true,
                tickfont: { // Use bigger fonts
                    size: 20
                },
                tickpadding: 15 // Add space between the labels and the graph
            },
            margin: {
                t: 20, // Top margin
                l: 250 // Left margin
            }
        }, {
            displayModeBar: false // Hide the mode bar
        });
    }
}

/************ drag & drop pictures *************/

function handleDragOver(e) {    //called when a dragged item is over the dropzone
    e.preventDefault();         //prevents browser's default bahavior to open file in new tab
}

function handleDrop(e) {        //called when an item is dropped onto the dropzone
    e.preventDefault();         //prevents browser's default bahavior to open file in new tab
    
    for (let i = 0; i < e.dataTransfer.items.length; i++) {                 // Loop through the items that were dropped
        let item = e.dataTransfer.items[i];
        if (item.kind === 'file' && item.type.indexOf('image/') === 0) {    // Check if the item is a file and if it's an image (other than file it could be string)
            let file = item.getAsFile();
            processImage(file);    //process image
            break;             // Stop looping after finding the first image
        }
    }
}

function resizeImg(img) {
    let aspectRatio = img.width / img.height;
    let newWidth, newHeight;

    if (img.width > img.height) {
        newWidth = 500;
        newHeight = newWidth / aspectRatio;
    } else {
        newHeight = 500;
        newWidth = newHeight * aspectRatio;
    }

    resizeCanvas(newWidth, newHeight);
    img.resize(newWidth, newHeight); // resize the image

    return img;
}

function processImage(file){
    // Load and display the image
    loadImage(URL.createObjectURL(file), img => {
        //code inside here is processed once the image is fully loaded
        img = resizeImg(img);
        image(img, 0, 0);
        classifier.classify(img, gotResult);
    });
}

/***************EventListener***********************/
let dropzone = document.getElementById('dropzone');
dropzone.addEventListener('dragover', handleDragOver, false);
dropzone.addEventListener('drop', handleDrop, false);

document.getElementById('image-upload').addEventListener('change', function(e) {
    let file = e.target.files[0];
    if (file.type.indexOf('image/') === 0) {
        processImage(file);
        classifyImg();
    }
});

document.getElementById('run-classification').addEventListener('click', function() {
    document.getElementById('image-upload').click();
});


//for the clickable example images
let exampleImages = document.getElementsByClassName('example-image');
for (let i = 0; i < exampleImages.length; i++) {
    exampleImages[i].addEventListener('click', function() {
        // Create a new Blob object from the image data
        fetch(this.src)
            .then(response => response.blob())
            .then(blob => {
                // Process and classify the image
                processImage(blob);
            });
    });
}