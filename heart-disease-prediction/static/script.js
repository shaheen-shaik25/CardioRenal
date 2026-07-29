// ===============================
// IntelliHeart AI
// script.js - Part 3A
// ===============================

// ----------------------------
// Sample Patient
// ----------------------------

const samplePatient = {

    age:63,
    sex:1,
    cp:3,
    trestbps:145,
    chol:233,
    fbs:1,
    restecg:0,
    thalach:150,
    exang:0,
    oldpeak:2.3,
    slope:0,
    ca:0,
    thal:1

};

// ----------------------------
// Fill Sample
// ----------------------------

document.getElementById("sampleBtn").onclick=function(){

    Object.keys(samplePatient).forEach(function(key){

        document.getElementById(key).value=samplePatient[key];

    });

};

// ----------------------------
// Draw Empty Gauge
// ----------------------------

drawGauge(0);

// ----------------------------
// Form Submit
// ----------------------------

document.getElementById("patientForm").addEventListener("submit",async function(e){

    e.preventDefault();

    showLoading();

    const data={

        age:document.getElementById("age").value,
        sex:document.getElementById("sex").value,
        cp:document.getElementById("cp").value,
        trestbps:document.getElementById("trestbps").value,
        chol:document.getElementById("chol").value,
        fbs:document.getElementById("fbs").value,
        restecg:document.getElementById("restecg").value,
        thalach:document.getElementById("thalach").value,
        exang:document.getElementById("exang").value,
        oldpeak:document.getElementById("oldpeak").value,
        slope:document.getElementById("slope").value,
        ca:document.getElementById("ca").value,
        thal:document.getElementById("thal").value

    };

    try{

        const response=await fetch("/api/predict",{

            method:"POST",

            headers:{

                "Content-Type":"application/json"

            },

            body:JSON.stringify(data)

        });

        const result = await response.json();

if (!response.ok) {
    throw new Error(result.error || "Prediction failed");
}

updateDashboard(result, data);

    }

    catch(err){

        alert("Prediction Failed");

        console.log(err);

    }

});

// ----------------------------
// Loading
// ----------------------------

function showLoading(){

document.getElementById("predictionText").innerHTML=`
<div class="text-center">

<div class="loader"></div>

<br>

Analyzing Patient...

</div>

`;

}

// ----------------------------
// Update Dashboard
// ----------------------------

// ----------------------------
// Update Dashboard
// ----------------------------

function updateDashboard(result, input) {

    const probability = Math.round(result.probability * 100);

    drawGauge(probability);

    document.getElementById("probability").innerHTML = probability + " %";

    document.getElementById("riskBar").style.width = probability + "%";

    document.getElementById("riskBar").innerHTML = probability + "%";

    if (probability < 30) {

        document.getElementById("riskBar").className = "progress-bar bg-success";

    } else if (probability < 70) {

        document.getElementById("riskBar").className = "progress-bar bg-warning";

    } else {

        document.getElementById("riskBar").className = "progress-bar bg-danger";

    }

    document.getElementById("predictionText").innerHTML = `

        <h3>${result.label}</h3>

        <p>Model : <b>${result.model_name}</b></p>

    `;

    summary(input, result, probability);

    recommend(probability);

    // Part 3B
    saveHistory(input, result, probability);

    confidenceBadge(probability);

    showToast("Prediction Completed Successfully!");

    // Part 3C
    scrollResult();

    animateCards();

}


// ----------------------------
// Plotly Gauge
// ----------------------------

function drawGauge(value){

var data=[{

type:"indicator",

mode:"gauge+number",

value:value,

number:{

suffix:"%"

},

title:{

text:"Heart Disease Risk"

},

gauge:{

axis:{

range:[0,100]

},

bar:{

color:"#dc3545"

},

steps:[

{

range:[0,30],

color:"#28a745"

},

{

range:[30,70],

color:"#ffc107"

},

{

range:[70,100],

color:"#dc3545"

}

]

}

}];

var layout={

margin:{

t:30,

b:20,

l:20,

r:20

},

paper_bgcolor:"white"

};

Plotly.newPlot("gaugeChart",data,layout,{

displayModeBar:false,

responsive:true

});

}

// ----------------------------
// Patient Summary
// ----------------------------

function summary(input,result,prob){

document.getElementById("summaryCard").innerHTML=`

<table class="table">

<tr>

<td><b>Age</b></td>

<td>${input.age}</td>

</tr>

<tr>

<td><b>Gender</b></td>

<td>${input.sex==1?"Male":"Female"}</td>

</tr>

<tr>

<td><b>Blood Pressure</b></td>

<td>${input.trestbps}</td>

</tr>

<tr>

<td><b>Cholesterol</b></td>

<td>${input.chol}</td>

</tr>

<tr>

<td><b>Prediction</b></td>

<td>${result.label}</td>

</tr>

<tr>

<td><b>Probability</b></td>

<td>${prob}%</td>

</tr>

</table>

`;

}

// ----------------------------
// Recommendation
// ----------------------------

function recommend(prob){

let html="";

if(prob<30){

html=`

<ul>

<li>Maintain healthy diet</li>

<li>Exercise regularly</li>

<li>Annual health check-up</li>

<li>Avoid smoking</li>

</ul>

`;

}

else if(prob<70){

html=`

<ul>

<li>Consult physician</li>

<li>Reduce cholesterol intake</li>

<li>Daily exercise</li>

<li>Monitor BP regularly</li>

</ul>

`;

}

else{

html=`

<ul>

<li><b>Immediate cardiologist consultation</b></li>

<li>ECG & Echocardiogram</li>

<li>Strict lifestyle modification</li>

<li>Medication as advised</li>

<li>Emergency care if symptoms worsen</li>

</ul>

`;

}

document.getElementById("recommendation").innerHTML=html;

}
// ==========================================
// IntelliHeart AI - script.js (Part 3B)
// ==========================================

// ----------------------------
// Save Prediction History
// ----------------------------
function saveHistory(input, result, probability) {

    let history = JSON.parse(localStorage.getItem("heartHistory")) || [];

    history.unshift({
        date: new Date().toLocaleString(),
        age: input.age,
        gender: input.sex == 1 ? "Male" : "Female",
        prediction: result.label,
        probability: probability
    });

    // Keep only latest 10 records
    history = history.slice(0, 10);

    localStorage.setItem("heartHistory", JSON.stringify(history));

    loadHistory();

    updateStats(history);
}

// ----------------------------
// Load Prediction History
// ----------------------------
function loadHistory() {

    const history = JSON.parse(localStorage.getItem("heartHistory")) || [];

    const tbody = document.getElementById("historyTable");

    if (!tbody) return;

    tbody.innerHTML = "";

    history.forEach((item) => {

        tbody.innerHTML += `
        <tr>
            <td>${item.date}</td>
            <td>${item.age}</td>
            <td>${item.gender}</td>
            <td>${item.prediction}</td>
            <td>${item.probability}%</td>
        </tr>
        `;

    });

}

// ----------------------------
// Dashboard Statistics
// ----------------------------
function updateStats(history) {

    const total = history.length;

    let high = 0;

    let low = 0;

    history.forEach((x) => {

        if (x.probability >= 50)
            high++;
        else
            low++;

    });

    const totalCard = document.getElementById("totalPredictions");
    const highCard = document.getElementById("highRiskCount");
    const lowCard = document.getElementById("lowRiskCount");

    if (totalCard) totalCard.innerHTML = total;
    if (highCard) highCard.innerHTML = high;
    if (lowCard) lowCard.innerHTML = low;

}

// ----------------------------
// Toast Notification
// ----------------------------
function showToast(message, type = "success") {

    const toast = document.createElement("div");

    toast.className =
        `toast align-items-center text-bg-${type} border-0 show position-fixed`;

    toast.style.bottom = "20px";
    toast.style.right = "20px";
    toast.style.zIndex = "9999";

    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>

            <button class="btn-close btn-close-white me-2 m-auto"
                onclick="this.parentElement.parentElement.remove()">
            </button>
        </div>
    `;

    document.body.appendChild(toast);

    setTimeout(() => {

        if (toast)
            toast.remove();

    }, 3500);

}

// ----------------------------
// Confidence Badge
// ----------------------------
function confidenceBadge(probability) {

    let badge = "";

    if (probability >= 80) {

        badge =
            `<span class="badge bg-danger">Very High Confidence</span>`;

    }
    else if (probability >= 60) {

        badge =
            `<span class="badge bg-warning text-dark">High Confidence</span>`;

    }
    else {

        badge =
            `<span class="badge bg-success">Moderate Confidence</span>`;

    }

    const predictionText = document.getElementById("predictionText");

    if (predictionText)
        predictionText.innerHTML += "<br><br>" + badge;

}

// ----------------------------
// Current Date
// ----------------------------
function currentDateTime() {

    return new Date().toLocaleString();

}

// ----------------------------
// Call after Prediction
// ----------------------------
// Add these lines at the END of updateDashboard()
// (after recommend(probability);)

//
// saveHistory(input, result, probability);
// confidenceBadge(probability);
// showToast("Prediction Completed Successfully!");
//

// ----------------------------
// Load History when page opens
// ----------------------------
window.addEventListener("load", function () {

    loadHistory();

    const history = JSON.parse(localStorage.getItem("heartHistory")) || [];

    updateStats(history);

});

// ======================================
// IntelliHeart AI - Part 3C
// ======================================

// ----------------------------
// Dark Mode
// ----------------------------
const themeBtn = document.getElementById("themeToggle");

if(themeBtn){

themeBtn.addEventListener("click",function(){

document.body.classList.toggle("dark-mode");

const icon=this.querySelector("i");

if(document.body.classList.contains("dark-mode")){

icon.className="fa-solid fa-sun";

localStorage.setItem("theme","dark");

}
else{

icon.className="fa-solid fa-moon";

localStorage.setItem("theme","light");

}

});

}

if(localStorage.getItem("theme")=="dark"){

document.body.classList.add("dark-mode");

if(themeBtn){

themeBtn.querySelector("i").className="fa-solid fa-sun";

}

}

// ----------------------------
// PDF Download
// ----------------------------

const downloadBtn = document.getElementById("downloadBtn");

if (downloadBtn) {

    downloadBtn.onclick = function () {

        const element = document.getElementById("report");

        const options = {
            margin: 0.5,
            filename: "Heart_Disease_Report.pdf",
            image: { type: "jpeg", quality: 1 },
            html2canvas: {
                scale: 2,
                useCORS: true
            },
            jsPDF: {
                unit: "in",
                format: "a4",
                orientation: "portrait"
            }
        };

        html2pdf()
            .set(options)
            .from(element)
            .save();

    };

}

// ----------------------------
// Reset Form
// ----------------------------

function resetDashboard(){

document.getElementById("patientForm").reset();

drawGauge(0);

document.getElementById("probability").innerHTML="0%";

document.getElementById("predictionText").innerHTML="Waiting...";

document.getElementById("summaryCard").innerHTML="No Prediction Yet";

document.getElementById("recommendation").innerHTML="Waiting for prediction...";

document.getElementById("riskBar").style.width="0%";

}

// ----------------------------
// Smooth Scroll
// ----------------------------

function scrollResult(){

window.scrollTo({

top:250,

behavior:"smooth"

});

}

// ----------------------------
// Card Animation
// ----------------------------

function animateCards(){

document.querySelectorAll(".card").forEach(function(card){

card.style.opacity="0";

card.style.transform="translateY(20px)";

setTimeout(function(){

card.style.transition="all .6s";

card.style.opacity="1";

card.style.transform="translateY(0px)";

},200);

});

}

// ----------------------------
// Update Dashboard Enhancement
// ----------------------------

// Add these lines at the END of updateDashboard()

//
// scrollResult();
//
// animateCards();
//

// ----------------------------
// Welcome Toast
// ----------------------------

window.addEventListener("load",function(){

setTimeout(function(){

showToast("Welcome to IntelliHeart AI","primary");

},800);

});